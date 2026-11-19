from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import distance_transform_edt
from matplotlib import image
import torch
from tqdm import tqdm
import lightning as L
from torch.utils.data import DataLoader, Dataset
import numpy as np
import tifffile
import os
import cv2
import math
import itertools
from typing import List, Union, Sequence, Optional, Generator, cast, TypeVar
import warnings
from torch.utils.data import Subset


def _check_images(images, expected_min=0, expected_max=255):
    if isinstance(images, list):
        images = np.array(images)
    if not isinstance(images, np.ndarray):
        raise ValueError(
            f"Images must be a list or numpy array, but got {type(images)}"
        )
    
    #check images are normalized to [0, 1]
    if np.max(images) > 1.0 or np.min(images) < 0.0:
        print("Found min and max values of images: ", np.min(images), np.max(images))
        print("Warning: Images are not normalized to [0, 1].Normalizing images to [0, 1] for training. Make sure this is intended.")
        if expected_min is not None and expected_max is not None:
            print("Normalizing images to [0, 1] using expected_min and expected_max: ", expected_min, expected_max)
            images = (images - expected_min) / (expected_max - expected_min)
        else:
            print("Warning: expected_min and expected_max are not provided. Normalizing images to [0, 1] using min and max of the images.")
            images = (images - np.min(images)) / (np.max(images) - np.min(images))

    return images


def relabel_mask(mask):
    """Relabel a mask to have consecutive labels starting from 1."""
    unique_labels = np.unique(mask)
    relabeled_mask = np.zeros_like(mask, dtype=np.int32)
    for new_label, old_label in enumerate(unique_labels):
        if old_label == 0:
            continue  # skip background
        relabeled_mask[mask == old_label] = new_label
    return relabeled_mask


class DefaultDataset(Dataset):
    """
    A default PyTorch Dataset for loading images and corresponding segmentation masks.
    Args:
        images (list or np.ndarray): A list or numpy array of image tensors.
        masks (list or np.ndarray): A list or numpy array of mask tensors.
        transform (callable, optional): Optional transform to be applied on the images and masks.

    Returns:
        tuple: A tuple containing the image tensor and the corresponding mask tensor.
        The image tensor has shape (C, H, W) and the mask tensor has shape (1, H, W).
    """

    def __init__(self, images, masks=None, transform=None, in_channels=1, max_expected_objects=1000, expected_min=0, expected_max=255):
        
        
        images=_check_images(images, expected_min=expected_min, expected_max=expected_max)
        
        self.images = images
        self.masks = masks
        self.transform = transform
        self.in_channels = in_channels
        self.max_expected_objects = max_expected_objects
        if self.masks is None:
            print(
                "DefaultDataset: No masks provided. The dataset will return empty masks. This should only be used for testing purposes."
            )
            self.masks = [
                np.zeros_like(self.images[i]) for i in range(len(self.images))
            ]
        else:
            
            self.masks = [relabel_mask(mask) for mask in masks]

        super().__init__()

    def _get_single_channel_item(self, idx):

        image = self.images[idx]
        mask = self.masks[idx]
        # Ensure image and mask are tensors
        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32).unsqueeze(
                0
            )  # Add channel dimension
        else:
            image = image.float().unsqueeze(0)

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        else:
            mask = mask.float().unsqueeze(0)

        # Optional transform logic
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
            return self._get_single_channel_item(idx)

class AugmentedDataset(Dataset):
    """Wrap a dataset and apply an augmenter callable to (image, mask).
    Supports augmenters that follow albumentations API (augmenter(image=..., mask=...))
    or simple callables that take/return (image, mask) as tensors.
    Preserves extra returned items (e.g. flows) from the underlying dataset.
    """

    def __init__(self, base_ds: Dataset, augmenter):
        self.base_ds = base_ds
        if not isinstance(augmenter, list):
            augmenter = [augmenter]
        self.augmenter = augmenter

    def __len__(self):
        return len(self.base_ds)

    def _to_numpy(self, img):
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy().squeeze()
        else:
            arr = np.array(img).squeeze()
        # channel-first -> channel-last for augmenters like albumentations
        return arr.copy()

    def _to_tensor(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.float()
        else:
            return torch.tensor(arr, dtype=torch.float32)

    def _apply_augmenter(self, aug, image, mask):
        try:
            result = aug(image=image, mask=mask)
        except TypeError:
            result = aug(image=self._to_numpy(image), mask=self._to_numpy(mask))

        if isinstance(result, dict):
            image = result.get("image", image)
            mask = result.get("mask", mask)
        elif isinstance(result, (list, tuple)):
            if len(result) < 2:
                raise RuntimeError(
                    "Augmenter must return at least (image, mask) or a dict with 'image' and 'mask'."
                )
            image, mask = result[0], result[1]
        else:
            raise RuntimeError(
                "Augmenter must return a tuple/list, a dict, or support in-place mutation via albumentations-style API."
            )

        if isinstance(image, np.ndarray):
            image = self._to_tensor(image)
        if isinstance(mask, np.ndarray):
            mask = self._to_tensor(mask)

        if isinstance(image, torch.Tensor) and image.ndim == 2:
            image = image.unsqueeze(0)
        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask

    def __getitem__(self, idx):
        item = self.base_ds[idx]
        # handle (image, mask) or (image, mask, extras...)
        if not isinstance(item, (list, tuple)):
            raise RuntimeError("Wrapped dataset must return tuple (image, mask[, ...])")
        image, mask, *extras = item

        for aug in self.augmenter:
            if aug is None:
                continue
            image, mask = self._apply_augmenter(aug, image, mask)

        return (image, mask, *extras)


class SegmentationDataModule(L.LightningDataModule):
    """
    A PyTorch Lightning DataModule for loading training and validation datasets for image segmentation tasks.
    Args:
        train_ds (Dataset): The training dataset.
        val_ds (Dataset): The validation dataset.
        batch_size (int): Batch size for the data loaders.
    """

    def __init__(
        self,
        train_ds=None,
        val_ds=None,
        test_ds=None,
        augmenter=None,
        batch_size=2,
        train_val_test_split=False,
        shuffle=True,
        seed=None,
    ):

        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        self.augmenter = augmenter

        self.batch_size = batch_size
        self.train_val_test_split = train_val_test_split
        self.shuffle = shuffle
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        if self.train_val_test_split:
            print(
                "SegmentationDataModuleWarning: train_val_test_split is set to True, the train_ds, val_ds and test_ds will be split from the train_ds. If you want to use custom train_ds, val_ds and test_ds, set train_val_test_split to False."
            )
            if self.train_ds is None:
                raise ValueError(
                    "train_ds must be provided when train_val_test_split is True"
                )
            if self.val_ds is not None or self.test_ds is not None:
                print(
                    "SegmentationDataModuleWarning: val_ds and test_ds will be ignored when train_val_test_split is True"
                )
            # Split the train_ds into train, val and test sets
            print(
                "Splitting the train_ds into train, val and test sets of lengths ",
                f"{int(len(self.train_ds)*0.8)}, {int(len(self.train_ds)*0.1)}, {len(self.train_ds)-int(len(self.train_ds)*0.8)-int(len(self.train_ds)*0.1)}",
            )
            self.train_ds, self.val_ds, self.test_ds = _video_dataset_split(
                self.train_ds,
                [
                    int(len(self.train_ds) * 0.8),
                    int(len(self.train_ds) * 0.1),
                    len(self.train_ds)
                    - int(len(self.train_ds) * 0.8)
                    - int(len(self.train_ds) * 0.1),
                ],
            )

        if self.augmenter is not None:
            if self.train_ds is not None:

                self.train_ds = AugmentedDataset(self.train_ds, self.augmenter)

    def train_dataloader(self):

        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=self.shuffle
        )

    def val_dataloader(self):

        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):

        return DataLoader(self.test_ds, batch_size=self.batch_size)

    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)


class SpagDataModule(SegmentationDataModule):
    """
    A PyTorch Lightning DataModule for loading training and validation datasets for image segmentation tasks.
    This is a specialized version of SegmentationDataModule for Flow Guided Temporal Averaging experiments.
    Args:
        root_folder (str): The root folder containing the dataset. Expected folder structure is:
        window_size (int): Size of the window for the FOG experiment. Default is 3.
        subfolders=["train", "val", "test"]. The train, val and test folders should contain a video.tiff file and a folder with the masks. Can be renamed to any other name or None, to ignore one or more of the splits.
        train_val_test_split (bool): If True, the train_ds, val_ds and test_ds will be split from the train_ds. If False, the train_ds, val_ds and test_ds will be used as provided.
        batch_size (int): Batch size for the data loaders.
        augmenter (callable, optional): Optional data augmentation function to be applied on the images and masks.
        seed (int, optional): Random seed for reproducibility. If None, no seed is set.

    Returns:
        A PyTorch Lightning DataModule with train, val and test datasets.
        The datasets are instances of DefaultFogDataset, which is a subclass of PyTorch Dataset.
        The images and masks are loaded from the video.tiff file and the folder,
        and the images are expected to be in the format (C, H, W) where C is the number of channels, H is the height and W is the width.
        The masks are expected to be in the format (1, H, W) where 1 is the number of channels (binary masks).
        The datasets can be used with PyTorch DataLoader for training and validation and testing.
    """

    def __init__(
        self,
        root_folder,
        subfolders=["train", "val", "test"],
        mask_dir_name="tracks",
        train_val_test_split=False,
        max_length_of_training_samples=200,
        max_length_of_validation_samples=50,
        max_length_of_testing_samples=200,
        batch_size=2,
        expected_min=None,
        expected_max=None,
        augmenter=None,
        shuffle=True,
        seed=None,
        precalculate_flow=False,
    ):
        if not os.path.exists(root_folder):
            raise ValueError(f"Root folder {root_folder} does not exist.")

        datasets = []
        train_val_or_test = 0  # 1=train, 2=val, 3=test
        for subfolder in subfolders:
            train_val_or_test += 1
            if subfolder is None:
                datasets.append(None)
                continue
            if not os.path.exists(os.path.join(root_folder, subfolder)):
                print(
                    f"FTGADataModuleWarning: {subfolder} folder does not exist in {root_folder}. Skipping {subfolder} split. Make sure this is intended."
                )
                datasets.append(None)
            else:
                video_path = os.path.join(root_folder, subfolder, "video.tiff")
                mask_path = os.path.join(root_folder, subfolder, mask_dir_name)
                if not os.path.exists(video_path):
                    raise ValueError(
                        f"Video file {video_path} does not exist in {subfolder} folder."
                    )
                if not os.path.exists(mask_path):
                    print(
                        f"FTGADataModuleWarning: Mask folder {mask_path} does not exist in {subfolder} folder. Using empty masks for {subfolder} split. Make sure this is intended."
                    )
                    masks = None
                else:
                    masks = _masks_folder_to_masks(mask_path)
                video = tifffile.imread(video_path)

                if train_val_or_test == 1:  # train
                    if (
                        max_length_of_training_samples is not None
                        and len(video) > max_length_of_training_samples
                    ):
                        print(
                            f"FogDataModuleWarning: Training video in {subfolder} is longer than max_length_of_training_samples ({len(video)}>{max_length_of_training_samples}). Truncating the dataset for faster training. Set max_length_of_training_samples to None to use the full dataset."
                        )
                        video = video[:max_length_of_training_samples]
                        if masks is not None:
                            masks = masks[:max_length_of_training_samples]
                elif train_val_or_test == 2:  # val
                    if (
                        max_length_of_validation_samples is not None
                        and len(video) > max_length_of_validation_samples
                    ):
                        print(
                            f"FogDataModuleWarning: Validation video in {subfolder} is longer than max_length_of_validation_samples ({len(video)}>{max_length_of_validation_samples}). Truncating the dataset for faster validation. Set max_length_of_validation_samples to None to use the full dataset."
                        )
                        video = video[:max_length_of_validation_samples]
                        if masks is not None:
                            masks = masks[:max_length_of_validation_samples]
                elif train_val_or_test == 3:  # test
                    if (
                        max_length_of_testing_samples is not None
                        and len(video) > max_length_of_testing_samples
                    ):
                        print(
                            f"FogDataModuleWarning: Testing video in {subfolder} is longer than max_length_of_testing_samples ({len(video)}>{max_length_of_testing_samples}). Truncating the dataset for faster testing. Set max_length_of_testing_samples to None to use the full dataset."
                        )
                        video = video[:max_length_of_testing_samples]
                        if masks is not None:
                            masks = masks[:max_length_of_testing_samples]

        
                datasets.append(DefaultDataset(video, masks, expected_min=expected_min, expected_max=expected_max))
               

        print("######## Successfully loaded datasets ########")

        print(f"Root folder: {root_folder}")
        # print(
        #     f"Loaded data with shape: {datasets[0][0][0].shape if datasets[0] is not None else 'N/A'} and masks with shape: {datasets[0][0][1].shape if datasets[0] is not None else 'N/A'}"
        # )
        print(
            f"Train dataset length: {len(datasets[0]) if datasets[0] is not None else 'N/A'}"
        )
        print(
            f"Val dataset length: {len(datasets[1]) if datasets[1] is not None else 'N/A'}"
        )
        print(
            f"Test dataset length: {len(datasets[2]) if datasets[2] is not None else 'N/A'}"
        )
        print("###############################################")
        super().__init__(
            *datasets,
            augmenter=augmenter,
            batch_size=batch_size,
            train_val_test_split=train_val_test_split,
            shuffle=shuffle,
            seed=seed,
        )


def _masks_folder_to_masks(mask_dir):
    """Load tif image masks from a folder and return them as a list of numpy arrays.
    folder should contain numbered tif images.
    """

    masks = []
    for mask_file in sorted(os.listdir(mask_dir)):
        if mask_file.endswith(".tif") or mask_file.endswith(".tiff"):
            mask_path = os.path.join(mask_dir, mask_file)
            mask = tifffile.imread(mask_path)
            masks.append(mask)
    masks = np.array(masks)
    return masks


_T = TypeVar("_T")
def _video_dataset_split(
    dataset: Dataset[_T],
    lengths: Sequence[Union[int, float]]) -> List[Subset[_T]]:
    r"""
    Splits a dataset into non-overlapping new datasets of given lengths. Does not randomize the indices before splitting, to keep the video order intact.
    If the sum of the input lengths is less than the length of the dataset,
    the last split will be smaller than the specified length.

    Modified from PyTorch's `torch.utils.data.random_split`
    """
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths: List[int] = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)  # type: ignore[arg-type]
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)  # type: ignore[arg-type]
        # add 1 to all the lengths in round-robin fashion until the remainder is 0
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                warnings.warn(
                    f"Length of split at index {i} is 0. "
                    f"This might result in an empty dataset."
                )

    # Cannot verify that dataset is Sized
    if sum(lengths) != len(dataset):  # type: ignore[arg-type]
        raise ValueError(
            "Sum of input lengths does not equal the length of the input dataset!"
        )
    indices = list(range(len(dataset)))  # type: ignore[arg-type]
    lengths = cast(Sequence[int], lengths)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(itertools.accumulate(lengths), lengths)
    ]



def _dist_for_label(mask, label):
    binary = mask == label
    if not np.any(binary):
        return label, None

    dist = distance_transform_edt(binary)
    values = dist[binary]
    max_val = np.max(values)
    if max_val <= 0:
        return label, None

    return label, (values / max_val).astype(np.float32)

from skimage.measure import regionprops


def rotating_gaussian_2d(x, y, A, x0, y0, sigma_x, sigma_y, theta):
    """Generate a 2D Gaussian distribution with rotation.
    Args:
        x (np.ndarray): The x-coordinates of the grid.
        y (np.ndarray): The y-coordinates of the grid.
        A (float): The amplitude of the Gaussian.
        x0 (float): The x-coordinate of the center of the Gaussian.
        y0 (float): The y-coordinate of the center of the Gaussian.
        sigma_x (float): The standard deviation of the Gaussian in the x direction.
        sigma_y (float): The standard deviation of the Gaussian in the y direction.
        theta (float): The rotation angle of the Gaussian in radians."""
    x0 = float(x0)
    y0 = float(y0)
    if abs(sigma_x) == 0:
        sigma_x = 1e-6
        print("RotatingGaussian2DWarning: sigma_x is zero. Make sure this is intended.")
    if abs(sigma_y) == 0:
        sigma_y = 1e-6
        print("RotatingGaussian2DWarning: sigma_y is zero. Make sure this is intended.")
    
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    return A * np.exp(-(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))


def  gaussian_ellipse_mask_from_labels(label_img):
    out = np.zeros_like(label_img, dtype=np.float32)
    coords=[]
    for region in regionprops(label_img):
        # Get ellipse parameters
        
        bbox=region.bbox
        bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col = bbox
        crow, ccol = region.centroid
        sigma_major = region.major_axis_length / 6
        sigma_minor = region.minor_axis_length / 6
        orientation = region.orientation-np.pi/2  # skimage's orientation is the angle between the major axis and the horizontal axis, measured counter-clockwise. We add pi/2 to get the angle of the major axis itself.
        mask=region.image
        id=region.label
        # Generate grid for the bounding box
        row = np.arange(0, bbox_max_row- bbox_min_row)
        col = np.arange(0, bbox_max_col - bbox_min_col)
        
        x, y = np.meshgrid(col, row)
        # Generate Gaussian values for the grid
        A = 1  # amplitude
        gaussian = rotating_gaussian_2d(x, y, A, ccol-bbox_min_col, crow-bbox_min_row, sigma_major, sigma_minor, orientation)
        #normalize gaussian to max of 1
        gaussian /= np.max(gaussian)        
        out[bbox_min_row:bbox_max_row, bbox_min_col:bbox_max_col][mask==1] = gaussian[mask==1]

        coords.append([id, crow, ccol])
    return out, np.array(coords)