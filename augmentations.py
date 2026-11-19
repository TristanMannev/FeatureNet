import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class PercentileNormalize(A.ImageOnlyTransform):
    """
    Normalizes an image to [0, 1] based on its percentiles.
    Clips extreme outliers to prevent hot pixels from ruining the scale.
    """
    def __init__(self, lower_perc=1.0, upper_perc=99.0, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.lower_perc = lower_perc
        self.upper_perc = upper_perc

    def apply(self, img, **params):
        # Convert to float32 to prevent integer math truncation
        img = img.astype(np.float32)
        
        # If the image is 3D (H, W, C), normalize each channel independently
        if img.ndim == 3:
            for c in range(img.shape[2]):
                channel = img[:, :, c]
                p_low = np.percentile(channel, self.lower_perc)
                p_high = np.percentile(channel, self.upper_perc)
                
                # Prevent division by zero if an image is completely blank
                if p_high - p_low > 1e-8:
                    channel_clipped = np.clip(channel, p_low, p_high)
                    img[:, :, c] = (channel_clipped - p_low) / (p_high - p_low)
                else:
                    img[:, :, c] = 0.0
        # If the image is 2D grayscale (H, W)
        else: 
            p_low = np.percentile(img, self.lower_perc)
            p_high = np.percentile(img, self.upper_perc)
            
            if p_high - p_low > 1e-8:
                img_clipped = np.clip(img, p_low, p_high)
                img = (img_clipped - p_low) / (p_high - p_low)
            else:
                img = np.zeros_like(img)
                
        return img

# Define the augmentation pipeline (Updated for Albumentations v1.4.0+)
train_transform = A.Compose([
    PercentileNormalize(lower_perc=0.1, upper_perc=99.9),

    # ==========================================
    # 1. RIGID GEOMETRIC (Free Data)
    # ==========================================
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    
    # ==========================================
    # 2. NON-RIGID & SHAPE 
    # ==========================================
    A.Affine(
        scale=(0.85, 1.5), 
        translate_percent=(-0.0625, 0.0625), 
        rotate=(-45, 45), 
        p=0.5
    ),
    
    A.ElasticTransform(alpha=1, sigma=50, p=0.2),
    A.GridDistortion(p=0.2),

    # ==========================================
    # 3. PIXEL-LEVEL (Microscope Artifacts)
    # ==========================================
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    
    # FIX 3: GaussNoise renamed to GaussianNoise, and var_limit was removed
    A.GaussNoise(p=0.3, std_range=(0.1,0.2)),
    
    PercentileNormalize(lower_perc=0.1, upper_perc=99.9),
    ToTensorV2()
])