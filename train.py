import os

from models.unet import SiameseUNet
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from data.data import SpagDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from augmentations import train_transform

#SPECIFY PARAMETERS FOR TRAINING
dir_name="0.2-50.0"
max_epochs=200
feat_channels=32
add_positional_encoding=True
margin=1.5
augmenter=train_transform

num_freqs=10

root_folder=os.path.join("/home/tristan/code/SINETRA/dataset/hydra_flow", dir_name)
subfolders=["111","222","333"]

data_module=SpagDataModule(
    root_folder=root_folder,
    subfolders=subfolders,
    batch_size=1,
    max_length_of_training_samples=200,
    max_length_of_validation_samples=20,
    max_length_of_testing_samples=20,
    expected_min=0,
    expected_max=255,
    augmenter= augmenter
)


model=SiameseUNet(in_channels=1, features=[64, 128, 256], feat_channels=feat_channels, contrastive_distance_threshold=margin, spots_channels=1, main_lr=1e-4, spots_lr=1e-4, max_epochs=max_epochs, add_positional_encoding=add_positional_encoding)
model.automatic_optimization = False
logger=TensorBoardLogger(
    save_dir=f"training_logs/{model.__class__.__name__}_aug/SinePositionalEncoding",
    name=dir_name
)

# save model every 40 epochs
best_val_segmentation_loss_callback = ModelCheckpoint(filename="best_model_{epoch}", monitor="val/loss_seg", mode="min", save_top_k=1, save_last=True, save_weights_only=True)
trainer=Trainer(max_epochs=max_epochs, accelerator="cuda", logger=logger, log_every_n_steps=10, callbacks=[best_val_segmentation_loss_callback])
trainer.fit(model, data_module)