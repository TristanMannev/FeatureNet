import time

import os
from models.unet import SiameseUNet
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from data.data import SpagDataModule
from lightning.pytorch.callbacks import ModelCheckpoint
from augmentations import train_transform
from lightning.pytorch.callbacks import ModelSummary

#SPECIFY PARAMETERS FOR TRAINING
dir_name="Fluo-N2DH-GOWT1"
max_epochs=200
feat_channels=8
add_positional_encoding=True
margin=1.5
augmenter=train_transform


root_folder=os.path.join("/home/tristan/data/CTC_SINETRA_seg", dir_name)
subfolders=["01","01","02"]

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

# CHECK FEATURES
model=SiameseUNet(in_channels=1, features=[16,32,64, 128], feat_channels=feat_channels, contrastive_distance_threshold=margin, seg_channels=1, feat_lr=1e-4, seg_lr=1e-4, max_epochs=max_epochs, add_positional_encoding=add_positional_encoding, reg_weight=0.001, transformer=True)
model.automatic_optimization = False
logger=TensorBoardLogger(
    save_dir=f"training_logs/{model.__class__.__name__}_aug/SinePositionalEncoding",
    name=dir_name
)


#print model architecture summary
model_summary = ModelSummary(max_depth=3)



best_val_segmentation_loss_callback = ModelCheckpoint(filename="best_model_{epoch}", monitor="val/loss_seg", mode="min", save_top_k=1, save_last=True, save_weights_only=True)
trainer=Trainer(max_epochs=max_epochs, accelerator="cuda", logger=logger, log_every_n_steps=10, callbacks=[model_summary, best_val_segmentation_loss_callback])
trainer.fit(model, data_module)