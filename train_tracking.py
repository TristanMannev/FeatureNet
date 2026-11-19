import os
from data.multi_frame_data import SpagDataModule
from augmentations import VideoAugmentation
from models.siamese_3d import SiameseUNet
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger

root_folder="/home/tristan/data/CTC_SINETRA_seg/"
data="Fluo-N2DH-GOWT1"
subfolders=["01","02","02"]

n_frames=5
min_cell_size=5
max_epochs=200



# CHANGED TO 2 3D CONVS AT THE START
#TODO: fix datamodle so its a slidingwindow



data_module=SpagDataModule(root_folder=os.path.join(root_folder, data), 
                           subfolders=subfolders, 
                           n_frames=n_frames, 
                           batch_size=1, 
             
                           augmenter=VideoAugmentation(), 
                           max_length_of_training_samples=200,
                           max_length_of_validation_samples=10,
                           max_length_of_testing_samples=10,
                           expected_min=0, 
                           expected_max=255)



model=SiameseUNet(in_channels=1, spots_channels=1, feat_channels=8, add_positional_encoding=True, contrastive_distance_threshold=1.5, max_epochs=max_epochs)


model.automatic_optimization = False
logger=TensorBoardLogger(
    save_dir=f"training_logs/tracking/{model.__class__.__name__}_aug/",
    name=data
)

# save model every 40 epochs
best_val_segmentation_loss_callback = ModelCheckpoint(filename="best_model_{epoch}", monitor="val/loss_seg", mode="min", save_top_k=1, save_last=True, save_weights_only=True)
trainer=Trainer(max_epochs=max_epochs, accelerator="cuda", logger=logger, log_every_n_steps=10, callbacks=[best_val_segmentation_loss_callback])
trainer.fit(model, data_module)