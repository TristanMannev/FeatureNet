import argparse
import os

from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from data.data import SpagDataModule
from models.unet import SiameseUNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SiameseUNet with command-line configurable parameters."
    )
    parser.add_argument(
        "--dir-name",
        type=str,
        default="Fluo-N2DH-GOWT1",
        help="Dataset folder name inside the data root.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/tristan/data/CTC_SINETRA_seg",
        help="Base folder containing the dataset directory.",
    )
    parser.add_argument(
        "--subfolders",
        type=str,
        nargs=3,
        default=["01", "02", "02"],
        help="Three subfolder identifiers used by the data module.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=400,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--feat-channels",
        type=int,
        default=32,
        help="Number of feature channels in the Siamese branch.",
    )
    parser.add_argument(
        "--add-positional-encoding",
        action="store_true",
        help="Enable positional encoding in the feature decoder.",
    )
    parser.add_argument(
        "--no-positional-encoding",
        action="store_false",
        dest="add_positional_encoding",
        help="Disable positional encoding in the feature decoder.",
    )
    parser.set_defaults(add_positional_encoding=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=20,
        help="Maximum number of training samples.",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=2,
        help="Maximum number of validation samples.",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=2,
        help="Maximum number of testing samples.",
    )
    parser.add_argument(
        "--main-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the main optimizer.",
    )
    parser.add_argument(
        "--spots-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the spots optimizer.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=40,
        help="Save a checkpoint every N epochs.",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="cuda",
        help="Lightning accelerator to use.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=10,
        help="How often to log training metrics.",
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=1.5,
        help="Margin for contrastive loss.",
    )
    parser.add_argument(
        "--reg-weight",
        type=float,
        default=0.001,
        help="Weight for the regularization term.",
    )
    parser.add_argument(
        "--transformer",
        action="store_true",
        help="Enable transformer module in the model.",
    )

    parser.add_argument(
        "--unet",
        action="store_false",
        dest="transformer",
        help="Disable transformer module in the model. Use U-Net instead.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    root_folder = os.path.join(args.data_root, args.dir_name)

    data_module = SpagDataModule(
        root_folder=root_folder,
        subfolders=args.subfolders,
        batch_size=args.batch_size,
        max_length_of_training_samples=args.train_samples,
        max_length_of_validation_samples=args.val_samples,
        max_length_of_testing_samples=args.test_samples,
    )

    model = SiameseUNet(
        in_channels=1,
        features=[64, 128, 256, 512],
        feat_channels=args.feat_channels,
        spots_channels=1,
        main_lr=args.main_lr,
        spots_lr=args.spots_lr,
        max_epochs=args.max_epochs,
        add_positional_encoding=args.add_positional_encoding,
        contrastive_distance_threshold=args.margin,
        reg_weight=args.reg_weight,
        transformer=args.transformer,

    )
    model.automatic_optimization = False

    logger = TensorBoardLogger(
        save_dir=f"training_logs/{model.__class__.__name__}/",
        name=args.dir_name,
    )

    checkpoint_callback = ModelCheckpoint(
        filename="model-{epoch:02d}",
        every_n_epochs=args.save_every,
        save_on_train_epoch_end=True,
        save_last=True,
        save_weights_only=True,
    )
    best_val_segmentation_loss_callback = ModelCheckpoint(
        filename="best_model",
        monitor="val/loss_seg",
        mode="min",
        save_top_k=1,
        save_weights_only=True,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=logger,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=[checkpoint_callback, best_val_segmentation_loss_callback],
    )
    trainer.fit(model, data_module)


if __name__ == "__main__":
    main()

#