import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import lightning as L
import os
import pickle
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
from itertools import chain
from torchvision.models.swin_transformer import SwinTransformerBlock



class AddCoordinates(nn.Module):
    """
    Appends normalized X and Y coordinate channels to a spatial tensor.
    This explicitly breaks translation invariance, helping the network 
    distinguish between identical objects at different locations.
    """
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r # Optional: Add a radial distance channel

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, C + 2, H, W) or (B, C + 3, H, W) if with_r=True
        """
        batch_size, _, height, width = x.size()
        
        # 1. Create linear spaces for y and x coordinates
        # We normalize them to the range [-1, 1] for stable training
        y_coords = torch.linspace(-1, 1, steps=height, device=x.device)
        x_coords = torch.linspace(-1, 1, steps=width, device=x.device)
        
        # 2. Create a grid using meshgrid
        # y_grid shape: (H, W), x_grid shape: (H, W)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 3. Expand to match batch size and add a channel dimension
        # Shape: (B, 1, H, W)
        y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # 4. Concatenate the coordinates to the input tensor
        coords = [x, y_grid, x_grid]
        
        # Optional: Add radial distance from the center
        if self.with_r:
            r_grid = torch.sqrt(torch.pow(x_grid, 2) + torch.pow(y_grid, 2))
            coords.append(r_grid)
            
        # Concatenate along the channel dimension (dim=1)
        return torch.cat(coords, dim=1)
    
class AddSinusoidalCoordinates(nn.Module):
    """
    Appends sinusoidal (sine and cosine) positional encodings to a spatial tensor.
    Unlike absolute coordinates, sinusoidal encodings allow the network to better 
    learn high-resolution, high-frequency spatial details and relative distances 
    (similar to NeRF).
    """
    def __init__(self, num_freqs=10):
        super().__init__()
        self.num_freqs = num_freqs
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            Tensor of shape (B, C + (4 * num_freqs), H, W)
        """
        batch_size, _, height, width = x.size()
        
        # 1. Create linear spaces for y and x coordinates
        # Normalized to [0, 1] to provide a consistent scale for frequencies
        y_coords = torch.linspace(0, 1, steps=height, device=x.device)
        x_coords = torch.linspace(0, 1, steps=width, device=x.device)
        
        # 2. Create a grid using meshgrid
        # y_grid shape: (H, W), x_grid shape: (H, W)
        y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 3. Generate base-2 frequency bands (NeRF style)
        # Frequencies: 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
        dim_t = torch.arange(self.num_freqs, dtype=torch.float32, device=x.device)
        freq_multiplier = 2.0 ** dim_t
        
        # 4. Multiply grids by frequencies
        # Scaled by 2*pi so the first freq (multiplier=1) is exactly 1 image period
        y_scaled = y_grid.unsqueeze(-1) * freq_multiplier * 2 * math.pi
        x_scaled = x_grid.unsqueeze(-1) * freq_multiplier * 2 * math.pi
        
        # 5. Compute Sine and Cosine
        # Permute from (H, W, num_freqs) to (num_freqs, H, W)
        pos_y_sin = y_scaled.sin().permute(2, 0, 1)
        pos_y_cos = y_scaled.cos().permute(2, 0, 1)
        pos_x_sin = x_scaled.sin().permute(2, 0, 1)
        pos_x_cos = x_scaled.cos().permute(2, 0, 1)
        
        # 6. Concatenate all positional channels
        # Shape: (4 * num_freqs, H, W)
        pos_enc = torch.cat([pos_y_sin, pos_y_cos, pos_x_sin, pos_x_cos], dim=0)
        
        # 7. Expand to match batch size
        # Shape: (B, 4 * num_freqs, H, W)
        pos_enc = pos_enc.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Concatenate to the original input tensor along the channel dimension (dim=1)
        return torch.cat([x, pos_enc], dim=1)
class UNetBlock(nn.Module):
    """
    A single block of the U-Net architecture, consisting of two convolutional layers
    followed by ReLU activations. This block is typically used in the encoder and decoder parts of the U-Net.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetBase(nn.Module):
    """
    A U-Net model for image segmentation tasks. It consists of an encoder, a bottleneck,
    and a decoder with skip connections.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        features (list): List of integers representing the number of features in each encoder block.
    """

    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.keep_embeddings = False #eather to save embeddings at different model depths 
        self.embeddings=[]
        prev_channels = in_channels
        for feat in features:
            self.encoder.append(UNetBlock(prev_channels, feat))
            prev_channels = feat

        self.bottleneck = UNetBlock(prev_channels, prev_channels * 2)
        prev_channels = prev_channels * 2

        self.up_transpose = nn.ModuleList()
        self.decoder = nn.ModuleList()
        for feat in reversed(features):
            self.up_transpose.append(
                nn.ConvTranspose2d(prev_channels, feat, kernel_size=2, stride=2)
            )
            self.decoder.append(UNetBlock(prev_channels, feat))
            prev_channels = feat

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        print(
            f"Initialized UNet with in_channels={in_channels}, out_channels={out_channels}"
        )

    def forward(self, x):

        skip_connections = []
        self.embeddings=[] #reset embeddings at each forward pass
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
            

        x = self.bottleneck(x)
        
        

        skip_connections = skip_connections[::-1]

        for idx in range(len(self.up_transpose)):
            if self.keep_embeddings:
                self.embeddings= self.embeddings + [x]
            x = self.up_transpose[idx](x)
            skip = skip_connections[idx]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx](x)

        self.embeddings= self.embeddings + [x]
        self.embeddings=self.embeddings[::-1] #reverse embeddings to have them in the order of the model (from input to output)
        return self.final_conv(x)

    def predict_step(self, batch, batch_idx):
        x = batch[0]
        return self(x)


class UnetDown(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.keep_embeddings = False #eather to save embeddings at different model depths 
        self.embeddings=[]
        prev_channels = in_channels
        for feat in features:
            self.encoder.append(UNetBlock(prev_channels, feat))
            prev_channels = feat

    def forward(self, x):

        skip_connections = []
        self.embeddings=[] #reset embeddings at each forward pass
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        return skip_connections, x


class SwinTransformerEncoder(nn.Module):
    """
    Hierarchical Swin-T encoder that exposes one skip tensor at each spatial level.

    The output layout is compatible with UnetUp:
    skips are in order from the input resolution to the finest downsampling level,
    and the bottleneck has half the spatial resolution of the last skip.
    """

    def __init__(
        self,
        in_channels=1,
        features=[64, 128, 256, 512],
        depths=None,
        window_size=7,
    ):
        super().__init__()
        if not features:
            raise ValueError("features must contain at least one channel width")

        self.features = list(features)
        if depths is None:
            depths = [2, 2, 6, 2] if len(features) == 4 else [2] * len(features)
        if len(depths) != len(features):
            raise ValueError("depths must contain one depth per encoder stage")

        self.patch_embed = nn.Conv2d(
            in_channels, features[0], kernel_size=3, stride=1, padding=1
        )

        self.stages = nn.ModuleList()
        self.transitions = nn.ModuleList()
        for stage_index, (feat, depth) in enumerate(zip(features, depths)):
            blocks = nn.ModuleList()
            for block_index in range(depth):
                blocks.append(
                    SwinTransformerBlock(
                        dim=feat,
                        num_heads=max(1, feat // 32),
                        window_size=[window_size, window_size],
                        shift_size=(
                            [0, 0] if block_index % 2 == 0 else [window_size // 2, window_size // 2]
                        ),
                        mlp_ratio=4.0,
                        dropout=0.0,
                        attention_dropout=0.0,
                        stochastic_depth_prob=0.0,
                    )
                )
            self.stages.append(blocks)

            if stage_index < len(features) - 1:
                self.transitions.append(
                    nn.Conv2d(
                        feat,
                        features[stage_index + 1],
                        kernel_size=2,
                        stride=2,
                    )
                )

        self.bottleneck_downsample = nn.Conv2d(
            features[-1],
            features[-1],
            kernel_size=2,
            stride=2,
        )

    def forward(self, x):
        x = self.patch_embed(x)
        skip_connections = []

        for stage_index, blocks in enumerate(self.stages):
            x = x.permute(0, 2, 3, 1)
            for block in blocks:
                x = block(x)
            skip_connections.append(x.permute(0, 3, 1, 2))

            if stage_index < len(self.transitions):
                x = skip_connections[-1]
                x = self.transitions[stage_index](x)

        bottleneck = self.bottleneck_downsample(skip_connections[-1])
        return skip_connections, bottleneck


class UnetUp(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        self.up_transpose = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.bottleneck = UNetBlock(features[-1], features[-1]*2)
        prev_channels = features[-1]*2
        for feat in reversed(features):
            self.up_transpose.append(
                nn.ConvTranspose2d(prev_channels, feat, kernel_size=2, stride=2)
            )
            self.decoder.append(UNetBlock(prev_channels, feat))
            prev_channels = feat

        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
    def forward(self, x, skip_connections):
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.up_transpose)):
            x = self.up_transpose[idx](x)
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx](x)
        return self.final_conv(x)
    
class UnetUpWithPositionalEncoding(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512],num_freqs=10):
        super().__init__()

        self.up_transpose = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.bottleneck = UNetBlock(features[-1], features[-1]*2)
        self.num_freqs=num_freqs
        prev_channels = features[-1]*2
        for feat in reversed(features):
            self.up_transpose.append(
                nn.ConvTranspose2d(prev_channels, feat, kernel_size=2, stride=2)
            )
            if feat==features[0]: #add positional encoding only at the last layer
                self.decoder.append(UNetBlock(prev_channels+4*self.num_freqs, feat)) 
            else:
                self.decoder.append(UNetBlock(prev_channels, feat))
            prev_channels = feat
        self.add_coordinates = AddSinusoidalCoordinates(num_freqs=self.num_freqs) #add sinusoidal positional encoding
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, skip_connections):
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.up_transpose)):
            x = self.up_transpose[idx](x)
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat((skip, x), dim=1)
            if idx ==len(self.up_transpose)-1: #add positional encoding only at the last layer
                x = self.add_coordinates(x)
            x = self.decoder[idx](x)

        return self.final_conv(x)

class SiameseUNetBase(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feat_channels=32, features=[64, 128, 256, 512], add_positional_encoding=False, transformer=True):
        super().__init__()
        self.in_channels = in_channels
        
        self.feat_channels = feat_channels
        self.out_channels = out_channels

        self.transformer = transformer
        if self.transformer:
            self.down = SwinTransformerEncoder(in_channels, features)
        else:
            self.down = UnetDown(in_channels, features)
        if add_positional_encoding:
            self.up_feat = UnetUpWithPositionalEncoding(feat_channels*out_channels, features)
        else:
            self.up_feat = UnetUp(feat_channels*out_channels, features)
        self.up_seg = UnetUp(out_channels, features)
        self.add_positional_encoding = add_positional_encoding

    def forward(self, x):
        skip_connections, bottleneck_input = self.down(x)
        seg_output = self.up_seg(bottleneck_input, skip_connections)
        feat_output = self.up_feat(bottleneck_input, skip_connections)
        return seg_output, feat_output.reshape(feat_output.shape[0], self.out_channels, self.feat_channels, feat_output.shape[2], feat_output.shape[3])
    

class UNet(L.LightningModule):
    """
    A PyTorch Lightning module for training a U-Net model for image segmentation tasks.
    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale images).
        out_channels (int): Number of output channels (e.g., 1 for binary segmentation).
        lr (float): Learning rate for the optimizer.
    """

    def __init__(self, in_channels=1, out_channels=1, lr=1e-3, max_epochs=30):

        self.save_hyperparameters()

        super().__init__()
        self.model = UNetBase(in_channels, out_channels)
        self.lr = lr
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.max_epochs = max_epochs
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, x, y)

        # Logging
        self.log("train_loss", loss)
        opt = self.trainer.optimizers[0]
        lr = opt.param_groups[0]["lr"]
        self.log("lr", lr, on_step=True, on_epoch=True, prog_bar=True)

        self.log("train/loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, x, y)
        self.log("val_loss", loss, prog_bar=True)
        preds = torch.sigmoid(logits)
        preds = (preds > 0.5).float().cpu().detach().numpy()
        y = (y > 0.5).float().cpu().detach().numpy()
        preds = preds.astype(np.int32)
        y = y.astype(np.int32)
        return loss

    def _compute_loss(self, logits, x, y):
        y = (y > 0).to(torch.float32).unsqueeze(1)  # Ensure y has shape (B, 1, H, W)
        return self.loss_fn(logits, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer=torch.optim.SGD(self.parameters(), lr=self.lr)

        # Warmup scheduler: linearly increase LR for the first 5 epochs
        warmup_epochs = int(self.max_epochs / 6)

        def warmup_lambda(epoch):
            return min(1.0, (epoch + 1) / warmup_epochs)

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_lambda
        )

        # feat scheduler: StepLR after warmup
        feat_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(self.max_epochs / 3), gamma=0.1
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, feat_scheduler],
            milestones=[warmup_epochs],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_after_backward(self):
        # Log the global norm of all gradients
        total_norm = 0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm**0.5
        self.log("train/grad_norm", total_norm, on_step=True, prog_bar=False)

    def predict_from_dataloader(
        self, dataloader, thresh=0.5, disable_tqdm=False, keep_pseudo_probs=False
    ):
        """
        Predicts segmentation masks from a given dataloader. Used mostly for evaluation, and in the final predict method.

        Args:
            dataloader (DataLoader): The dataloader containing the data to predict on.
            thresh (float): Threshold for binary segmentation.

        Returns:
            torch.Tensor or tuple: Predictions as a tensor
        """
        predictions = []
        self.eval()
        for batch in tqdm(dataloader, desc="Predicting", disable=disable_tqdm):

            x = batch[0]
            x = x.to(self.device)
            preds = torch.sigmoid(self(x))
            if not keep_pseudo_probs:

                # save pred in pkl
                # pickle.dump(preds.cpu().detach().numpy(), open("preds.pkl", "wb"))
                preds = (preds > thresh).float()
            else:
                # If keep_pseudo_probs is True, we return the raw model output
                preds = preds.cpu().detach()
            predictions.append(preds)
        predictions = torch.cat(predictions, dim=0)

        return predictions

    def predict(self, datamodule, dataloader="test", disable_tqdm=False, keep_pseudo_probs=False):
        """
        Predicts segmentation masks from a given datamodule. Used mostly for evaluation, and in the final predict method.

        Args:
            datamodule (LightningDataModule): The datamodule containing the data to predict on.
            thresh (float): Threshold for binary segmentation.

        Returns:
            torch.Tensor or tuple: Predictions as a tensor
        """
        self.eval()
        predictions = self.predict_from_dataloader(
            dataloader=getattr(datamodule, f"{dataloader}_dataloader")(),
            disable_tqdm=disable_tqdm,
            keep_pseudo_probs=keep_pseudo_probs,
        )
        return predictions
    

    def load_best_threshold(self, checkpoint_path):
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint["state_dict"])

        # find folder in which the checkpoint is located
        checkpoint_dir = os.path.dirname(checkpoint_path)
        print(f"Model weights loaded from {checkpoint_path}")
        if os.path.exists(os.path.join(checkpoint_dir, "unet_best_threshold.pkl")):
            self.best_threshold = pickle.load(
                open(os.path.join(checkpoint_dir, "unet_best_threshold.pkl"), "rb")
            )
            print(f"Loaded best threshold: {self.best_threshold}")
        else:
            self.best_threshold = None
            print("No best threshold file found")
        print(f"Model weights loaded from {checkpoint_path}")




class SiameseUNet(L.LightningModule):
    """
    It consists of an encoder, a bottleneck, and a decoder with skip connections, and a prediction head for FGTA.
    Args:
        n_frames (int): Number of input frames
        lr (float): Learning rate for the optimizer.

    """
  
    def __init__(
        self,
        in_channels: int = 1,
        seg_channels: int = 1,
        feat_channels: int = 32,
        features: List[int] = [64, 128, 256, 512],
        feat_lr: float = 1e-3,
        seg_lr: float = 1e-4,
        max_epochs: int = 30,
        contrastive_distance_threshold: float = 1.5,
        add_positional_encoding: bool = False,
        var_weight: float = 1.0,
        dist_weight: float = 1.0,
        reg_weight: float = 0.1,
        transformer: bool = True,

    ):
        super().__init__()
        self.feat_lr = feat_lr
        self.seg_lr = seg_lr
        self.in_channels = in_channels
        self.seg_channels = seg_channels
        self.feat_channels = feat_channels
        self.features = features
        self.max_epochs = max_epochs
        self.contrastive_distance_threshold = contrastive_distance_threshold
        self.add_positional_encoding = add_positional_encoding
        self.var_weight = var_weight
        self.dist_weight = dist_weight
        self.reg_weight = reg_weight
        self.loss_fn_seg = nn.BCEWithLogitsLoss()
        self.loss_fn_feat = None # contrastive loss to be implemented
        self.transformer = transformer
        self.model = SiameseUNetBase(in_channels, seg_channels, feat_channels, features, add_positional_encoding, transformer)

        self.save_hyperparameters()
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        opt_feat, opt_seg = self.optimizers()

        opt_feat.zero_grad()
        opt_seg.zero_grad()
        x, y = batch
        logits, feat_output = self(x)
        y_seg = (y > 0).to(torch.float32)
        loss_seg = self.loss_fn_seg(logits, y_seg)  

        total_loss, var_loss, dist_loss, reg_loss = self.compute_instance_contrastive_loss(
            dense_features=feat_output,
            instance_mask=y,
            margin=self.contrastive_distance_threshold,
            var_weight=self.var_weight,
            dist_weight=self.dist_weight,
            reg_weight=self.reg_weight,
        )

        self.log("train/var_loss", var_loss, on_step=True, prog_bar=True)
        self.log("train/dist_loss", dist_loss, on_step=True, prog_bar=True)
        self.log("train/reg_loss", reg_loss, on_step=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_step=True, prog_bar=True)
        self.log("train/loss_seg", loss_seg, on_step=True, prog_bar=True)

        lr_feat = opt_feat.param_groups[0]["lr"]
        lr_seg = opt_seg.param_groups[0]["lr"]
        self.log("train/lr_feat", lr_feat, on_step=True, prog_bar=True)
        self.log("train/lr_seg", lr_seg, on_step=True, prog_bar=True)

        self.manual_backward(loss_seg, retain_graph=True)
        self.manual_backward(total_loss)
        self.clip_gradients(
            opt_seg, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        self.clip_gradients(
            opt_feat, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        opt_feat.step()
        opt_seg.step()


    def compute_instance_contrastive_loss(
        self,
        dense_features,
        instance_mask,
        margin=1.5,
        var_weight=1.0,
        dist_weight=1.0,
        reg_weight=0.001,
    ):
        """
        Computes the intra-cell consistency loss, inter-cell contrastive loss,
        and origin regularization loss.
        
        Args:
            dense_features: Tensor of shape (B, T, C, H, W)
            instance_mask: Tensor of shape (B, T, H, W) with ground truth integer labels.
            margin: The minimum feature distance required between different cells.
            var_weight: Weight applied to the internal consistency (pull) loss.
            dist_weight: Weight applied to the contrastive (push) loss.
            reg_weight: Weight applied to the regularization (norm -> 0) loss.
            
        Returns:
            total_loss (scalar), var_loss (scalar), dist_loss (scalar), reg_loss (scalar)
        """
        B, T, C, H, W = dense_features.shape
        
        batch_total_loss = []
        batch_var_loss = []
        batch_dist_loss = []
        batch_reg_loss = []
        
        for b in range(B):
            feat_flat = dense_features[b].view(C, H * W * T)
            mask_flat = instance_mask[b].view(H * W * T)
            
            # 1. Identify unique cells (ignore background 0)
            unique_ids = torch.unique(mask_flat)
            cell_ids = unique_ids[unique_ids > 0]
            num_cells = len(cell_ids)
            
            # Handle edge case: No cells in THIS SPECIFIC image
            if num_cells == 0:
                zero_loss = torch.tensor(0.0, device=dense_features.device, requires_grad=True)
                batch_total_loss.append(zero_loss)
                batch_var_loss.append(zero_loss)
                batch_dist_loss.append(zero_loss)
                batch_reg_loss.append(zero_loss)
                continue
                
            # 2. Create boolean masks for each cell safely 
            cell_masks = (mask_flat.unsqueeze(1) == cell_ids.unsqueeze(0)).to(feat_flat.dtype)
            
            # 3. Calculate cell means
            cell_sums = feat_flat @ cell_masks               # (C, num_cells)
            cell_counts = cell_masks.sum(dim=0)              # (num_cells,)
            cell_means = cell_sums / (cell_counts + 1e-8)    # (C, num_cells)
            
            # -------------------------------------------------------------
            # LOSS 1: Variance / Consistency Loss (PULL)
            # -------------------------------------------------------------
            uniform_flat = cell_means @ cell_masks.T 
            
            fg_mask = (mask_flat > 0).to(feat_flat.dtype)
            num_fg_pixels = fg_mask.sum()
            
            sq_diff = ((feat_flat - uniform_flat) ** 2).sum(dim=0)
            var_loss = (sq_diff * fg_mask).sum() / (num_fg_pixels + 1e-8)
            
            # -------------------------------------------------------------
            # LOSS 2: Contrastive / Distance Loss (PUSH)
            # -------------------------------------------------------------
            if num_cells > 1:
                means_t = cell_means.T 
                dist_matrix = torch.cdist(means_t, means_t, p=2.0)
                push_penalty = torch.clamp(margin - dist_matrix, min=0.0)
                eye = torch.eye(num_cells, device=dense_features.device)
                mask_off_diag = 1.0 - eye
                push_penalty = push_penalty * mask_off_diag
                dist_loss = push_penalty.sum() / (num_cells * (num_cells - 1))
            else:
                dist_loss = torch.tensor(0.0, device=dense_features.device)

            # -------------------------------------------------------------
            # LOSS 3: Regularization Loss (Pull cluster means toward 0)
            # -------------------------------------------------------------
            reg_loss = torch.linalg.vector_norm(cell_means, ord=2, dim=0).mean()
                
            # -------------------------------------------------------------
            # Total Loss
            # -------------------------------------------------------------
            total_loss = (
                (var_weight * var_loss)
                + (dist_weight * dist_loss)
                + (reg_weight * reg_loss)
            )

            batch_total_loss.append(total_loss)
            batch_var_loss.append(var_loss)
            batch_dist_loss.append(dist_loss)
            batch_reg_loss.append(reg_loss)

        mean_total_loss = torch.stack(batch_total_loss).mean()
        mean_var_loss = torch.stack(batch_var_loss).mean()
        mean_dist_loss = torch.stack(batch_dist_loss).mean()
        mean_reg_loss = torch.stack(batch_reg_loss).mean()
        
        return mean_total_loss, mean_var_loss, mean_dist_loss, mean_reg_loss

    
    def configure_optimizers(self):
        feat_params = list(self.model.down.parameters()) + list(self.model.up_feat.parameters())
        seg_params = list(self.model.down.parameters()) + list(self.model.up_seg.parameters())

        feat_optimizer = torch.optim.Adam(params=feat_params, lr=self.feat_lr)
        seg_optimizer = torch.optim.Adam(params=seg_params, lr=self.seg_lr)
        # optimizer=torch.optim.SGD(self.parameters(), lr=self.lr)

        # Warmup scheduler: linearly increase LR for the first 5 epochs
        warmup_epochs = int(self.max_epochs / 6)

        def warmup_lambda(epoch):
            return min(1.0, (epoch + 1) / warmup_epochs)
    
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            feat_optimizer, lr_lambda=warmup_lambda
        )

        # feat scheduler: StepLR after warmup
        step_scheduler = torch.optim.lr_scheduler.StepLR(
            feat_optimizer, step_size=int(self.max_epochs / 3), gamma=0.1
        )

        feat_scheduler = torch.optim.lr_scheduler.SequentialLR(
            feat_optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=[warmup_epochs],
        )

        seg_scheduler = torch.optim.lr_scheduler.StepLR(
            seg_optimizer, step_size=int(self.max_epochs / 3), gamma=0.1
        )

        return [
            {
                "optimizer": feat_optimizer,
                "lr_scheduler": {
                    "scheduler": feat_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            },
            {
                "optimizer": seg_optimizer,
                "lr_scheduler": {
                    "scheduler": seg_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        ]


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, feat_output = self(x)
        y_seg = (y > 0).to(torch.float32)
        loss_seg = self.loss_fn_seg(logits, y_seg)  
        
        total_loss, var_loss, dist_loss, reg_loss = self.compute_instance_contrastive_loss(
            dense_features=feat_output,
            instance_mask=y,
            margin=self.contrastive_distance_threshold,
            var_weight=self.var_weight,
            dist_weight=self.dist_weight,
            reg_weight=self.reg_weight,
        )
    
        self.log("val/var_loss", var_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dist_loss", dist_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/reg_loss", reg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_seg", loss_seg, on_step=False, on_epoch=True, prog_bar=True)
        
    def on_train_epoch_end(self):
        feat_scheduler, seg_scheduler = self.lr_schedulers()
        feat_scheduler.step()
        seg_scheduler.step()

        if self.current_epoch % 20 == 0:
            dir_path = getattr(self.logger, "log_dir", None)
            if dir_path is not None:
                checkpoint_dir = os.path.join(dir_path, "checkpoints")
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    checkpoint_dir, f"unet_epoch_{self.current_epoch}.ckpt"
                )
                torch.save({"state_dict": self.state_dict()}, checkpoint_path)
        return super().on_train_epoch_end()

    
    def predict_from_dataloader(
        self, dataloader, thresh=0.5, disable_tqdm=False, keep_pseudo_probs=True
    ):
        """
        Predicts segmentation masks from a given dataloader. Used mostly for evaluation, and in the final predict method.

        Args:
            dataloader (DataLoader): The dataloader containing the data to predict on.
            thresh (float): Threshold for binary segmentation.

        Returns:
            torch.Tensor or tuple: Predictions as a tensor
        """
        predictions = []
        self.eval()
        for batch in tqdm(dataloader, desc="Predicting", disable=disable_tqdm):

            x = batch[0]
            x = x.to(self.device)
            seg_output, feat_output = self(x)
            seg_probs = torch.sigmoid(seg_output)
            if not keep_pseudo_probs:

                # save pred in pkl
                # pickle.dump(preds.cpu().detach().numpy(), open("preds.pkl", "wb"))
                seg_output = (seg_probs > thresh).float()
                feat_output = feat_output.cpu().detach()
            else:
                # If keep_pseudo_probs is True, we return the raw model output
                seg_output = seg_probs.cpu().detach()
                feat_output = feat_output.cpu().detach()
            predictions.append((seg_output, feat_output))
        return predictions
    

    def predict(self, datamodule, dataloader="test", threshold=0.5, cluster_method='dbscan',min_cell_size=100, disable_tqdm=False, connected_components=False, num_workers=1 ):
        """
        Predicts segmentation masks from a given datamodule. Used mostly for evaluation, and in the final predict method.

        Args:
            datamodule (LightningDataModule): The datamodule containing the data to predict on.
            thresh (float): Threshold for binary segmentation.

        Returns:
            torch.Tensor or tuple: Predictions as a tensor
        """
        from models.utils import cluster_features_to_instances, metrics_per_label, cluster_features_to_instances_connected_components
        self.eval()
        out=[]
        per_instance_features=[]
        per_instance_variance=[]
        for batch in tqdm(getattr(datamodule, f"{dataloader}_dataloader")(), desc="Predicting", disable=disable_tqdm):
            x = batch[0]
            x = x.to(self.device)
            seg_output, feat_output = self(x)
            bin_mask = (torch.sigmoid(seg_output) > threshold).float().detach()
            feat_output = feat_output.cpu().detach()
            if connected_components:
                instance_mask=cluster_features_to_instances_connected_components(feat_output, bin_mask, min_cell_size=min_cell_size, margin=self.contrastive_distance_threshold,method=cluster_method,num_workers=num_workers)
            else:
                instance_mask = cluster_features_to_instances(feat_output, bin_mask, method=cluster_method, margin=self.contrastive_distance_threshold, min_cell_size=min_cell_size)
            avg_features,var_features = metrics_per_label(instance_mask, feat_output, include_background=False)
            per_instance_features.append(avg_features)
            per_instance_variance.append(var_features)
            out.append(instance_mask)

        return out, per_instance_features, per_instance_variance
    