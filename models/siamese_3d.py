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



class AddCoordinates(nn.Module):
    """
    Appends normalized coordinate channels to a spatial tensor.
    For 4D inputs this adds Y/X channels. For 5D inputs this adds T/Y/X channels.
    This explicitly breaks translation invariance, helping the network
    distinguish between identical objects at different locations.
    """
    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, C, H, W) or (B, C, T, H, W)
        Returns:
            Tensor with coordinate channels appended.
        """
        batch_size = x.size(0)

        if x.dim() == 4:
            _, _, height, width = x.size()
            y_coords = torch.linspace(-1, 1, steps=height, device=x.device)
            x_coords = torch.linspace(-1, 1, steps=width, device=x.device)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing="ij")
            y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
            x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
            coords = [x, y_grid, x_grid]
            if self.with_r:
                r_grid = torch.sqrt(x_grid.pow(2) + y_grid.pow(2))
                coords.append(r_grid)
            return torch.cat(coords, dim=1)

        if x.dim() != 5:
            raise ValueError(f"Expected a 4D or 5D tensor, got shape {tuple(x.shape)}")

        _, _, depth, height, width = x.size()
        t_coords = torch.linspace(-1, 1, steps=depth, device=x.device)
        y_coords = torch.linspace(-1, 1, steps=height, device=x.device)
        x_coords = torch.linspace(-1, 1, steps=width, device=x.device)
        t_grid, y_grid, x_grid = torch.meshgrid(t_coords, y_coords, x_coords, indexing="ij")
        t_grid = t_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        y_grid = y_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        x_grid = x_grid.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        coords = [x, t_grid, y_grid, x_grid]
        if self.with_r:
            r_grid = torch.sqrt(t_grid.pow(2) + x_grid.pow(2) + y_grid.pow(2))
            coords.append(r_grid)
        return torch.cat(coords, dim=1)
    

class UNetBlock(nn.Module):
    """
    A single block of the U-Net architecture, consisting of two convolutional layers
    followed by ReLU activations. This block is typically used in the encoder and decoder parts of the U-Net.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), padding=(0, 1, 1)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UnetDown(nn.Module):
    def __init__(self, in_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.encoder = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.keep_embeddings = False
        self.embeddings = []
        prev_channels = in_channels
        for i,feat in enumerate(features):
            if i==0:
                self.encoder.append(UNetBlock(prev_channels, feat, kernel_size=(3, 3, 3), padding=(0, 1, 1)))
            else:
                self.encoder.append(UNetBlock(prev_channels, feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            prev_channels = feat

    def forward(self, x):

        skip_connections = []
        self.embeddings = []
        for i, down in enumerate(self.encoder):

            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        return skip_connections, x


class UnetUp(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        self.up_transpose = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.bottleneck = UNetBlock(features[-1], features[-1]*2, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        prev_channels = features[-1]*2
        for i, feat in enumerate(reversed(features)):
            self.up_transpose.append(
                nn.ConvTranspose3d(prev_channels, feat, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )
            self.decoder.append(UNetBlock(prev_channels, feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            prev_channels = feat

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)
    def forward(self, x, skip_connections):
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.up_transpose)):
            x = self.up_transpose[idx](x)
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            x = self.decoder[idx](x)
        return self.final_conv(x)
    
class UnetUpWithPositionalEncoding(nn.Module):
    def __init__(self, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        self.up_transpose = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.bottleneck = UNetBlock(features[-1], features[-1]*2, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        prev_channels = features[-1]*2
        for i, feat in enumerate(reversed(features)):
            self.up_transpose.append(
                nn.ConvTranspose3d(prev_channels, feat, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )
            if feat==features[0]: #add positional encoding only at the last layer
                self.decoder.append(UNetBlock(prev_channels+3, feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            else:
                self.decoder.append(UNetBlock(prev_channels, feat, kernel_size=(1, 3, 3), padding=(0, 1, 1)))
            prev_channels = feat
        self.add_coordinates = AddCoordinates(with_r=False)
        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x, skip_connections):
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(len(self.up_transpose)):
            x = self.up_transpose[idx](x)
            skip = skip_connections[idx]
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
            x = torch.cat((skip, x), dim=1)
            if idx ==len(self.up_transpose)-1: #add positional encoding only at the last layer
                x = self.add_coordinates(x)
            x = self.decoder[idx](x)

        return self.final_conv(x)

class SiameseUNetBase(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feat_channels=32, features=[64, 128, 256, 512], add_positional_encoding=False):
        super().__init__()
        self.in_channels = in_channels
        
        self.feat_channels = feat_channels
        self.out_channels = out_channels

        self.down = UnetDown(in_channels, features)
        if add_positional_encoding:
            self.up_feat = UnetUpWithPositionalEncoding(feat_channels, features)
        else:
            self.up_feat = UnetUp(feat_channels, features)
        self.up_spots = UnetUp(out_channels, features)
        self.add_positional_encoding = add_positional_encoding

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)
        skip_connections, bottleneck_input = self.down(x)
        spots_output = self.up_spots(bottleneck_input, skip_connections)
        feat_output = self.up_feat(bottleneck_input, skip_connections)
        spots_output = spots_output.squeeze(1)
        feat_output = feat_output.permute(0, 2, 1, 3, 4).contiguous()
        return spots_output, feat_output



    
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
        spots_channels: int = 1,
        feat_channels: int = 32,
        features: List[int] = [64, 128, 256, 512],
        main_lr: float = 1e-3,
        spots_lr: float = 1e-4,
        max_epochs: int = 30,
        contrastive_distance_threshold: float = 1.5,
        add_positional_encoding: bool = False,
        
    ):


        super().__init__()
        self.main_lr = main_lr
        self.spots_lr = spots_lr
        self.in_channels = in_channels
        self.spots_channels = spots_channels
        self.feat_channels = feat_channels
        self.features = features
        self.max_epochs = max_epochs
        self.contrastive_distance_threshold = contrastive_distance_threshold
        self.add_positional_encoding = add_positional_encoding
        self.loss_fn_seg = nn.BCEWithLogitsLoss()
        self.loss_fn_feat = None # contrastive loss to be implemented
        self.model = SiameseUNetBase(in_channels, spots_channels, feat_channels, features, add_positional_encoding)

        self.save_hyperparameters()

    def forward(self, x):
        x= self.model(x)
        return x
    
    def training_step(self, batch, batch_idx):
        opt_main, opt_spots = self.optimizers()

        opt_main.zero_grad()
        opt_spots.zero_grad()
        x,y= batch
        logits, feat_output = self(x)
        y_seg = (y > 0).to(torch.float32)
        loss_seg = self.loss_fn_seg(logits, y_seg)  
        

        total_loss, var_loss, dist_loss = self.compute_instance_contrastive_loss(dense_features=feat_output, instance_mask=y, margin=self.contrastive_distance_threshold)
   

        self.log("train/var_loss", var_loss, on_step=True, prog_bar=True)
        self.log("train/dist_loss", dist_loss, on_step=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_step=True, prog_bar=True)
        self.log("train/loss_seg", loss_seg, on_step=True, prog_bar=True)
        #log lr
        lr_main = opt_main.param_groups[0]["lr"]
        lr_spots = opt_spots.param_groups[0]["lr"]
        self.log("train/lr_main", lr_main, on_step=True, prog_bar=True)
        self.log("train/lr_spots", lr_spots, on_step=True, prog_bar=True)
        # Update both optimizers wrt their respective losses
        self.manual_backward(loss_seg, retain_graph=True)
        self.manual_backward(total_loss)
        self.clip_gradients(
            opt_spots, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        self.clip_gradients(
            opt_main, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        opt_main.step()
        opt_spots.step()

    def on_train_start(self):
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)
        return super().on_train_start()


    def compute_instance_contrastive_loss(self, dense_features, instance_mask, margin=1.5, var_weight=1.0, dist_weight=1.0):
        """
        Computes the intra-cell consistency loss and inter-cell contrastive loss.
        
        Args:
            dense_features: Tensor of shape (B, T, C, H, W)
            instance_mask: Tensor of shape (B, T, H, W) with ground truth integer labels.
            margin: The minimum feature distance required between different cells.
            var_weight: Weight applied to the internal consistency (pull) loss.
            dist_weight: Weight applied to the contrastive (push) loss.
            
        Returns:
            total_loss (scalar), var_loss (scalar), dist_loss (scalar)
        """
        B, T, C, H, W = dense_features.shape
        # Use standard lists to collect losses for each item in the batch
        batch_total_loss = []
        batch_var_loss = []
        batch_dist_loss = []
        
        for b in range(B):
            feat_flat = dense_features[b].reshape(C, H * W * T)
            mask_flat = instance_mask[b].reshape(H * W * T)
            
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
                continue # Move to the next image in the batch, DO NOT return
                
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
                # Requires grad isn't strictly necessary here if var_loss has it, 
                # but it is safer to include it.
                dist_loss = torch.tensor(0.0, device=dense_features.device)
                
            # -------------------------------------------------------------
            # Total Loss
            # -------------------------------------------------------------
            total_loss = (var_weight * var_loss) + (dist_weight * dist_loss)

            # Append to our lists
            batch_total_loss.append(total_loss)
            batch_var_loss.append(var_loss)
            batch_dist_loss.append(dist_loss)

        # Stack the lists into tensors and calculate the mean across the batch dimension
        # torch.stack is fully differentiable and avoids in-place operations
        mean_total_loss = torch.stack(batch_total_loss).mean()
        mean_var_loss = torch.stack(batch_var_loss).mean()
        mean_dist_loss = torch.stack(batch_dist_loss).mean()
        
        return mean_total_loss, mean_var_loss, mean_dist_loss
    def configure_optimizers(self):
        main_params = list(self.model.down.parameters()) + list(self.model.up_feat.parameters())
        spots_params = list(self.model.down.parameters()) + list(self.model.up_spots.parameters())

        main_optimizer = torch.optim.Adam(params=main_params, lr=self.main_lr)
        spots_optimizer = torch.optim.Adam(params=spots_params, lr=self.spots_lr)
        # optimizer=torch.optim.SGD(self.parameters(), lr=self.lr)

        # Warmup scheduler: linearly increase LR for the first 5 epochs
        warmup_epochs = int(self.max_epochs / 6)

        def warmup_lambda(epoch):
            return min(1.0, (epoch + 1) / warmup_epochs)
    
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            main_optimizer, lr_lambda=warmup_lambda
        )

        # Main scheduler: StepLR after warmup
        step_scheduler = torch.optim.lr_scheduler.StepLR(
            main_optimizer, step_size=int(self.max_epochs / 3), gamma=0.1
        )

        main_scheduler = torch.optim.lr_scheduler.SequentialLR(
            main_optimizer,
            schedulers=[warmup_scheduler, step_scheduler],
            milestones=[warmup_epochs],
        )

        spots_scheduler = torch.optim.lr_scheduler.StepLR(
            spots_optimizer, step_size=int(self.max_epochs / 3), gamma=0.1
        )

        return [
            {
                "optimizer": main_optimizer,
                "lr_scheduler": {
                    "scheduler": main_scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                }
            },
            {
                "optimizer": spots_optimizer,
                "lr_scheduler": {
                    "scheduler": spots_scheduler,
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
        
        total_loss, var_loss, dist_loss = self.compute_instance_contrastive_loss(dense_features=feat_output, instance_mask=y, margin=self.contrastive_distance_threshold)
    
        self.log("val/var_loss", var_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/dist_loss", dist_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_seg", loss_seg, on_step=False, on_epoch=True, prog_bar=True)

        
    def on_train_epoch_end(self):
        main_scheduler, spots_scheduler = self.lr_schedulers()
        main_scheduler.step()
        spots_scheduler.step()

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
            spots_output, feat_output = self(x)
            spots_probs = torch.sigmoid(spots_output)
            if not keep_pseudo_probs:

                # save pred in pkl
                # pickle.dump(preds.cpu().detach().numpy(), open("preds.pkl", "wb"))
                spots_output = (spots_probs > thresh).float()
                feat_output = feat_output.cpu().detach()
            else:
                # If keep_pseudo_probs is True, we return the raw model output
                spots_output = spots_probs.cpu().detach()
                feat_output = feat_output.cpu().detach()
            predictions.append((spots_output, feat_output))
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
            spots_output, feat_output = self(x)
            bin_mask = (torch.sigmoid(spots_output) > threshold).float().detach()
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
    