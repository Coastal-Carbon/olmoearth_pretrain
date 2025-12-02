"""
Train OlmoEarth Reconstructor on OlmoEarth training data.
"""

import sys
import logging
import time
import gc
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
import torch
from upath import UPath
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import wandb
import boto3

from olmoearth_pretrain.data.dataset import OlmoEarthDataset, GetItemArgs
from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.train.masking import MaskValue, MaskedOlmoEarthSample
from olmoearth_pretrain.model_loader import load_model_from_id, ModelID
from olmoearth_pretrain.nn.flexi_vit import Reconstructor


@dataclass
class ReconstructorTrainingConfig:
    """Configuration for training the Reconstructor."""
    
    # Paths
    dataset_path: str = (
        "s3://cc-dataocean/scratch/20251114_olmo_example/"
        "h5py_data_w_missing_timesteps_zstd_3_128_x_4/"
        "cdl_gse_landsat_openstreetmap_raster_"
        "sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/"
        "1138828"
    )
    checkpoint_dir: str = "./checkpoints_reconstructor"
    
    # Training
    num_epochs: int = 40
    learning_rate: float = 5e-4
    max_grad_norm: float = 100.0

    # Data
    patch_size: int = 1  # Pixel-level output
    encoder_patch_size: int = 4  # Encoder patch size - keep at 4 to avoid OOM (32×32 latent tokens)
    max_patch_size: int = 1  # ConvTranspose2d kernel size (no upsampling needed at patch_size=1)
    num_samples_per_epoch: int = 2  # Load 2 samples at a time
    max_total_samples: int = 1000  # Total unique samples to train on
    val_fraction: float = 0.2  # Use 20% for validation (800 train, 200 val)

    # Modalities - specify in one place
    supported_modalities: list = field(
        default_factory=lambda: [Modality.SENTINEL1, Modality.SENTINEL2_L2A]
    )

    # Device
    device: str = "cuda:1"

    # Modalities to reconstruct
    supported_modality_names: list = field(default_factory=lambda: ["sentinel1", "sentinel2_l2a", "landsat", "cdl", "latlon"])


def get_api_key_from_parameter_store(parameter_name: str, region: str = 'us-east-1') -> str:
    """Retrieve API key from AWS Parameter Store."""
    ssm = boto3.client('ssm', region_name=region)
    response = ssm.get_parameter(
        Name=parameter_name,
        WithDecryption=True
    )
    return response['Parameter']['Value']


def compute_pairwise_correlation_loss(reconstructed, ground_truth, return_channel_correlations=False):
    """Compute loss by maximizing correlation between reconstructed and ground truth bands.
    
    For each band (channel), compute the correlation coefficient R between the 
    reconstructed band and ground truth band across all spatial pixels.
    Then maximize the mean correlation across all bands.
    
    Args:
        reconstructed: Tensor of shape [B, H, W, C] (reconstructed bands)
        ground_truth: Tensor of shape [B, H, W, C] (original bands)
        return_channel_correlations: If True, return per-channel correlations for analysis
        
    Returns:
        loss: Scalar loss value (1 - mean_correlation, so minimizing loss maximizes correlation)
        channel_correlations: (Optional) List of per-channel correlation values for logging
    """
    B, H, W, C = reconstructed.shape
    
    # Flatten spatial dimensions: [B, H*W, C]
    recon_flat = reconstructed.reshape(B, H*W, C)
    gt_flat = ground_truth.reshape(B, H*W, C)
    
    # Compute correlation for each band using vectorized operations (maintains gradients)
    correlations = []
    
    # For each sample in batch
    for b in range(B):
        recon_b = recon_flat[b]  # [H*W, C]
        gt_b = gt_flat[b]  # [H*W, C]

        # For each channel, compute correlation across spatial pixels
        for c in range(C):
            recon_band = recon_b[:, c]  # [H*W]
            gt_band = gt_b[:, c]  # [H*W]
            
            # Compute Pearson correlation coefficient (all tensor ops to preserve gradients)
            recon_mean = recon_band.mean()
            recon_std = recon_band.std() + 1e-8
            recon_normalized = (recon_band - recon_mean) / recon_std
            
            gt_mean = gt_band.mean()
            gt_std = gt_band.std() + 1e-8
            gt_normalized = (gt_band - gt_mean) / gt_std
            
            # Correlation = mean of element-wise product of normalized values
            correlation = torch.mean(recon_normalized * gt_normalized)
            correlations.append(correlation)

    # Stack correlations and compute mean (all tensor ops for backprop)
    mean_correlation = torch.stack(correlations).mean()

    # Loss = 1 - correlation (so maximizing correlation minimizes loss)
    loss = 1.0 - mean_correlation
    
    if return_channel_correlations:
        # Extract values for logging AFTER computing loss (doesn't affect gradients)
        channel_correlations = [c.item() for c in correlations]
        return loss, channel_correlations
    return loss


def visualize_reconstruction_pairs(target_images, reconstructed_images, epoch, output_dir):
    """Visualize pairs of target and reconstructed Sentinel-2 images.
    
    Args:
        target_images: List of numpy arrays [H, W, C] with target images
        reconstructed_images: List of numpy arrays [H, W, C] with reconstructed images
        epoch: Epoch number for filename
        output_dir: Directory to save visualization
    """
    num_samples = len(target_images)
    if num_samples == 0:
        return
    
    # Limit to 4 samples for visualization
    num_samples = min(num_samples, 4)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 4 * num_samples))
    gs = gridspec.GridSpec(num_samples, 2, figure=fig, hspace=0.3, wspace=0.2)
    
    for idx in range(num_samples):
        target = target_images[idx]
        recon = reconstructed_images[idx]
        
        # Use RGB bands (if available) - typically indices 2, 1, 0 for Sentinel-2 L2A
        # Fallback to first 3 channels if not enough bands
        if target.shape[2] >= 3:
            rgb_indices = [2, 1, 0]
        else:
            rgb_indices = list(range(min(3, target.shape[2])))
        
        target_rgb = target[:, :, rgb_indices]
        recon_rgb = recon[:, :, rgb_indices]
        
        # Normalize to 0-1 for visualization
        target_min, target_max = target_rgb.min(), target_rgb.max()
        recon_min, recon_max = recon_rgb.min(), recon_rgb.max()
        
        if target_max > target_min:
            target_rgb = (target_rgb - target_min) / (target_max - target_min)
        if recon_max > recon_min:
            recon_rgb = (recon_rgb - recon_min) / (recon_max - recon_min)
        
        target_rgb = np.clip(target_rgb, 0, 1)
        recon_rgb = np.clip(recon_rgb, 0, 1)
        
        # Plot target
        ax_target = fig.add_subplot(gs[idx, 0])
        im_target = ax_target.imshow(target_rgb, interpolation='nearest')
        ax_target.set_title(f"Sample {idx+1} - Target", fontsize=12, weight='bold')
        ax_target.axis('off')
        
        # Plot reconstructed
        ax_recon = fig.add_subplot(gs[idx, 1])
        im_recon = ax_recon.imshow(recon_rgb, interpolation='nearest')
        ax_recon.set_title(f"Sample {idx+1} - Reconstructed", fontsize=12, weight='bold')
        ax_recon.axis('off')
    
    # Save figure
    output_path = Path(output_dir) / f"reconstruction_epoch_{epoch:03d}.png"
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    return output_path


class ReconstructorTrainer:
    """Trainer for OlmoEarth Reconstructor."""
    
    def __init__(self, config: ReconstructorTrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Setup logging
        self.logger = self._setup_logging()
        self.logger.info("="*80)
        self.logger.info("OlmoEarth Reconstructor Training: Data Loading")
        self.logger.info("="*80)
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self._load_dataset()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers = []
        logger.addHandler(ch)
        
        return logger
    
    def _load_dataset(self):
        """Load OlmoEarth training dataset."""
        self.logger.info(f"Loading dataset from {self.config.dataset_path}")
        
        # Load all modalities - model handles None values for missing modalities
        training_modalities = [
            "cdl", "gse", "landsat", "latlon", "openstreetmap_raster",
            "sentinel1", "sentinel2_l2a", "srtm", "worldcereal", "worldcover",
            "era5_10", "worldpop", "wri_canopy_height_map"
        ]
        
        self.dataset = OlmoEarthDataset(
            h5py_dir=UPath(self.config.dataset_path),
            training_modalities=training_modalities,
            dtype=np.float32,
            normalize=False,
        )
        
        # Prepare dataset - scans S3 once to discover samples and cache metadata
        self.logger.info("Preparing dataset (discovering samples from S3, this may take a few minutes)...")
        start_time = time.time()
        self.dataset.prepare()
        elapsed = time.time() - start_time
        self.logger.info(f"✓ Dataset prepared with {len(self.dataset)} samples in {elapsed:.1f}s")
    
    def _get_sample(self, sample_idx: int):
        """Get a sample from the dataset."""
        try:
            _, sample = self.dataset[GetItemArgs(
                idx=sample_idx,
                patch_size=self.config.patch_size,  # Dataset-level patch processing
                sampled_hw_p=32  # Load 32x32 spatial region to reduce memory
            )]
            return sample_idx, sample  # Return both idx and sample
        except Exception as e:
            self.logger.warning(f"Failed to load sample {sample_idx}: {e}")
            return None, None
    
    def _create_masked_sample(self, sample):
        """Create a masked OlmoEarth sample for training.
        
        Masks one month's worth of Sentinel-2 data for reconstruction.
        Only includes sentinel1, sentinel2_l2a, timestamps, and latlon.
        Converts to tensors and moves to device immediately to minimize CPU memory.
        
        Args:
            sample: OlmoEarthSample from dataset
            
        Returns:
            MaskedOlmoEarthSample with Sentinel-2 masked for one month (reconstruction target)
        """
        # Get shapes
        sentinel1 = sample.sentinel1  # [H, W, T, C_s1]
        sentinel2_l2a = sample.sentinel2_l2a  # [H, W, T, C_s2]
        
        if sentinel2_l2a is None:
            return None, None

        H, W, T, C_s2 = sentinel2_l2a.shape
        
        # Convert to tensors IMMEDIATELY (don't keep as numpy arrays)
        sentinel1_batch = torch.from_numpy(sentinel1[None, ...]).float() if sentinel1 is not None else None  # [1, H, W, T, C_s1]
        sentinel2_batch = torch.from_numpy(sentinel2_l2a[None, ...]).float()  # [1, H, W, T, C_s2]
        
        # Delete source numpy arrays to free memory
        del sentinel1, sentinel2_l2a
        
        # Create timestamps tensor [1, T, 3] for [day, month, year]
        if hasattr(sample, 'timestamps') and sample.timestamps is not None:
            timestamps = torch.from_numpy(sample.timestamps[None, ...]).long()  # [1, T, 3]
        else:
            # Create default timestamps (day, month, year)
            timestamps = torch.zeros((1, T, 3), dtype=torch.long)
            for t in range(T):
                timestamps[0, t, 0] = 1 + (t % 28)  # Day 1-28
                timestamps[0, t, 1] = t % 12  # Month 0-11
                timestamps[0, t, 2] = 2024  # Year
        
        # Determine mask month from timestamps
        mask_month = int(torch.randint(0, 12, (1,)).item())
        
        # Create masks on device directly
        sentinel1_num_band_sets = Modality.SENTINEL1.num_band_sets if sentinel1_batch is not None else 0
        if sentinel1_batch is not None:
            sentinel1_mask = torch.full(
                (1, H, W, T, sentinel1_num_band_sets),
                MaskValue.ONLINE_ENCODER.value,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            sentinel1_mask = None
        
        sentinel2_num_band_sets = Modality.SENTINEL2_L2A.num_band_sets
        sentinel2_mask = torch.full(
            (1, H, W, T, sentinel2_num_band_sets),
            MaskValue.ONLINE_ENCODER.value,
            dtype=torch.float32,
            device=self.device,
        )

        # Mask one random month for Sentinel-2 reconstruction
        # Find all timesteps with the target month
        for t in range(T):
            if int(timestamps[0, t, 1]) == mask_month:
                sentinel2_mask[:, :, :, t, :] = MaskValue.DECODER.value
        
        # Move tensors to device
        timestamps_tensor = timestamps.to(self.device)
        sentinel1_batch = sentinel1_batch.to(self.device) if sentinel1_batch is not None else None
        sentinel2_batch = sentinel2_batch.to(self.device)
        
        # Create latlon mask if available
        if hasattr(sample, 'latlon') and sample.latlon is not None:
            latlon = sample.latlon[None, ...]  # [1, 2]
            latlon_tensor = torch.from_numpy(latlon).float().to(self.device)
            latlon_mask = torch.full((1, 1), MaskValue.ONLINE_ENCODER.value, dtype=torch.float32, device=self.device)
        else:
            latlon_tensor = None
            latlon_mask = None
        
        # Create masked sample - only sentinel1, sentinel2_l2a, timestamps, and latlon
        masked_sample = MaskedOlmoEarthSample(
            timestamps=timestamps_tensor,
            sentinel1=sentinel1_batch,
            sentinel1_mask=sentinel1_mask,
            sentinel2_l2a=sentinel2_batch,
            sentinel2_l2a_mask=sentinel2_mask,
            latlon=latlon_tensor,
            latlon_mask=latlon_mask,
        )
        
        return masked_sample, mask_month
    
    def train(self):
        """Training loop: Add Reconstructor and train on data.
        
        This method loads the model, creates the Reconstructor, and trains it
        to reconstruct masked Sentinel-2 images from other modalities.
        """
        self.logger.info("="*80)
        self.logger.info("Training Reconstructor")
        self.logger.info("="*80)
        
        # Initialize WandB

        wandb_api_key = get_api_key_from_parameter_store('/development/hum-ai-model-factory/weights_and_biases_api_key')
        wandb.login(key=wandb_api_key)
        wandb.init(
            project="olmoearth-reconstructor",
            name=f"reconstructor-training",
            config={
                "num_epochs": self.config.num_epochs,
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.num_samples_per_epoch,
                "num_total_samples": self.config.max_total_samples,
                "encoder_patch_size": self.config.encoder_patch_size,
                "max_patch_size": self.config.max_patch_size,
            }
        )
        self.logger.info("✓ WandB initialized successfully")

        wandb_enabled = True

        # Load base model
        self.logger.info("Loading OlmoEarth base model...")
        model = load_model_from_id(ModelID.OLMOEARTH_V1_BASE)
        model.to(self.device)
        
        # Freeze encoder to speed up training
        for param in model.encoder.parameters():
            param.requires_grad = False
        
        # Add Reconstructor
        self.logger.info("Creating Reconstructor...")
        
        # Set base model to eval mode (encoder/decoder frozen)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        
        model.reconstructor = Reconstructor(
            decoder=model.decoder,
            supported_modalities=self.config.supported_modalities,
            max_patch_size=self.config.max_patch_size,  # Use larger kernel for smooth transitions
        )
        
        # Only Reconstructor is trainable
        model.reconstructor.to(self.device)
        model.reconstructor.train()
        
        # Count parameters
        reconstructor_params = sum(p.numel() for p in model.reconstructor.parameters())
        reconstructor_trainable = sum(p.numel() for p in model.reconstructor.parameters() if p.requires_grad)
        self.logger.info(f"Reconstructor: {reconstructor_trainable:,} trainable / {reconstructor_params:,} total parameters")
        
        # Setup optimizer and loss
        optimizer = torch.optim.Adam(
            model.reconstructor.parameters(),
            lr=self.config.learning_rate
        )
        
        # Each epoch will train on 80% and validate on 20% of samples
        samples_per_training = self.config.max_total_samples
        num_train_samples = int(samples_per_training * (1.0 - self.config.val_fraction))
        num_val_samples = samples_per_training - num_train_samples
        
        # Training loop
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            self.logger.info("Training phase...")
            train_loss = 0.0
            train_samples = 0
            target_images = []
            reconstructed_images = []
            
            sample_index = 0  # Start from sample 0
            
            while train_samples < num_train_samples:
                batch_size = min(self.config.num_samples_per_epoch, num_train_samples - train_samples)
                samples_to_process = []
                
                for _ in range(batch_size):
                    loaded_idx, sample = self._get_sample(sample_index)
                    if sample is not None:
                        samples_to_process.append(sample)
                    sample_index += 1  # Sequential, don't cycle
                
                if not samples_to_process:
                    break

                # Process the batch
                for sample_idx, sample in enumerate(samples_to_process):
                    # Create masked sample
                    masked_sample, mask_month = self._create_masked_sample(sample)
                    if masked_sample is None:
                        continue

                    # Forward pass through encoder (frozen) and decoder (frozen)
                    with torch.no_grad():
                        encoder_output, decoder_output, pooled_output, _, _ = model(
                            masked_sample,
                            patch_size=self.config.encoder_patch_size
                        )
                    
                    # Get reconstructor output
                    reconstructed = model.reconstructor(
                        encoder_output,
                        timestamps=masked_sample.timestamps,
                        patch_size=self.config.patch_size
                    )
                    
                    # Compute loss on masked month using pairwise correlation
                    if hasattr(reconstructed, 'sentinel2_l2a') and reconstructed.sentinel2_l2a is not None:
                        # Get all timesteps for the masked month
                        masked_month_indices = []
                        for t in range(masked_sample.sentinel2_l2a.shape[3]):
                            if int(masked_sample.timestamps[0, t, 1]) == mask_month:
                                masked_month_indices.append(t)
                        
                        if len(masked_month_indices) > 0:
                            recon_s2 = reconstructed.sentinel2_l2a
                            gt_s2 = masked_sample.sentinel2_l2a
                            
                            if recon_s2.ndim == 7:
                                continue
                            
                            recon_month = recon_s2[:, :, :, masked_month_indices, :]
                            gt_month = gt_s2[:, :, :, masked_month_indices, :]
                            
                            # Ensure same spatial size
                            if recon_month.shape[1] != gt_month.shape[1] or recon_month.shape[2] != gt_month.shape[2]:
                                min_h = min(recon_month.shape[1], gt_month.shape[1])
                                min_w = min(recon_month.shape[2], gt_month.shape[2])
                                recon_month = recon_month[:, :min_h, :min_w, :, :]
                                gt_month = gt_month[:, :min_h, :min_w, :, :]
                            
                            # Flatten spatial+temporal dimensions, keep channels separate
                            recon_month_flat = recon_month.reshape(recon_month.shape[0], -1, recon_month.shape[-1])
                            gt_month_flat = gt_month.reshape(gt_month.shape[0], -1, gt_month.shape[-1])
                            
                            if recon_month_flat.numel() > 1e8:
                                continue
                            
                            # Reshape to [B, H, W, C] format for loss function
                            recon_for_loss = recon_month_flat.unsqueeze(2)
                            gt_for_loss = gt_month_flat.unsqueeze(2)
                            
                            # Compute loss
                            correlation_loss, channel_correlations = compute_pairwise_correlation_loss(
                                recon_for_loss, gt_for_loss, return_channel_correlations=True
                            )
                            
                            loss = correlation_loss
                            
                            # Backward pass
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.reconstructor.parameters(), self.config.max_grad_norm)
                            optimizer.step()

                            train_loss += loss.item()
                            train_samples += 1
                            
                            # Log per-channel correlations
                            s2_band_names = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09']
                            corr_str = " | ".join([f"{s2_band_names[i]}: {c:+.3f}" for i, c in enumerate(channel_correlations[:12])])
                            self.logger.info(f"    Train - Loss={loss.item():.6f} | {corr_str}")
                            
                            # Store images for visualization
                            if len(target_images) < 4:
                                target_avg = gt_month[0].mean(dim=2).detach().cpu().numpy()
                                recon_avg = recon_month[0].mean(dim=2).detach().cpu().numpy()
                                target_images.append(target_avg)
                                reconstructed_images.append(recon_avg)
                            
                            # Clean up GPU memory
                            del masked_sample, encoder_output, decoder_output, pooled_output, reconstructed
                            del recon_month, gt_month, recon_month_flat, gt_month_flat, loss
                            torch.cuda.empty_cache()
                            gc.collect()

            avg_train_loss = train_loss / max(1, train_samples)
            self.logger.info(f"Epoch {epoch+1} TRAIN: avg_loss={avg_train_loss:.6f} ({train_samples} samples)")

            self.logger.info("Validation phase...")
            model.reconstructor.eval()
            val_loss = 0.0
            val_samples = 0
            
            sample_index = num_train_samples  # Start from sample 800
            samples_loaded = 0
            
            with torch.no_grad():
                while val_samples < num_val_samples:
                    batch_size = min(self.config.num_samples_per_epoch, num_val_samples - val_samples)
                    samples_to_process = []
                    
                    for _ in range(batch_size):
                        loaded_idx, sample = self._get_sample(sample_index)
                        if sample is not None:
                            samples_to_process.append(sample)
                        sample_index += 1  # Sequential
                    
                    if not samples_to_process:
                        break

                    # Process validation batch
                    for sample_idx, sample in enumerate(samples_to_process):
                        masked_sample, mask_month = self._create_masked_sample(sample)
                        if masked_sample is None:
                            continue

                        encoder_output, decoder_output, pooled_output, _, _ = model(
                            masked_sample,
                            patch_size=self.config.encoder_patch_size
                        )
                        
                        reconstructed = model.reconstructor(
                            encoder_output,
                            timestamps=masked_sample.timestamps,
                            patch_size=self.config.patch_size
                        )
                        
                        if hasattr(reconstructed, 'sentinel2_l2a') and reconstructed.sentinel2_l2a is not None:
                            masked_month_indices = []
                            for t in range(masked_sample.sentinel2_l2a.shape[3]):
                                if int(masked_sample.timestamps[0, t, 1]) == mask_month:
                                    masked_month_indices.append(t)
                            
                            if len(masked_month_indices) > 0:
                                recon_s2 = reconstructed.sentinel2_l2a
                                gt_s2 = masked_sample.sentinel2_l2a
                                
                                if recon_s2.ndim == 7:
                                    continue
                                
                                recon_month = recon_s2[:, :, :, masked_month_indices, :]
                                gt_month = gt_s2[:, :, :, masked_month_indices, :]
                                
                                if recon_month.shape[1] != gt_month.shape[1] or recon_month.shape[2] != gt_month.shape[2]:
                                    min_h = min(recon_month.shape[1], gt_month.shape[1])
                                    min_w = min(recon_month.shape[2], gt_month.shape[2])
                                    recon_month = recon_month[:, :min_h, :min_w, :, :]
                                    gt_month = gt_month[:, :min_h, :min_w, :, :]
                                
                                recon_month_flat = recon_month.reshape(recon_month.shape[0], -1, recon_month.shape[-1])
                                gt_month_flat = gt_month.reshape(gt_month.shape[0], -1, gt_month.shape[-1])
                                
                                if recon_month_flat.numel() > 1e8:
                                    continue
                                
                                recon_for_loss = recon_month_flat.unsqueeze(2)
                                gt_for_loss = gt_month_flat.unsqueeze(2)
                                
                                correlation_loss, channel_correlations = compute_pairwise_correlation_loss(
                                    recon_for_loss, gt_for_loss, return_channel_correlations=True
                                )
                                
                                val_loss += correlation_loss.item()
                                val_samples += 1
                                
                                s2_band_names = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09']
                                corr_str = " | ".join([f"{s2_band_names[i]}: {c:+.3f}" for i, c in enumerate(channel_correlations[:12])])
                                self.logger.info(f"    Val - Loss={correlation_loss.item():.6f} | {corr_str}")
                                
                                del masked_sample, encoder_output, decoder_output, pooled_output, reconstructed
                                del recon_month, gt_month, recon_month_flat, gt_month_flat
                                torch.cuda.empty_cache()
                                gc.collect()
            
            model.reconstructor.train()  # Back to training mode
            
            avg_val_loss = val_loss / max(1, val_samples)
            self.logger.info(f"Epoch {epoch+1} VAL: avg_loss={avg_val_loss:.6f} ({val_samples} samples)")
            
            # Log to WandB
            if wandb_enabled:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "train_samples": train_samples,
                    "val_samples": val_samples,
                })

            # Save checkpoint
            checkpoint_path = Path(self.config.checkpoint_dir) / f"reconstructor_epoch_{epoch+1}.pt"
            torch.save(model.reconstructor.state_dict(), checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            
            # Visualize reconstructions
            if target_images and reconstructed_images:
                vis_path = visualize_reconstruction_pairs(target_images, reconstructed_images, epoch+1, self.config.checkpoint_dir)
                self.logger.info(f"Saved visualization: {vis_path}")
                
                # Log visualization to WandB
                if wandb_enabled:
                    wandb.log({
                        "reconstruction_visualization": wandb.Image(str(vis_path))
                    })
        
        self.logger.info("="*80)
        self.logger.info("Training Complete")
        self.logger.info("="*80)
        
        # Finish WandB run
        if wandb_enabled:
            wandb.finish()
    
    def loop_through_data(self):
        """Loop through training data and verify loading.

        Just verify we can iterate through the dataset successfully.
        """
        self.logger.info("="*80)
        self.logger.info("Looping through training data")
        self.logger.info("="*80)
        
        # Use a subset of samples
        num_samples = min(self.config.num_samples_per_epoch, len(self.dataset))

        self.logger.info(f"Processing {num_samples} samples from dataset of {len(self.dataset)} total")
        self.logger.info("")

        successful_loads = 0
        failed_loads = 0
        modality_counts = {}

        start_time = time.time()

        for sample_idx in range(num_samples):
            sample_start = time.time()

            # Load sample
            loaded_idx, sample = self._get_sample(sample_idx)
            
            load_time = time.time() - sample_start
            
            if sample is None:
                failed_loads += 1
                self.logger.warning(f"Sample {sample_idx+1}/{num_samples}: FAILED to load (time: {load_time:.2f}s)")
                continue
            
            successful_loads += 1
            
            # Log what modalities are available
            modalities_present = []
            for modality_name in self.config.supported_modality_names:
                if hasattr(sample, modality_name) and getattr(sample, modality_name) is not None:
                    modalities_present.append(modality_name)
                    modality_counts[modality_name] = modality_counts.get(modality_name, 0) + 1
            
            self.logger.info(
                f"Sample {sample_idx+1}/{num_samples}: ✓ Loaded in {load_time:.2f}s | "
                f"Modalities: {', '.join(modalities_present)}"
            )

        # Summary
        elapsed = time.time() - start_time
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("Stage 1 Summary")
        self.logger.info("="*80)
        self.logger.info(f"Total time: {elapsed:.1f}s")
        self.logger.info(f"Successful loads: {successful_loads}/{num_samples} ({100*successful_loads/num_samples:.1f}%)")
        self.logger.info(f"Failed loads: {failed_loads}/{num_samples}")
        self.logger.info(f"Average load time per sample: {elapsed/num_samples:.2f}s")
        self.logger.info("")
        self.logger.info("Modality availability:")
        for modality_name in self.config.supported_modality_names:
            count = modality_counts.get(modality_name, 0)
            pct = 100 * count / successful_loads if successful_loads > 0 else 0
            self.logger.info(f"  - {modality_name}: {count}/{successful_loads} samples ({pct:.1f}%)")
        self.logger.info("")
        self.logger.info("="*80)
        self.logger.info("Stage 1 Complete - Ready to proceed to training")
        self.logger.info("="*80)


if __name__ == "__main__":

    config = ReconstructorTrainingConfig()
    trainer = ReconstructorTrainer(config)
    trainer.loop_through_data()
    trainer.train()
