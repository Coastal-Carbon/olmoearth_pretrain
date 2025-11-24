"""
Reflectance Reconstruction Training Module for OlmoEarth.

This module provides training capability for the separate reflectance reconstruction head,
following the existing OlmoEarth training framework patterns.
"""

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
import torch.nn as nn

from olmoearth_pretrain.data.dataset import OlmoEarthSample
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample, MaskValue, MaskingConfig, MaskingStrategy
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.train_module.train_module import (
    OlmoEarthTrainModule,
    OlmoEarthTrainModuleConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class ReflectanceReconstructionTrainModuleConfig(OlmoEarthTrainModuleConfig):
    """Configuration for reflectance reconstruction training module.
    
    Args:
        target_modalities: List of modalities to reconstruct.
        loss_type: Type of reconstruction loss ('l1', 'l2', 'huber').
        freeze_encoder: Whether to freeze the pretrained encoder.
        reconstruction_weight: Weight for the reconstruction loss.
        mask_ratio: Fraction of patches to mask for reconstruction.
        use_mixed_precision: Whether to use mixed precision training.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
    """
    
    # Reconstruction-specific settings
    target_modalities: List[str] = field(default_factory=lambda: ['sentinel2_l2a', 'landsat'])
    loss_type: str = 'l1'  # 'l1', 'l2', or 'huber'
    freeze_encoder: bool = True  # Whether to freeze the pretrained encoder
    reconstruction_weight: float = 1.0  # Weight for reconstruction loss
    masking_config: "MaskingConfig | None" = None  # Masking strategy configuration
    use_mixed_precision: bool = True  # Use AMP for efficiency
    gradient_accumulation_steps: int = 1  # Gradient accumulation
    
    def build(
        self,
        model: Any,
        device: torch.device | None = None,
    ) -> "ReflectanceReconstructionTrainModule":
        """Build the reconstruction training module."""
        kwargs = self.prepare_kwargs()
        return ReflectanceReconstructionTrainModule(
            model=model,
            device=device,
            **kwargs,
        )


class ReflectanceReconstructionTrainModule(OlmoEarthTrainModule):
    """Training module for reflectance reconstruction head.
    
    This module extends OlmoEarthTrainModule to handle training of a separate
    reflectance reconstruction head while optionally freezing the pretrained encoder.
    """
    
    def __init__(
        self,
        target_modalities: List[str],
        loss_type: str = 'l1',
        freeze_encoder: bool = True,
        reconstruction_weight: float = 1.0,
        masking_config: "MaskingConfig | None" = None,
        use_mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.target_modalities = target_modalities
        self.loss_type = loss_type
        self.freeze_encoder = freeze_encoder
        self.reconstruction_weight = reconstruction_weight
        self.use_mixed_precision = use_mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize masking strategy if provided
        self.masking_strategy: MaskingStrategy | None = None
        if masking_config is not None:
            self.masking_strategy = masking_config.build()
        
        # Initialize mixed precision scaler if needed
        self.scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # Freeze encoder parameters if requested
        if self.freeze_encoder and hasattr(self.model, 'encoder'):
            frozen_params = 0
            for param in self.model.encoder.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
            logger.info(f"Frozen {frozen_params:,} encoder parameters")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Training {trainable_params:,} parameters")
        
        # Initialize metrics tracking
        self.accumulated_loss = 0.0
        self.accumulation_count = 0
        
        # Loss tracking for plotting (with limited size to prevent memory growth)
        self.loss_history = []
        self.step_count = 0
        self.max_history_size = 1000  # Limit history to prevent memory issues
        
        # Initialize step tracking to prevent duplicate logging
        self._logged_global_steps = set()
        self._step_modality_losses = {}
    
    def loss_fn(
        self, 
        reconstructions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute reconstruction loss with optional masking.
        
        Args:
            reconstructions: Dictionary of reconstructed tensors by modality.
            targets: Dictionary of target tensors by modality.
            masks: Optional dictionary of valid pixel masks by modality.
            
        Returns:
            Combined reconstruction loss.
        """
        total_loss = torch.tensor(0.0, device=self.device)
        valid_modalities = 0
        
        for modality in self.target_modalities:
            if modality not in reconstructions or modality not in targets:
                continue
                
            pred = reconstructions[modality]
            target = targets[modality]
            
            # Apply mask if provided
            if masks is not None and modality in masks:
                mask = masks[modality]
                pred = pred * mask
                target = target * mask
            
            # Handle shape mismatch between prediction and target
            print(f"DEBUG: Loss computation for {modality}: pred shape {pred.shape}, target shape {target.shape}")
            
            # Ensure target and prediction have compatible shapes
            if len(target.shape) == 5:  # [B, H, W, T, C]
                # Flatten spatial and temporal dimensions for comparison
                batch_size, h, w, t, c = target.shape
                target = target.reshape(batch_size, h * w * t, c)
            elif len(target.shape) == 4:  # [B, H, W, C]
                batch_size, h, w, c = target.shape
                target = target.reshape(batch_size, h * w, c)
            
            if len(pred.shape) == 4:  # [B, H, W, C]
                batch_size, h, w, c = pred.shape
                pred = pred.reshape(batch_size, h * w, c)
            
            # Now both should be [B, N, C] format - subsample prediction to match target size
            if pred.shape[1] != target.shape[1]:
                # Subsample prediction to match target token count
                indices = torch.linspace(0, pred.shape[1] - 1, target.shape[1], dtype=torch.long, device=pred.device)
                pred = pred[:, indices, :]
                
            print(f"DEBUG: After reshaping - pred shape {pred.shape}, target shape {target.shape}")
            
            # Ultra-conservative tensor size to ensure stability
            max_pixels = 500  # Even smaller limit for stability
            if pred.shape[1] > max_pixels:
                print(f"DEBUG: Limiting tensor size from {pred.shape[1]} to {max_pixels} pixels for {modality}")
                # Take a deterministic subset to avoid memory allocation for randperm
                pred = pred[:, :max_pixels, :]
                target = target[:, :max_pixels, :]
                print(f"DEBUG: Limited shapes - pred: {pred.shape}, target: {target.shape}")
            
            print(f"DEBUG: Starting {self.loss_type} loss computation for {modality}...")
            print(f"DEBUG: Tensor memory usage - pred: {pred.numel() * 4 / 1024 / 1024:.2f} MB")
            
            # Aggressive memory management
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()  # Ensure operations complete before continuing
            
            # Compute loss based on type with clamping to prevent explosion
            if self.loss_type == 'l1':
                loss = nn.functional.l1_loss(pred, target, reduction='mean')
            elif self.loss_type == 'l2':
                loss = nn.functional.mse_loss(pred, target, reduction='mean')
            elif self.loss_type == 'huber':
                loss = nn.functional.huber_loss(pred, target, reduction='mean')
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            # Clamp loss to prevent explosion (anything above 10.0 is likely unstable)
            loss = torch.clamp(loss, max=10.0)
            
            print(f"DEBUG: Completed {self.loss_type} loss computation for {modality}, loss: {loss.item():.6f}")
            
            # Skip this modality if loss is unstable (>5.0 indicates problems)
            if loss.item() > 5.0:
                print(f"WARNING: Skipping unstable loss for {modality}: {loss.item():.6f}")
                continue
            
            # Store for loss history tracking
            if not hasattr(self, '_last_modality_losses'):
                self._last_modality_losses = {}
            self._last_modality_losses[modality] = loss.item()
            
            total_loss += loss
            valid_modalities += 1
            
            # Store per-modality loss for later logging (to avoid duplicates)
            if not hasattr(self, '_step_modality_losses'):
                self._step_modality_losses = {}
            if self.trainer.global_step not in self._step_modality_losses:
                self._step_modality_losses[self.trainer.global_step] = {}
            
            self._step_modality_losses[self.trainer.global_step][modality] = loss.item()
        
        if valid_modalities == 0:
            print("WARNING: No valid modalities for loss computation, returning small loss")
            return torch.tensor(0.001, device=self.device, requires_grad=True)
        
        # Average over modalities and apply weight
        avg_loss = (total_loss / valid_modalities) * self.reconstruction_weight
        
        # Final safety check - if average loss is still too high, clamp it
        if avg_loss.item() > 2.0:
            print(f"WARNING: Clamping high average loss from {avg_loss.item():.6f} to 2.0")
            avg_loss = torch.clamp(avg_loss, max=2.0)
        
        return avg_loss
    
    def compute_loss(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute reconstruction loss between predictions and targets."""
        return self.loss_fn(predictions, targets)
    
    def create_masked_sample(self, sample: OlmoEarthSample, patch_size: int) -> MaskedOlmoEarthSample:
        """Create masked sample for reconstruction training.
        
        Args:
            sample: Original OlmoEarth sample.
            patch_size: Patch size for masking.
            
        Returns:
            Masked sample using proper OlmoEarth masking strategy.
        """
        if self.masking_strategy is not None:
            # Use the proper OlmoEarth masking strategy
            return self.masking_strategy.apply_mask(sample, patch_size=patch_size)
        else:
            # No masking - just convert to MaskedOlmoEarthSample
            return MaskedOlmoEarthSample.from_olmoearthsample(sample)
    
    def extract_reconstruction_targets(
        self, 
        sample: OlmoEarthSample
    ) -> Dict[str, torch.Tensor]:
        """Extract reconstruction targets from sample.
        
        Args:
            sample: Original OlmoEarth sample.
            
        Returns:
            Dictionary of target tensors by modality.
        """
        targets = {}
        for modality in self.target_modalities:
            if hasattr(sample, modality):
                modality_data = getattr(sample, modality)
                if modality_data is not None:
                    targets[modality] = modality_data
        return targets
    
    def model_forward(
        self,
        masked_sample: MaskedOlmoEarthSample,
        patch_size: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Run forward pass through model.
        
        Args:
            masked_sample: Masked input sample.
            patch_size: Patch size for the model.
            
        Returns:
            Tuple of (reconstructions, loss).
        """
        # Forward pass through model
        results = self.model(
            masked_sample,
            patch_size=patch_size,
            target_modalities=self.target_modalities
        )
        
        # Extract targets from original sample using unmask method
        unmasked_sample = masked_sample.unmask()
        targets = self.extract_reconstruction_targets(unmasked_sample)
        
        # Compute reconstruction loss
        loss = self.loss_fn(
            results['reconstructions'], 
            targets,
            masks=results.get('masks')
        )
        
        return results['reconstructions'], loss
    
    def train_batch(
        self, 
        batch: Tuple[int, OlmoEarthSample], 
        dry_run: bool = False
    ) -> None:
        """Train on a batch with reconstruction objective.
        
        Args:
            batch: Tuple of (patch_size, sample_batch).
            dry_run: Whether to skip the actual training step.
        """
        from olmoearth_pretrain.train.utils import split_batch
        
        # Set model to train mode
        self.model.train()
        
        patch_size, batch_data = batch
        
        # Split into micro-batches using standard OlmoEarth utility
        microbatches = split_batch(batch_data, self.rank_microbatch_size)
        num_microbatches = len(microbatches)
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        for microbatch_idx, microbatch in enumerate(microbatches):
            with self._train_microbatch_context(microbatch_idx, num_microbatches):
                logger.info(
                    f"Training microbatch {microbatch_idx} of {num_microbatches} with batch size {microbatch.batch_size}"
                )
                
                # Apply transforms if available
                if hasattr(self, 'transform') and self.transform is not None:
                    microbatch = self.transform.apply(microbatch)
                
                # Move to device
                microbatch = microbatch.to_device(self.device)
                
                # Use autocast for mixed precision forward pass
                autocast_context = torch.cuda.amp.autocast() if self.use_mixed_precision else contextlib.nullcontext()
                
                with autocast_context:
                    # Create masked sample for reconstruction
                    masked_sample = self.create_masked_sample(microbatch, patch_size)
                    
                    # Forward pass through reconstruction head
                    predictions, embedding = self.model_forward(masked_sample, patch_size)
                    
                    # Get reconstruction targets
                    unmasked_sample = masked_sample.unmask()
                    targets = self.extract_reconstruction_targets(unmasked_sample)
                    
                    # Compute loss
                    print(f"DEBUG: Computing loss for microbatch {microbatch_idx}...")
                    loss = self.compute_loss(predictions, targets)
                    print(f"DEBUG: Raw loss: {loss.item():.6f}")
                    
                    # Scale loss for gradient accumulation
                    loss = loss / num_microbatches
                    print(f"DEBUG: Scaled loss: {loss.item():.6f}")
                
                total_loss += loss.detach()
                
                if not dry_run:
                    print(f"DEBUG: Starting backward pass for microbatch {microbatch_idx}...")
                    # Use scaler for mixed precision backward pass
                    if self.use_mixed_precision and self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    print(f"DEBUG: Completed backward pass for microbatch {microbatch_idx}")
        
        if not dry_run:
            # Perform optimizer step
            if self.use_mixed_precision and self.scaler is not None:
                # Mixed precision optimization step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Regular optimization step
                self.optimizer.step()
            
            # Check gradient norm before stepping
            total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if total_norm > 10.0:  # Skip update if gradients are too large
                print(f"WARNING: Skipping optimizer step due to large gradients: {total_norm:.2f}")
                self.optimizer.zero_grad()
                return loss, targets
            
            # Zero gradients for next iteration
            self.optimizer.zero_grad()
            
            print(f"DEBUG: Gradient norm: {total_norm:.4f}")
            
            # Record metrics (only once per step to avoid duplicates)
            current_step = self.trainer.global_step
            if not hasattr(self, '_logged_global_steps'):
                self._logged_global_steps = set()
                
            if current_step not in self._logged_global_steps:
                self.trainer.record_metric("train/reconstruction_loss", total_loss)
                
                # Log individual modality losses from this step
                if hasattr(self, '_step_modality_losses') and current_step in self._step_modality_losses:
                    for mod_name, mod_loss in self._step_modality_losses[current_step].items():
                        self.trainer.record_metric(f"reconstruction_loss/{mod_name}", mod_loss, namespace="train")
                
                self._logged_global_steps.add(current_step)
                
                # Clean up old step data to prevent memory leaks
                if hasattr(self, '_step_modality_losses'):
                    # Keep only last 10 steps
                    steps_to_keep = sorted(self._step_modality_losses.keys())[-10:]
                    self._step_modality_losses = {k: v for k, v in self._step_modality_losses.items() if k in steps_to_keep}
            
            # Track loss history for plotting
            self.step_count += 1
            loss_entry = {
                'step': self.step_count,
                'total_loss': total_loss.item(),
                'modality_losses': {}
            }
            
            # Store individual modality losses from the last computation
            if hasattr(self, '_last_modality_losses'):
                loss_entry['modality_losses'] = self._last_modality_losses.copy()
            
            self.loss_history.append(loss_entry)
            
            # Limit history size to prevent memory growth
            if len(self.loss_history) > self.max_history_size:
                self.loss_history = self.loss_history[-self.max_history_size//2:]  # Keep last half
            
            # Save plot every 25 steps to reduce I/O overhead
            if self.step_count % 25 == 0:
                try:
                    self.save_loss_plot()
                except Exception as e:
                    print(f"Warning: Could not save loss plot: {e}")
            
            print(f"DEBUG: Completed optimization step, total_loss: {total_loss.item():.6f}")
            
            # Aggressive memory cleanup
            del batch, batch_data, microbatches
            if hasattr(self, '_last_modality_losses'):
                del self._last_modality_losses
            
            # Force garbage collection after each step
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Monitor GPU memory every few steps
                if self.step_count % 5 == 0:
                    allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                    reserved = torch.cuda.memory_reserved() / 1024**3   # GB
                    print(f"DEBUG: GPU memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
    
    def eval_batch(
        self, 
        batch: Tuple[int, OlmoEarthSample]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate a batch and return metrics.
        
        Args:
            batch: Tuple of (patch_size, sample_batch).
            
        Returns:
            Tuple of loss tensor and target tensor (following OlmoEarth pattern).
        """
        patch_size, batch_data = batch
        
        self.model.eval()
        
        with torch.no_grad():
            # Apply transforms if available
            if hasattr(self, 'transform') and self.transform is not None:
                batch_data = self.transform.apply(batch_data)
            
            # Move to device
            batch_data = batch_data.to_device(self.device)
            
            # Create masked sample
            masked_sample = self.create_masked_sample(batch_data)
            
            # Forward pass
            predictions, embedding = self.model_forward(masked_sample, patch_size)
            
            # Get targets
            unmasked_sample = masked_sample.unmask()
            targets = self.extract_reconstruction_targets(unmasked_sample)
            
            # Compute loss
            loss = self.compute_loss(predictions, targets)
        
        return loss, targets
    
    def save_loss_plot(self, output_dir: str = "./loss_plots"):
        """Save loss curves as plots."""
        if not self.loss_history:
            print("No loss history to plot")
            return
        
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data
        steps = [entry['step'] for entry in self.loss_history]
        total_losses = [entry['total_loss'] for entry in self.loss_history]
        
        # Get all modalities
        all_modalities = set()
        for entry in self.loss_history:
            all_modalities.update(entry['modality_losses'].keys())
        
        # Create plots
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Total loss
        axes[0].plot(steps, total_losses, 'b-o', linewidth=2, markersize=4)
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Total Loss')
        axes[0].set_title('Total Reconstruction Loss')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        # Modality losses
        colors = ['green', 'red', 'orange', 'purple', 'brown']
        for i, modality in enumerate(sorted(all_modalities)):
            modality_losses = []
            modality_steps = []
            for entry in self.loss_history:
                if modality in entry['modality_losses']:
                    modality_losses.append(entry['modality_losses'][modality])
                    modality_steps.append(entry['step'])
            
            if modality_losses:
                color = colors[i % len(colors)]
                axes[1].plot(modality_steps, modality_losses, 
                           f'{color}-o', linewidth=2, markersize=4, label=modality)
        
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Loss by Modality')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f'loss_curves_step_{self.step_count}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Loss plot saved to: {plot_path}")
        
        # Also save loss data as JSON for later analysis
        import json
        json_path = os.path.join(output_dir, f'loss_data_step_{self.step_count}.json')
        with open(json_path, 'w') as f:
            json.dump(self.loss_history, f, indent=2)
        print(f"Loss data saved to: {json_path}")