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
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample
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
        
        # Store normalization stats for inference denormalization
        self.norm_stats = None  # Will be set from dataset normalizers
        
        # Initialize mixed precision scaler if needed
        self.scaler = torch.cuda.amp.GradScaler() if (use_mixed_precision and torch.cuda.is_available()) else None
        
        # Safety: disable mixed precision if CUDA not available
        if use_mixed_precision and not torch.cuda.is_available():
            self.use_mixed_precision = False
            logger.warning("Mixed precision disabled: CUDA not available")
        
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
        
    def set_normalization_stats(self, normalizer_computed, normalizer_predefined):
        """Set normalization statistics for target normalization and inference denormalization."""
        self.normalizer_computed = normalizer_computed
        self.normalizer_predefined = normalizer_predefined
        logger.info("Normalization stats set for inference-ready model")
        
    def denormalize_predictions(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Denormalize predictions back to original satellite reflectance units for inference."""
        if not hasattr(self, 'normalizer_computed') or not hasattr(self, 'normalizer_predefined'):
            logger.warning("No normalization stats available for denormalization")
            return predictions
            
        from olmoearth_pretrain.data.constants import Modality
        
        denormalized_preds = {}
        for modality, pred in predictions.items():
            try:
                modality_spec = Modality.get(modality)
                pred_np = pred.detach().cpu().numpy()
                
                # Apply inverse normalization (computed first, then predefined fallback)
                try:
                    denorm_np = self.normalizer_computed.denormalize(modality_spec, pred_np)
                    logger.debug(f"Applied computed denormalization to {modality} predictions")
                except (KeyError, ValueError, AttributeError) as e:
                    logger.debug(f"Computed denormalization failed for {modality}: {e}, using predefined")
                    denorm_np = self.normalizer_predefined.denormalize(modality_spec, pred_np)
                
                denormalized_preds[modality] = torch.from_numpy(denorm_np).to(pred.device, dtype=pred.dtype)
            except Exception as e:
                logger.warning(f"Could not denormalize {modality}: {e}, keeping normalized")
                denormalized_preds[modality] = pred
                
        return denormalized_preds
    
    def loss_fn(
        self, 
        reconstructions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor],
        masks: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Simple reconstruction loss following OlmoEarth patterns.

        Args:
            reconstructions: Dictionary of reconstructed tensors by modality.
            targets: Dictionary of target tensors by modality.  
            masks: Optional dictionary of valid pixel masks (unused).
            
        Returns:
            Combined reconstruction loss.
        """
        import torch.nn.functional as F
        
        total_loss = 0.0
        valid_modalities = 0
        
        for modality in self.target_modalities:
            if modality not in reconstructions or modality not in targets:
                continue
                
            pred = reconstructions[modality]  # May be [B, H, W, C] or [B, H, W, T, C]
            target = targets[modality]        # [B, H, W, T, C] or [B, H, W, C]
            
            # Handle temporal dimension properly - preserve time structure for missing data reconstruction
            if target.dim() == 5 and pred.dim() == 4:  # Target has time, pred doesn't
                # Prediction is spatially consistent - expand to match target temporal dimension
                pred = pred.unsqueeze(3).repeat(1, 1, 1, target.shape[3], 1)  # [B, H, W, T, C]
            elif target.dim() == 4 and pred.dim() == 5:  # Pred has time, target doesn't
                # This shouldn't happen in supervised reconstruction, but handle gracefully
                pred = pred.mean(dim=3)  # [B, H, W, C]
            
            # Handle spatial mismatches using interpolation
            if pred.shape[:3] != target.shape[:3]:
                # Handle both 4D and 5D tensors for spatial interpolation
                if pred.dim() == 5:  # [B, H, W, T, C]
                    # Reshape to [B*T, C, H, W] for interpolation
                    B, H, W, T, C = pred.shape
                    pred_reshaped = pred.permute(0, 3, 4, 1, 2).reshape(B*T, C, H, W)
                    
                    BT, HT, WT, TT, CT = target.shape
                    target_reshaped = target.permute(0, 3, 4, 1, 2).reshape(BT*TT, CT, HT, WT)
                else:  # [B, H, W, C]
                    pred_reshaped = pred.permute(0, 3, 1, 2)
                    target_reshaped = target.permute(0, 3, 1, 2)
                
                if pred_reshaped.shape[-2:] != target_reshaped.shape[-2:]:
                    pred_reshaped = F.interpolate(
                        pred_reshaped.float(),
                        size=target_reshaped.shape[-2:],
                        mode='bilinear',
                        align_corners=True
                    )
                
                # Reshape back to original format
                if pred.dim() == 5:  # [B, H, W, T, C]
                    pred = pred_reshaped.reshape(B, T, C, target_reshaped.shape[-2], target_reshaped.shape[-1]).permute(0, 3, 4, 1, 2)
                    target = target_reshaped.reshape(BT, TT, CT, target_reshaped.shape[-2], target_reshaped.shape[-1]).permute(0, 3, 4, 1, 2)
                else:  # [B, H, W, C]
                    pred = pred_reshaped.permute(0, 2, 3, 1)
                    target = target_reshaped.permute(0, 2, 3, 1)
            
            # CRITICAL: Filter missing values BEFORE normalization to prevent extreme outliers
            from olmoearth_pretrain.data.constants import MISSING_VALUE, SENTINEL1_NODATA
            
            # Create comprehensive missing value mask - check for common satellite missing values
            missing_value_mask = (
                (target == MISSING_VALUE) |           # OlmoEarth missing value (-99999)
                (target == SENTINEL1_NODATA) |        # Sentinel-1 no data (-32768)
                (torch.abs(target) > 50000) |         # Extreme outliers (likely missing data sentinels)
                (~torch.isfinite(target)) |           # NaN/Inf values
                (~torch.isfinite(pred))               # Ensure predictions are also finite
            )
            valid_mask = ~missing_value_mask
            
            # Store original range for logging if needed
            if valid_mask.any():
                valid_target_orig = target[valid_mask]
                original_target_min = valid_target_orig.min().item()
                original_target_max = valid_target_orig.max().item()
            else:
                original_target_min = float('nan')
                original_target_max = float('nan')
            
            # Use targets as-is (dataset should have normalized them)
            target_normalized = target
            
            if not valid_mask.any():
                logger.warning(f"No valid pixels for {modality}, skipping")
                continue
            
            # Extract valid values for loss computation (using normalized targets)
            valid_pred = pred[valid_mask]
            valid_target_norm = target_normalized[valid_mask]
            valid_target_orig = target[valid_mask]  # Keep original for logging
            

            
            # Compute loss only on valid pixels (using normalized targets)
            if self.loss_type == 'l1':
                loss = F.l1_loss(valid_pred, valid_target_norm)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(valid_pred, valid_target_norm)
            elif self.loss_type == 'huber':
                loss = F.huber_loss(valid_pred, valid_target_norm)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
            
            # Report high losses but let the model learn naturally
            if loss.item() > 5000:  # Only log very high losses
                logger.info(f"📊 {modality} loss: {loss.item():.1f} (valid pixels: {100*valid_mask.float().mean():.1f}%)")
            
            if torch.isfinite(loss):
                total_loss += loss
                valid_modalities += 1
        
        if valid_modalities == 0:
            return torch.tensor(0.0, requires_grad=True, device=self.device)
        
        final_loss = (total_loss / valid_modalities) * self.reconstruction_weight
        
        # Let large losses happen - this is normal learning!
        if final_loss.item() > 50000:  # Only flag truly extreme cases
            logger.info(f"📈 High loss during learning: {final_loss.item():.1f}")
            logger.info(f"   Reconstruction head is learning - this should decrease over time")
            logger.info(f"   Training {valid_modalities} modalities: {self.target_modalities}")
            
        return final_loss
    
    def compute_loss(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute reconstruction loss between predictions and targets."""
        return self.loss_fn(predictions, targets)
    

    
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
        sample: OlmoEarthSample,
        patch_size: int,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Run forward pass through model using simple supervised learning pattern.
        
        Args:
            sample: Regular OlmoEarth sample.
            patch_size: Patch size for the model.
            
        Returns:
            Tuple of (reconstructions, loss).
        """
        # Forward pass - the reconstruction model handles masking and fast_pass internally
        results = self.model(
            sample,
            patch_size=patch_size,
            target_modalities=self.target_modalities
        )
        
        # Extract targets from original sample
        targets = self.extract_reconstruction_targets(sample)
        
        # Compute reconstruction loss
        loss = self.loss_fn(
            results['reconstructions'], 
            targets
        )
        
        return results['reconstructions'], loss
    
    def train_batch(
        self, 
        batch: Tuple[int, OlmoEarthSample], 
        dry_run: bool = False
    ) -> None:
        """Clean reconstruction training following evaluation task patterns.
        
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
                # Apply transforms if available
                if hasattr(self, 'transform') and self.transform is not None:
                    microbatch = self.transform.apply(microbatch)
                
                # Move to device
                microbatch = microbatch.to_device(self.device)
                
                # Use autocast for mixed precision
                autocast_context = torch.cuda.amp.autocast() if self.use_mixed_precision else contextlib.nullcontext()
                
                with autocast_context:
                    # Forward pass using evaluation pattern
                    predictions, loss = self.model_forward(microbatch, patch_size)
                    
                    # Scale loss for gradient accumulation
                    loss = loss / num_microbatches
                
                total_loss += loss.detach()
                
                if not dry_run:
                    # Backward pass
                    if self.use_mixed_precision and self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
        
        if not dry_run:
            # Gradient clipping and optimizer step
            if self.use_mixed_precision and self.scaler is not None:
                try:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Step with scaler (this checks for inf/nan)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                except AssertionError as e:
                    if "No inf checks were recorded" in str(e):
                        logger.error(f"Mixed precision error: {e}")
                        logger.error("Falling back to FP32 for this step")
                        # Fallback to regular optimization
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                    else:
                        raise
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()            # Zero gradients
            self.optimizer.zero_grad()
            
            # Record metrics with higher precision
            self.trainer.record_metric("train/reconstruction_loss", total_loss)
            self.trainer.record_metric("optim/total_grad_norm", float(grad_norm))
            
            # Debug logging with high precision (every 100 steps to avoid spam)
            if hasattr(self.trainer, 'state') and self.trainer.state.global_step % 100 == 0:
                print(f"[DEBUG] Step {self.trainer.state.global_step}: grad_norm = {grad_norm:.8f}, loss = {total_loss:.8f}")
    
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
            
            # Forward pass using evaluation pattern
            predictions, loss = self.model_forward(batch_data, patch_size)
            
            # Get targets for return value
            targets = self.extract_reconstruction_targets(batch_data)
            
        return loss, targets
    
