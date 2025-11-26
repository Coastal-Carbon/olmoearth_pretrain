"""Simple validation evaluator callback for reconstruction training."""

import logging
from dataclasses import dataclass
from typing import Any

import torch
from olmo_core.train.callbacks.callback import Callback, CallbackConfig
from olmo_core.train.common import Duration
from olmo_core.train.trainer import Trainer
from torch.utils.data import DataLoader

from olmoearth_pretrain.data.concat import OlmoEarthConcatDatasetConfig
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig, collate_olmoearth_pretrain
from olmoearth_pretrain.train.callbacks.wandb import OlmoEarthWandBCallback

logger = logging.getLogger(__name__)


@dataclass
class ValidationEvaluatorCallback(Callback):
    """Runs validation evaluation periodically during training."""

    eval_interval: Duration
    validation_dataset_config: Any  # Store dataset config to build validation dataloader
    validation_dataloader_config: Any  # Store dataloader config
    validation_dataloader: DataLoader = None  # Will be built lazily

    def _build_validation_dataloader(self):
        """Build the validation dataloader from the configs."""
        if self.validation_dataloader is not None:
            return
        
        try:
            # Build validation dataset (just the second dataset from concat config)
            if isinstance(self.validation_dataset_config, OlmoEarthConcatDatasetConfig):
                # Get the validation dataset (second one in the concat config)
                val_dataset_config = self.validation_dataset_config.dataset_configs[1]
                val_dataset = val_dataset_config.build()
            else:
                logger.warning("Expected OlmoEarthConcatDatasetConfig for validation")
                return
                
            # Build validation dataloader with same config as training but for validation data
            self.validation_dataloader = self.validation_dataloader_config.build(
                val_dataset,
                collator=collate_olmoearth_pretrain,
                dp_process_group=self.trainer.train_module.dp_process_group,
            )
            logger.info("Built validation dataloader successfully")
            
        except Exception as e:
            logger.error(f"Failed to build validation dataloader: {e}")
            self.validation_dataloader = None

    def post_step(self) -> None:
        """Run validation evaluation in-loop."""
        eval_interval_steps = self.trainer.convert_duration_to_steps(self.eval_interval)
        
        # Skip if not time for evaluation
        if self.step <= 1 or self.step % eval_interval_steps != 0:
            return

        # Build validation dataloader if not built yet
        self._build_validation_dataloader()
        if self.validation_dataloader is None:
            logger.warning("No validation dataloader available, skipping validation")
            return

        logger.info(f"Running validation evaluation at step {self.step}")
        
        # Set model to eval mode
        self.trainer.train_module.model.eval()
        
        # Initialize the validation dataloader for this epoch
        self.validation_dataloader.reshuffle()
        
        total_val_loss = torch.zeros([], device=self.trainer.device)
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.validation_dataloader):
                try:
                    # Use the eval_batch method from the train module
                    val_loss, _ = self.trainer.train_module.eval_batch(batch)
                    if val_loss is not None:
                        total_val_loss += val_loss.detach()
                        num_batches += 1
                    
                    # Limit validation to just 3 batches for speed while maintaining some stability
                    if batch_idx >= 2:  # Validate on 3 batches (0, 1, 2)
                        break
                        
                except Exception as e:
                    logger.warning(f"Validation batch {batch_idx} failed: {e}")
                    continue
        
        # Set model back to train mode
        self.trainer.train_module.model.train()
        
        if num_batches > 0:
            avg_val_loss = total_val_loss / num_batches
            
            # Record metrics to trainer (use train/ namespace to group with training loss)
            self.trainer.record_metric("train/val_loss", avg_val_loss)
            
            # Also log to W&B manually to ensure it appears
            try:
                wandb_callback = next(
                    callback
                    for callback in self.trainer._iter_callbacks()
                    if isinstance(callback, OlmoEarthWandBCallback)
                )
                if wandb_callback.enabled:
                    wandb_callback.wandb.log({
                        "train/val_loss": avg_val_loss.item()
                    })
            except StopIteration:
                logger.warning("No W&B callback found, validation metrics not logged to W&B")
            
            logger.info(f"Validation loss at step {self.step}: {avg_val_loss:.6f} (from {num_batches} batches)")
        else:
            logger.warning("No valid validation batches processed")



@dataclass
class ValidationEvaluatorCallbackConfig(CallbackConfig):
    """Config for the validation evaluator callback."""
    
    eval_interval: Duration
    dataset_config: Any = None  # Pass this from trainer config builder
    dataloader_config: Any = None  # Pass this from trainer config builder
    enabled: bool = True
    
    def build(self, trainer: Trainer) -> Callback | None:
        """Build the validation evaluator callback."""
        if not self.enabled:
            return None
        
        if self.dataset_config is None or self.dataloader_config is None:
            logger.warning("Dataset or dataloader config not provided, validation evaluation disabled")
            return None
        
        return ValidationEvaluatorCallback(
            eval_interval=self.eval_interval,
            validation_dataset_config=self.dataset_config,
            validation_dataloader_config=self.dataloader_config,
        )