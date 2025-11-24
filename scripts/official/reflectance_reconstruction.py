"""
Official training script for reflectance reconstruction head.

This script trains a separate reflectance reconstruction head on top of a pretrained
OlmoEarth encoder, following the established OlmoEarth training patterns.
"""

import logging
from pathlib import Path
from typing import List

import torch
from olmo_core.config import DType
from olmo_core.distributed.parallel.data_parallel import (
    DataParallelConfig,
    DataParallelType,
)
from olmo_core.optim import AdamWConfig
from olmo_core.optim.scheduler import CosWithWarmup
from olmo_core.train.callbacks import (
    CheckpointerCallback,
    ConfigSaverCallback,
    GarbageCollectorCallback,
    GPUMemoryMonitorCallback,
)
from olmo_core.train.checkpoint import CheckpointerConfig
from olmo_core.train.common import Duration, LoadStrategy
from olmo_core.train.config import TrainerConfig

from olmoearth_pretrain.data.constants import Modality
from olmoearth_pretrain.data.dataloader import OlmoEarthDataLoaderConfig
from olmoearth_pretrain.data.dataset import OlmoEarthDatasetConfig
from olmoearth_pretrain.internal.common import (
    build_common_components as build_common_components_default,
)
from olmoearth_pretrain.internal.experiment import (
    CommonComponents,
    OlmoEarthVisualizeConfig,
    SubCmd,
    main,
)
from olmoearth_pretrain.nn.flexi_vit import PoolingType
from olmoearth_pretrain.train.callbacks import (
    DownstreamEvaluatorCallbackConfig,
    OlmoEarthSpeedMonitorCallback,
    OlmoEarthWandBCallback,
)
from olmoearth_pretrain.train.callbacks.evaluator_callback import DownstreamTaskConfig
from olmoearth_pretrain.train.loss import LossConfig
from olmoearth_pretrain.train.masking import MaskingConfig

from olmoearth_pretrain.train.train_module.reflectance_reconstruction import (
    ReflectanceReconstructionTrainModuleConfig,
)
from olmoearth_pretrain.train.masking import MaskingConfig
from olmoearth_pretrain.model_loader import ModelID
from olmoearth_pretrain.nn.reflectance_reconstruction import (
    ReflectanceReconstructionConfig
)


def robust_olmoearth_collator(batch):
    """Custom collator that handles missing data in OlmoEarth samples."""
    # Extract samples from batch (handle tuples if present)
    samples = []
    for item in batch:
        if isinstance(item, tuple) and len(item) == 2:
            _, sample = item
            samples.append(sample)
        else:
            samples.append(item)
    
    # Use OlmoEarth's built-in collation method if available
    if samples and hasattr(samples[0], 'collate'):
        return samples[0].collate(samples)
    
    # Fallback to default collation
    from torch.utils.data import default_collate
    return default_collate(samples)

logger = logging.getLogger(__name__)

# Configuration constants  
MAX_PATCH_SIZE = 8
MIN_PATCH_SIZE = 1  # Match pretrained model's min patch size


def build_common_components(
    script: str,
    cmd: SubCmd,
    run_name: str,
    cluster: str,
    overrides: List[str],
) -> CommonComponents:
    """Build the common components for reflectance reconstruction experiment."""
    return build_common_components_default(script, cmd, run_name, cluster, overrides)


def build_model_config(common: CommonComponents) -> ReflectanceReconstructionConfig:
    """Build model config for reflectance reconstruction.
    
    This creates a model configuration that combines a pretrained OlmoEarth encoder
    with a new reflectance reconstruction head.
    
    Command-line overrides can be used to customize:
        model.pretrained_model_id="OLMOEARTH_V1_BASE"
        model.target_modalities="[\"sentinel2_l2a\",\"landsat\"]" 
        model.freeze_encoder=true
    """
    return ReflectanceReconstructionConfig(
        # Use model ID to load pretrained OlmoEarth model (default configuration)
        pretrained_model_id=ModelID.OLMOEARTH_V1_BASE,  # Override with: model.pretrained_model_id=...
        
        # Fallback to file path if needed (usually not used with ModelID)
        pretrained_encoder_path=None,
        
        # Reconstruction head configuration
        target_modalities=['sentinel2_l2a', 'landsat'],  # Override with: model.target_modalities=...
        freeze_encoder=True,  # Override with: model.freeze_encoder=...
        use_sigmoid_output=True,  # Override with: model.use_sigmoid_output=...
        hidden_size_multiplier=1.0,  # Override with: model.hidden_size_multiplier=...
        
        # No patch size overrides - use model's original settings
        override_min_patch_size=None,
        override_max_patch_size=None
    )


def build_train_module_config(
    common: CommonComponents,
) -> ReflectanceReconstructionTrainModuleConfig:
    """Build train module config for reflectance reconstruction."""
    
    # Optimizer configuration with conservative learning rate
    optim_config = AdamWConfig(
        lr=1e-5,  # Reduced from 1e-4 to prevent gradient explosion
        betas=(0.9, 0.999),  # More stable beta2
        eps=1e-8,  # Smaller epsilon for numerical stability
        weight_decay=1e-3,  # Reduced weight decay
    )
    
    return ReflectanceReconstructionTrainModuleConfig(
        # Standard OlmoEarth settings
        optim_config=optim_config,
        rank_microbatch_size=2,  # Reduced from 4 to 2 to prevent OOM
        compile_model=False,  # Disable compilation for now
        dp_config=None,  # Single device for now. Override with: train_module.dp_config=...
        autocast_precision=DType.float16,  # Override with: train_module.autocast_precision=...
        max_grad_norm=0.1,  # Much more aggressive gradient clipping
        
        # Scheduler configuration with longer warmup for stability
        scheduler=CosWithWarmup(
            warmup_steps=100,  # Shorter warmup for small experiments
            alpha_f=0.01,  # Don't decay LR too much
        ),
        
        # Masking configuration for reconstruction training
        masking_config=MaskingConfig(
            strategy_config={
                "type": "random",
                "encode_ratio": 0.3,  # Keep 30% of tokens visible to encoder 
                "decode_ratio": 0.7,  # Reconstruct 70% of tokens
            }
        ),
        
        # Reconstruction-specific settings
        target_modalities=['sentinel2_l2a', 'landsat'],  # Override with: train_module.target_modalities=...
        loss_type='l1',  # Override with: train_module.loss_type=...
        freeze_encoder=True,  # Override with: train_module.freeze_encoder=...
        reconstruction_weight=1.0,  # Override with: train_module.reconstruction_weight=...
        use_mixed_precision=True,  # Override with: train_module.use_mixed_precision=...
        gradient_accumulation_steps=1,  # Override with: train_module.gradient_accumulation_steps=...
    )


def build_dataloader_config(common: CommonComponents) -> OlmoEarthDataLoaderConfig:
    """Build dataloader config for reflectance reconstruction."""
    import tempfile
    
    return OlmoEarthDataLoaderConfig(
        work_dir=str(Path(tempfile.gettempdir()) / "olmoearth_reconstruction"),
        global_batch_size=4,  # Reduced batch size to prevent OOM
        seed=3622,  # Fixed seed. Override with: data_loader.seed=...
        num_workers=0,  # 0 workers to avoid multiprocessing issues. Override with: data_loader.num_workers=...
        prefetch_factor=2,
        min_patch_size=MIN_PATCH_SIZE,    # Match model's min_patch_size 
        max_patch_size=MAX_PATCH_SIZE,    # Match model's max_patch_size 
        sampled_hw_p_list=[4, 8],  # Height/width in patches: 128/32=4, 128/16=8
        token_budget=None,    # No token budget limit for reconstruction
        drop_last=False,      # Don't drop incomplete batches to handle missing data better
        shuffle=True,         # Shuffle training data
        num_dataset_repeats_per_epoch=1,  # Single pass through dataset per epoch
    )


def build_dataset_config(common: CommonComponents) -> OlmoEarthDatasetConfig:
    """Build dataset config for reflectance reconstruction."""
    return OlmoEarthDatasetConfig(
        # Dataset directory - using the user's specific dataset
        h5py_dir=(
            's3://cc-dataocean/scratch/20251114_olmo_example/'
            'h5py_data_w_missing_timesteps_zstd_3_128_x_4/'
            'cdl_gse_landsat_openstreetmap_raster_'
            'sentinel1_sentinel2_l2a_srtm_worldcereal_worldcover_worldpop_wri_canopy_height_map/'
            '1138828/'
        ),  # Override with: dataset.h5py_dir=...
        
        # Training modalities (use common.training_modalities)
        training_modalities=common.training_modalities,  # ['sentinel2_l2a', 'sentinel1', 'landsat']
        
        # Data configuration
        dtype="float32",  # Override with: dataset.dtype=...
        normalize=True,  # Override with: dataset.normalize=...
        cache_dir=None,  # Override with: dataset.cache_dir=...
        dataset_percentage=1.0,  # Override with: dataset.dataset_percentage=...
        seed=0,  # Override with: dataset.seed=...
    )


def build_trainer_config(common: CommonComponents) -> TrainerConfig:
    """Build trainer config for reflectance reconstruction."""
    checkpointer_config = CheckpointerConfig(work_dir=common.save_folder)
    
    trainer_config = (
        TrainerConfig(
            work_dir=common.save_folder,
            save_folder=common.save_folder,
            max_duration=Duration(10000, "steps"),  # 10K steps. Override with: trainer.max_duration=...
            metrics_collect_interval=50,  # Override with: trainer.metrics_collect_interval=...
            cancel_check_interval=25,   # Override with: trainer.cancel_check_interval=...
            load_strategy=LoadStrategy.if_available,
            checkpointer=checkpointer_config,
        )
        # .with_callback("speed_monitor", OlmoEarthSpeedMonitorCallback())  # Disabled due to missing _encoder_ratio attribute
        .with_callback("gpu_memory_monitor", GPUMemoryMonitorCallback())
        .with_callback("config_saver", ConfigSaverCallback())
        .with_callback(
            "checkpointer",
            CheckpointerCallback(
                save_interval=1000,  # Save every 1000 steps. Override with: trainer.callbacks.checkpointer.save_interval=...
                ephemeral_save_interval=250,  # Temporary saves every 250 steps
            ),
        )
    )
    return trainer_config


def build_visualize_config(common: CommonComponents) -> OlmoEarthVisualizeConfig:
    """Build visualization config for reflectance reconstruction."""
    return OlmoEarthVisualizeConfig(
        output_dir=str(f"{common.save_folder}/visualizations") if common.save_folder else "./visualizations",
        num_samples=16,  # Override with: visualize.num_samples=...
        std_multiplier=2.0,  # Override with: visualize.std_multiplier=...
    )


if __name__ == "__main__":
    main(
        common_components_builder=build_common_components,
        model_config_builder=build_model_config,
        train_module_config_builder=build_train_module_config,
        dataset_config_builder=build_dataset_config,
        dataloader_config_builder=build_dataloader_config,
        trainer_config_builder=build_trainer_config,
        visualize_config_builder=build_visualize_config,
    )