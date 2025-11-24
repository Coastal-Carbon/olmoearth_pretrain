"""
Separate Reflectance Reconstruction Head for OlmoEarth

This creates a new model that uses a pretrained OlmoEarth encoder with a 
dedicated reconstruction head for actual reflectance values.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Any, Dict, List, Optional
from copy import deepcopy
from dataclasses import dataclass, field

try:
    from olmo_core.config import Config
except ImportError:
    # Fallback for when olmo_core is not available
    Config = object

from olmoearth_pretrain.nn.flexi_vit import TokensAndMasks, Encoder
from olmoearth_pretrain.nn.flexi_patch_embed import FlexiPatchReconstruction
from olmoearth_pretrain.data.constants import Modality, ModalitySpec
from olmoearth_pretrain.train.masking import MaskedOlmoEarthSample


class ReflectanceReconstructionHead(nn.Module):
    """Dedicated reconstruction head for actual reflectance values."""
    
    def __init__(
        self,
        embedding_size: int,
        supported_modalities: list[ModalitySpec],
        max_patch_size: int,
        hidden_size: Optional[int] = None,
        use_sigmoid_output: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size or embedding_size
        self.max_patch_size = max_patch_size
        self.supported_modalities = supported_modalities
        self.use_sigmoid_output = use_sigmoid_output
        
        # Create reconstruction modules for each modality
        self.modality_heads = nn.ModuleDict({})
        self.dropout = dropout
        for modality in supported_modalities:
            self.modality_heads[modality.name] = self._create_modality_head(modality, dropout)
    
    def _create_modality_head(self, modality: ModalitySpec, dropout: float = 0.1) -> nn.Module:
        """Create reconstruction head for a specific modality."""
        
        if modality.get_tile_resolution() == 0:
            # Non-spatial modality (e.g., weather data)
            total_channels = sum(len(bandset) for bandset in modality.bandsets_as_indices())
            return nn.Sequential(
                nn.LayerNorm(self.embedding_size),
                nn.Linear(self.embedding_size, self.hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_size, total_channels),
                nn.Sigmoid() if self.use_sigmoid_output else nn.Identity()
            )
        else:
            # Spatial modality (satellite imagery)
            total_channels = sum(len(bandset) for bandset in modality.bandsets_as_indices())
            return SpatialReconstructionModule(
                embedding_size=self.embedding_size,
                hidden_size=self.hidden_size,
                out_channels=total_channels,
                max_patch_size=self.max_patch_size,
                use_sigmoid_output=self.use_sigmoid_output,
                dropout=dropout
            )
    
    def forward(
        self, 
        encoded_tokens: TokensAndMasks, 
        patch_size: int,
        target_modalities: Optional[list[str]] = None
    ) -> dict[str, Tensor]:
        """Apply reconstruction to specified modalities."""
        
        reconstructions = {}
        modalities_to_process = target_modalities or [m.name for m in self.supported_modalities]
        
        for modality_name in modalities_to_process:
            if modality_name in self.modality_heads and hasattr(encoded_tokens, modality_name):
                modality_tokens = getattr(encoded_tokens, modality_name)
                head = self.modality_heads[modality_name]
                
                if isinstance(head, SpatialReconstructionModule):
                    reconstruction = head(modality_tokens, patch_size)
                else:
                    reconstruction = head(modality_tokens)
                
                reconstructions[modality_name] = reconstruction
        
        return reconstructions


class SpatialReconstructionModule(nn.Module):
    """Spatial reconstruction module for satellite imagery."""
    
    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        out_channels: int,
        max_patch_size: int,
        use_sigmoid_output: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.out_channels = out_channels
        self.max_patch_size = max_patch_size
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        
        # Spatial reconstruction using transposed convolution with smaller stride
        # Use stride=1 instead of max_patch_size to prevent massive upsampling
        self.spatial_reconstruction = nn.ConvTranspose2d(
            hidden_size,
            out_channels,
            kernel_size=3,  # Use smaller kernel
            stride=1,       # Use stride=1 instead of max_patch_size=8
            padding=1,      # Add padding to maintain dimensions
            bias=True
        )
        
        # Optional output normalization
        self.output_activation = nn.Sigmoid() if use_sigmoid_output else nn.Identity()
        
    def forward(self, x: Tensor, patch_size: int) -> Tensor:
        """
        Args:
            x: Input tokens [B, N, D] where N is the number of tokens
            patch_size: Patch size for reconstruction
            
        Returns:
            Reconstructed spatial data [B, C, H, W]
        """
        batch_size = x.shape[0]
        num_tokens = x.shape[1]
        embed_dim = x.shape[2]
        
        # Calculate spatial dimensions from number of tokens
        # Assuming square token arrangement
        tokens_per_side = int(num_tokens ** 0.5)
        if tokens_per_side * tokens_per_side != num_tokens:
            # Fallback: use sqrt and pad/truncate if needed
            tokens_per_side = int(num_tokens ** 0.5)
        
        # The input x has shape [B, H_patches, W_patches, T, C, embed_dim]
        # We need to reshape it to spatial format for reconstruction
        original_shape = x.shape
        batch_size = original_shape[0]
        embed_dim = original_shape[-1]
        
        # Flatten to [B, N, D] format
        x_flat = x.view(batch_size, -1, embed_dim)
        
        # For spatial reconstruction, we need to organize tokens into a spatial grid
        num_tokens = x_flat.shape[1]
        
        # Calculate a reasonable spatial arrangement
        spatial_side = int(num_tokens ** 0.5)
        if spatial_side * spatial_side != num_tokens:
            # Pad to make it square if needed
            spatial_side = int(num_tokens ** 0.5) + 1
            total_needed = spatial_side * spatial_side
            pad_size = total_needed - num_tokens
            padding = torch.zeros(batch_size, pad_size, embed_dim, device=x_flat.device, dtype=x_flat.dtype)
            x_flat = torch.cat([x_flat, padding], dim=1)
            
        # Reshape to spatial format [B, H, W, D]
        x = x_flat.view(batch_size, spatial_side, spatial_side, embed_dim)
        
        # Apply feature processing
        x_processed = self.feature_processor(x)  # [B*T, H, W, hidden_size]
        
        # Prepare for convolution: [B*T, hidden_size, H, W]
        x_conv = x_processed.permute(0, 3, 1, 2)
        
        # Apply spatial reconstruction - but don't upsample massively
        x_recon = self.spatial_reconstruction(x_conv)  # This creates too large outputs
        
        # Instead of upsampling by full patch_size, keep reasonable output size
        # Limit output to maximum 512x512 to prevent memory issues
        max_output_size = 512
        if x_recon.shape[2] > max_output_size or x_recon.shape[3] > max_output_size:
            # Downsample to reasonable size using adaptive pooling
            x_recon = torch.nn.functional.adaptive_avg_pool2d(x_recon, (max_output_size, max_output_size))
            print(f"DEBUG: Downsampled reconstruction output to {x_recon.shape}")
        
        # Reshape back to spatial format: [B*T, H, W, out_channels]
        x_recon = x_recon.permute(0, 2, 3, 1)
        
        # Apply output activation
        x_recon = self.output_activation(x_recon)
        
        # For now, return as [B, H, W, C] without handling time dimension
        # TODO: Properly handle time dimension restoration if needed
        
        return x_recon


class OlmoEarthWithReflectanceHead(nn.Module):
    """OlmoEarth model with separate reflectance reconstruction head."""
    
    def __init__(
        self,
        pretrained_model: Any,  # Should be the full OlmoEarth model
        reconstruction_config: dict,
        freeze_encoder: bool = True,
    ):
        super().__init__()
        
        # Use pretrained encoder
        if freeze_encoder:
            self.encoder = pretrained_model.encoder
            # Freeze encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("🔒 Frozen encoder parameters")
        else:
            self.encoder = deepcopy(pretrained_model.encoder)
            print("🔓 Encoder parameters will be fine-tuned")
        
        # Create dedicated reflectance reconstruction head
        self.reflectance_head = ReflectanceReconstructionHead(
            embedding_size=self.encoder.embedding_size,
            **reconstruction_config
        )
        
        # Store config
        self.freeze_encoder = freeze_encoder
        
    def forward(
        self, 
        x: MaskedOlmoEarthSample, 
        patch_size: int,
        target_modalities: Optional[list[str]] = None,
        return_encoder_output: bool = False
    ) -> dict[str, Any]:
        """
        Args:
            x: Masked input sample
            patch_size: Patch size for processing
            target_modalities: Which modalities to reconstruct
            return_encoder_output: Whether to return encoder features
            
        Returns:
            Dictionary with reconstructions and optionally encoder features
        """
        
        # Get encoded representations
        encoder_output = self.encoder(x, patch_size=patch_size)
        
        # Extract tokens_and_masks from encoder output
        if isinstance(encoder_output, dict) and 'tokens_and_masks' in encoder_output:
            encoded_tokens = encoder_output['tokens_and_masks']
        else:
            encoded_tokens = encoder_output
        
        # Apply reflectance reconstruction
        reconstructions = self.reflectance_head(
            encoded_tokens, 
            patch_size=patch_size,
            target_modalities=target_modalities
        )
        
        result = {
            'reconstructions': reconstructions,
        }
        
        if return_encoder_output:
            result['encoder_output'] = encoder_output
            
        return result
    
    def get_reconstruction_loss(
        self, 
        reconstructions: dict[str, Tensor], 
        targets: dict[str, Tensor],
        masks: Optional[dict[str, Tensor]] = None,
        loss_type: str = 'l1'
    ) -> dict[str, Tensor]:
        """Compute reconstruction loss for each modality."""
        
        if loss_type == 'l1':
            loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            loss_fn = nn.MSELoss(reduction='none')
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        losses = {}
        total_loss = 0
        
        for modality_name in reconstructions:
            if modality_name in targets:
                pred = reconstructions[modality_name]
                target = targets[modality_name]
                
                # Compute base loss
                loss = loss_fn(pred, target)
                
                # Apply mask if provided
                if masks and modality_name in masks:
                    mask = masks[modality_name]
                    # Expand mask to match loss dimensions if needed
                    while mask.dim() < loss.dim():
                        mask = mask.unsqueeze(-1)
                    
                    masked_loss = loss * mask
                    modality_loss = masked_loss.sum() / (mask.sum() + 1e-8)
                else:
                    modality_loss = loss.mean()
                
                losses[f'{modality_name}_loss'] = modality_loss
                total_loss += modality_loss
        
        losses['total_loss'] = total_loss
        return losses


def create_reflectance_reconstruction_model(
    pretrained_model_path: str = None,
    pretrained_model_id: str = None,
    target_modalities: list[str] = None,
    max_patch_size: int = 32,
    freeze_encoder: bool = True,
    use_sigmoid_output: bool = False,
    hidden_size_multiplier: float = 1.0,
    override_min_patch_size: int = None,
    override_max_patch_size: int = None,
) -> OlmoEarthWithReflectanceHead:
    """Create a model with reflectance reconstruction head."""
    
    # Load pretrained model
    if pretrained_model_path:
        # Load from file path
        pretrained_model = torch.load(pretrained_model_path, map_location='cpu')
    elif pretrained_model_id:
        # Load using OlmoEarth model loader
        from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
        
        # Convert string to ModelID if needed
        if isinstance(pretrained_model_id, str):
            model_id = ModelID(pretrained_model_id)
        else:
            model_id = pretrained_model_id
            
        pretrained_model = load_model_from_id(model_id, load_weights=True)
    else:
        raise ValueError("Must provide either pretrained_model_path or pretrained_model_id")
    
    # Override patch sizes if provided to fix tensor dimension issues
    if override_min_patch_size is not None:
        pretrained_model.encoder.min_patch_size = override_min_patch_size
        # Also update config if it exists
        if hasattr(pretrained_model.encoder, 'config'):
            pretrained_model.encoder.config.min_patch_size = override_min_patch_size
        # Also update kwargs if they exist
        if hasattr(pretrained_model.encoder, '_kwargs'):
            pretrained_model.encoder._kwargs['min_patch_size'] = override_min_patch_size
        print(f"Overrode encoder min_patch_size to {override_min_patch_size}")
    
    if override_max_patch_size is not None:
        pretrained_model.encoder.max_patch_size = override_max_patch_size
        # Also update config if it exists
        if hasattr(pretrained_model.encoder, 'config'):
            pretrained_model.encoder.config.max_patch_size = override_max_patch_size
        # Also update kwargs if they exist
        if hasattr(pretrained_model.encoder, '_kwargs'):
            pretrained_model.encoder._kwargs['max_patch_size'] = override_max_patch_size
        print(f"Overrode encoder max_patch_size to {override_max_patch_size}")
    
    # Get modality specifications
    supported_modalities = []
    for modality_name in target_modalities:
        try:
            modality_spec = Modality.get(modality_name)
            supported_modalities.append(modality_spec)
        except KeyError:
            print(f"Warning: Unknown modality {modality_name}, skipping")
    
    # Configure reconstruction head
    embedding_size = pretrained_model.encoder.embedding_size
    reconstruction_config = {
        'supported_modalities': supported_modalities,
        'max_patch_size': max_patch_size,
        'hidden_size': int(embedding_size * hidden_size_multiplier),
        'use_sigmoid_output': use_sigmoid_output,
        'dropout': 0.1,
    }
    
    # Create model with reconstruction head
    model = OlmoEarthWithReflectanceHead(
        pretrained_model=pretrained_model,
        reconstruction_config=reconstruction_config,
        freeze_encoder=freeze_encoder,
    )
    
    return model


# Example usage
if __name__ == "__main__":
    print("Example: Creating OlmoEarth model with reflectance reconstruction head")
    print()
    print("# 1. Create model")
    print("model = create_reflectance_reconstruction_model(")
    print("    pretrained_model_path='path/to/pretrained/model.pt',")
    print("    target_modalities=['sentinel2_l2a', 'landsat'],")
    print("    freeze_encoder=True,")
    print("    use_sigmoid_output=True  # For normalized reflectance [0,1]")
    print(")")
    print()
    print("# 2. Forward pass")
    print("results = model(masked_sample, patch_size=32, target_modalities=['sentinel2_l2a'])")
    print("reconstructions = results['reconstructions']")
    print()
    print("# 3. Compute loss")
    print("losses = model.get_reconstruction_loss(")
    print("    reconstructions=reconstructions,")
    print("    targets=original_satellite_data,")
    print("    masks=valid_pixel_masks,")
    print("    loss_type='l1'")
    print(")")
    print("total_loss = losses['total_loss']")


@dataclass
class ReflectanceReconstructionConfig(Config):
    """Configuration for OlmoEarth with reflectance reconstruction head."""
    
    # Pretrained encoder configuration (use either path or model_id)
    pretrained_encoder_path: str | None = None
    pretrained_model_id: str | None = None  # e.g., 'OLMOEARTH_V1_BASE'
    
    # Reconstruction head configuration  
    target_modalities: list[str] = field(default_factory=lambda: ["sentinel2_l2a", "landsat"])
    freeze_encoder: bool = True
    use_sigmoid_output: bool = False
    hidden_size_multiplier: float = 1.0
    
    # Patch size configuration to fix tensor dimension issues
    override_min_patch_size: int | None = None
    override_max_patch_size: int | None = None
    
    def build(self) -> "OlmoEarthWithReflectanceHead":
        """Build the model with reconstruction head."""
        return create_reflectance_reconstruction_model(
            pretrained_model_path=self.pretrained_encoder_path,
            pretrained_model_id=self.pretrained_model_id,
            target_modalities=self.target_modalities,
            freeze_encoder=self.freeze_encoder,
            use_sigmoid_output=self.use_sigmoid_output,
            hidden_size_multiplier=self.hidden_size_multiplier,
            override_min_patch_size=self.override_min_patch_size,
            override_max_patch_size=self.override_max_patch_size,
        )