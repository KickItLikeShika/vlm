from typing import Optional, Tuple
import torch
import torch.nn as nn


class SiglipVisionConfig:
    def __init__(
        self,
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        image_size=224,
        patch_size=16,
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,
        **kwargs
    ):
        super().__init_()
        self.hidden_size = hidden_size  # transformer size
        self.intermediate_size = intermediate_size  # mlp size
        self.num_hidden_layers = num_hidden_layers  # num of layers
        self.num_attention_heads = num_attention_heads  # num of heads
        self.num_channels = num_channels  # num of channels
        self.image_size = image_size  # image size
        self.patch_size = patch_size  # patch size
        self.layer_norm_eps = layer_norm_eps  # layer norm epsilon
        self.attention_dropout = attention_dropout  # attention dropout
        self.num_image_tokens = num_image_tokens  # num of image tokens


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embeddings = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding='valid'  # no padding is added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embeddings = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.tensor:
        _, _, height, width = pixel_values.shape  # [Batch_size, num_channels, height, width]
        # convolve the 'patch_size' kernel over the image, with no overlapping patches since the stride is equal to the kernel size 
        # the output o the convoluation will have shape [Batch_size, embed_dim, num_patches_h, num_patches_w]
        # where num_patches_h = height // patch_size and num_patches_w = width // patch_size
        
        # [Batch_size, embed_dim, num_patches_h, num_patches_w] -> [Batch_size, embed_dim, num_patches]
        # where num_patches = num_patches_h * num_patches_w
        patch_embeds = self.patch_embeddings(pixel_values)
        
        # [batch_size, num_patches_h, num_patches_w] -> [batch_size, num_patches, embed_dim]     
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        
        # add position embeddings to each patch. each positional encoding is a vector of size 'embed_dim'
        embeddings = embeddings + self.positional_embeddings(self.position_ids)
        
        # [batch_size, num_patches, embed_dim]
        return embeddings


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5  # equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [batch_size, num_patches, embed_dim]
        batch_size, seq_len, embed_dim = hidden_states.size()

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # [batch_size, num_heads, num_patches, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).tanspose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [batch_size, num_heads, num_patches, head_dim] @ [batch_size, num_heads, head_dim, num_patches] -> [batch_size, num_heads, num_patches, num_patches]
        attn_weights = (q @ k.transpose(2, 3)) * self.scale
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but got {attn_weights.size()}")

        # apply softmax. attn_weights: [batch_size, num_heads, num_patches, num_patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

        # dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        
        attn_output = attn_weights @ v  # [batch_size, num_heads, num_patches, head_dim]
        
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(f"Attention output should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but got {attn_output.size()}")

        # [batch_size, num_heads, num_patches, head_dim] -> [batch_size, num_patches, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # [batch_size, num_patches, num_heads, head_dim] -> [batch_size, num_patches, embed_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)

        # [batch_size, num_patches, embed_dim]
        attn_output = self.out_proj(attn_output)
        
        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.fc1 = nn.Linear(self.embed_dim, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, self.embed_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, intermediate_size]
        hidden_states = self.fc1(hidden_states)

        hidden_states = nn.functional.gelu(hidden_states, approximate='tanh')

        # [batch_size, num_patches, intermediate_size] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipVisionEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # residual: [batch_size, num_patches, embed_dim]
        residual = hidden_states

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm1(hidden_states)

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states, _ = self.self_attn(hidden_states)

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = residual + hidden_states

        # residual: [batch_size, num_patches, embed_dim]
        residual = hidden_states

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.layer_norm2(hidden_states)

        # [batch_size, num_patches, embed_dim] -> [batch_size, num_patches, embed_dim]
        hidden_states = self.mlp(hidden_states)

        # [batch_size, num_patches, embed_dim]
        hidden_states = residual + hidden_states

        return hidden_states


class SiglipVisionEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [SiglipVisionEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim]
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipVisionEncoder(config)
        self.post_layer_norm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [Batch_size, num_channels, hieght, width] -> [Batch_size, num_patches, embed_dim]
        hidden_states = self.embeddings(pixel_values)
        last_hidden_states = self.encoder(hidden_states)
        return self.post_layer_norm(last_hidden_states)


class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionTransformer(config)
    
    def forward(self, pixel_values):
        # [Batch_size, num_channels, image_size, image_size] -> [Batch_size, num_patches, embed_dim]
        return self.vision_model(pixel_values)
