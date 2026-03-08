
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import HeUniform

import torch
import torch.nn as nn


# ================= KERAS HYBRID ======================

# Positional Embedding
class AddPositionEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.pos_emb = self.add_weight(
            name="pos_embedding",
            shape=(1, num_patches, embed_dim),
            initializer="zeros",
            trainable=True
        )

    def call(self, x):
        return x + self.pos_emb


# Transformer Block (Pre-LN)
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads=4, mlp_dim=384, dropout=0.1):
        super().__init__()

        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout
        )
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)

        self.mlp = tf.keras.Sequential([
            layers.Dense(
                mlp_dim,
                activation="gelu",
                kernel_initializer=HeUniform()
            ),
            layers.Dropout(dropout),
            layers.Dense(
                embed_dim,
                kernel_initializer=HeUniform()
            ),
            layers.Dropout(dropout),
        ])

    def call(self, x):
        attn_out = self.attn(self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


# CNN Block (IDENTICAL TO BASELINE)
def conv_block(x, filters):
    x = layers.Conv2D(
        filters,
        (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer=HeUniform()
    )(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    return x


# Build Hybrid
def build_keras_hybrid_model(input_shape=(64, 64, 3)):

    inputs = layers.Input(shape=input_shape)

    # CNN backbone → 8x8
    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)

    H, W, C = 8, 8, 128

    x = layers.Reshape((H * W, C))(x)
    x = AddPositionEmbedding(H * W, C)(x)

    for _ in range(2):
        x = TransformerBlock(
            embed_dim=C,
            num_heads=4,
            mlp_dim=384,
            dropout=0.1
        )(x)

    x = layers.Reshape((H, W, C))(x)

    x = conv_block(x, 256)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        256,
        use_bias=False,
        kernel_initializer=HeUniform()
    )(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs, name="keras_cnn_vit_hybrid")


# ================= PYTORCH HYBRID =====================

class TorchHybridModel(nn.Module):
    def __init__(self):
        super().__init__()

        # ----- SAME conv_block as baseline -----
        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        # CNN backbone: 8x8
        self.stage1 = conv_block(3, 32)
        self.stage2 = conv_block(32, 64)
        self.stage3 = conv_block(64, 128)

        # Transformer
        self.embed_dim = 128
        self.num_tokens = 8 * 8

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_tokens, self.embed_dim)
        )

        self.transformer_blocks = nn.ModuleList([
            self._build_transformer_block(),
            self._build_transformer_block()
        ])

        # Continue CNN
        self.stage4 = conv_block(128, 256)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Same classifier as baseline
        self.classifier = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        self.apply(self._init_weights)

    # Transformer Block (Pre-LN)
    def _build_transformer_block(self):
        return nn.ModuleDict({
            "norm1": nn.LayerNorm(self.embed_dim, eps=1e-6),
            "attn": nn.MultiheadAttention(
                embed_dim=self.embed_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            ),
            "norm2": nn.LayerNorm(self.embed_dim, eps=1e-6),
            "mlp": nn.Sequential(
                nn.Linear(self.embed_dim, 384),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(384, self.embed_dim),
                nn.Dropout(0.1),
            )
        })

    # Same initialization rule as baseline
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):

        # CNN backbone
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)  # (B,128,8,8)

        B, C, H, W = x.shape

        # Flatten to tokens
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x = x + self.pos_embedding

        # Transformer blocks
        for block in self.transformer_blocks:
            attn_input = block["norm1"](x)
            attn_out, _ = block["attn"](attn_input, attn_input, attn_input)
            x = x + attn_out
            x = x + block["mlp"](block["norm2"](x))

        # Restore spatial
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # Continue CNN
        x = self.stage4(x)

        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        return self.classifier(x)  # RAW LOGITS
