
# KERAS CNN BASELINE
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import HeUniform


def conv_block_keras(x, filters):
    x = layers.Conv2D(
        filters, (3, 3),
        padding="same",
        use_bias=False,
        kernel_initializer=HeUniform()
    )(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    return x


def build_keras_cnn(input_shape=(64, 64, 3)):
    inputs = layers.Input(shape=input_shape)

    x = conv_block_keras(inputs, 32)
    x = conv_block_keras(x, 64)
    x = conv_block_keras(x, 128)
    x = conv_block_keras(x, 256)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, use_bias=False, kernel_initializer=HeUniform())(x)
    x = layers.BatchNormalization(momentum=0.99)(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(1, activation="sigmoid")(x)

    return models.Model(inputs, outputs)


# PYTORCH CNN BASELINE
import torch
import torch.nn as nn


class CNNBaseline(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(256, 256, bias=False),
            nn.BatchNorm1d(256, eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)
