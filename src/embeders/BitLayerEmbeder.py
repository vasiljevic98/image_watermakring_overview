from embeders.Embeder import Embeder
import numpy as np
from dataclasses import dataclass

import transformations
import utils

class BitLayerEmbeder(Embeder):
    def __init__(self):
        pass

    def encode(self, image: np.ndarray, watermark: np.ndarray, params: dict) -> np.ndarray:
        bit = params['layer']

        watermark_resized = np.pad(watermark, 
                                    ((0, image.shape[0] - watermark.shape[0]), 
                                     (0, image.shape[1] - watermark.shape[1])))

        print(watermark_resized.shape)
        dropped_img = transformations.drop_bit_layer(image,bit)
        embedded_img = dropped_img | (watermark_resized << bit)

        return embedded_img
        

    def decode(self, image: np.ndarray, bit: int) -> np.ndarray:
        return utils.get_bit_layer(image, bit)
        