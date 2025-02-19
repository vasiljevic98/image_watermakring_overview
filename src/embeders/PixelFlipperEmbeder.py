from embeders.Embeder import Embeder
import numpy as np
from dataclasses import dataclass

import transformations
import utils

class PixelFlipperEmbeder(Embeder):
    def __init__(self, params: dict):
        self.coord1 = params['coord1'] if 'coord1' in params else (3,0)
        self.coord2 = params['coord2'] if 'coord2' in params else (1,2)
        self.block_size = params['block_size'] if 'block_size' in params else 1
        

    def encode(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        resized_watermark = utils.fit_to_size(watermark, (image.shape[0] // self.block_size, image.shape[1] // self.block_size))
        result_img = np.copy(image)

        for i in range(resized_watermark.shape[0]):
            for j in range(resized_watermark.shape[1]):
                x1, y1 = i * self.block_size + self.coord1[0], j * self.block_size + self.coord1[1]
                x2, y2 = i * self.block_size + self.coord2[0], j * self.block_size + self.coord2[1]
                pix1, pix2 = image[x1, y1], image[x2, y2]

                if (resized_watermark[i, j] == 1 and pix1 < pix2) or (resized_watermark[i, j] == 0 and pix1 >= pix2):
                    result_img[x1, y1], result_img[x2, y2] = pix2, pix1

        return result_img
    
    def decode(self, image: np.ndarray) -> np.ndarray:
        result_img = np.zeros((image.shape[0] // self.block_size, image.shape[1] // self.block_size))
        for i in range(result_img.shape[0]):
            for j in range(result_img.shape[1]):
                x1, y1 = i * self.block_size + self.coord1[0], j * self.block_size + self.coord1[1]
                x2, y2 = i * self.block_size + self.coord2[0], j * self.block_size + self.coord2[1]
                pix1, pix2 = image[x1, y1], image[x2, y2]
                result_img[i, j] = 1 if pix1 < pix2 else 0
        return result_img