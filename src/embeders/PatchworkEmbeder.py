from embeders.Embeder import Embeder
import numpy as np
from dataclasses import dataclass
import cv2
import utils

class PatchworkEmbeder(Embeder):

    def __init__(self,
    bit_square_size: int = 8,
    noise_type: str = 'gaussian',
    seed: int = 42
    ):
        np.random.seed(seed)
        self.zero_noise = (utils.get_noise_image((bit_square_size,bit_square_size), noise_type) * 255).astype(np.uint8)
        self.one_noise = (utils.get_noise_image((bit_square_size,bit_square_size), noise_type) * 255).astype(np.uint8)
         
    def encode(self, image, watermark, noise_level) -> np.ndarray:
        bit_square_size = self.zero_noise.shape[0]
    
        if image.shape[0] % bit_square_size != 0 or image.shape[1] % bit_square_size != 0:
            raise ValueError('Image dimensions must be a multiple of the bit square size')
            
        watermark_shape = [image.shape[0]//bit_square_size, image.shape[1]//bit_square_size]

        if watermark.shape[0] > watermark_shape[0] or watermark.shape[1] > watermark_shape[1]:
            raise ValueError('Watermark image is too big')
        
        watermark_to_embed = np.pad(watermark, 
                                    ((0, watermark_shape[0] - watermark.shape[0]), 
                                     (0, watermark_shape[1] - watermark.shape[1])))

        noise_map = utils.map_bits_to_image(image.shape, watermark_to_embed, self.zero_noise, self.one_noise)
        encoded_image = cv2.addWeighted(image, 1-noise_level, noise_map, noise_level, 0)
        return encoded_image

    def decode(self, image) -> np.ndarray:
        
        bit_square_size = self.zero_noise.shape[0] 
        vertical_tiles = image.shape[0] // bit_square_size
        horizontal_tiles = image.shape[1] // bit_square_size

        decoded_bits = np.zeros((vertical_tiles, horizontal_tiles))
        for i in range(vertical_tiles):
            for j in range(horizontal_tiles):
                roi = image[i*bit_square_size:(i+1)*bit_square_size, j*bit_square_size:(j+1)*bit_square_size]
                corr0 = utils.correlate_images_cv2_matchTemplate(roi, self.zero_noise)
                corr1 = utils.correlate_images_cv2_matchTemplate(roi, self.one_noise)
                decoded_bits[i, j] = 1 if corr1 > corr0 else 0

        return np.array(decoded_bits.reshape((vertical_tiles, horizontal_tiles)))
    
    