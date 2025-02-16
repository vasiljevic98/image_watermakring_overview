from embeders.Embeder import Watermarker
import numpy as np
from utils import idct2, dct2
import math
import matplotlib.pyplot as plt

class RandomDCTWatermarker(Watermarker):
    def __init__(self, b_cut=64, seed=1, block_size=8, fact=8, permute=True):
        self.b_cut = b_cut
        self.seed = seed
        self.block_size = block_size
        self.fact =  fact
        self.permute = permute

    def encode(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
            cut_image = image[self.b_cut: -self.b_cut, self.b_cut: -self.b_cut]
            cut_shape = cut_image.shape

            [cut_hight, cut_width] = cut_shape
            [org_hight, org_width] = image.shape
            [watermark_hight, watermark_width] = watermark.shape

            blocks_v = cut_hight // self.block_size
            blocks_h = cut_width // self.block_size

            if blocks_v * blocks_h < watermark.size:
                raise ValueError(f'Not enough space in the image. Image can store {blocks_v * blocks_h} bits, but {watermark.size} bits were provided')
            
            # blocks available for watermarking
            blocks_available = blocks_v * blocks_h

            imf = np.float32(image)

            for i in range(0, org_hight, self.block_size):
                for j in range(0, org_width, self.block_size):
                    dst = dct2(imf[i:i+self.block_size, j:j+self.block_size]/1.0)
                    imf[i:i+self.block_size, j:j+self.block_size] = idct2(dst)

            final = np.copy(image)
            
            print('available blocks:', blocks_v * blocks_h)
            np.random.seed(self.seed)

            blocks_needed = watermark_hight * watermark_width
            block_order = np.random.permutation(blocks_available) if self.permute else np.arange(blocks_available)
            print('order start:', block_order[0:10])

            for i in range(blocks_needed):
                watermark_pixel = watermark[i // watermark_width, i % watermark_width]

                x = block_order[i]

                # coordinates of the block top left corner
                roi_y = (x//blocks_h) * self.block_size + self.b_cut
                roi_x = (x % blocks_h) * self.block_size + self.b_cut

                dct_block = dct2(imf[roi_y:roi_y+self.block_size, roi_x:roi_x+self.block_size])
                final[roi_y:roi_y+self.block_size, roi_x:roi_x+self.block_size] = idct2(dct_block)

                # # ovo efektivno zaokruzuje na blizi ceo broj i skalira faktorom
                elem =  math.floor(dct_block[0, 0]/self.fact+0.5)

                final_elem = elem
                if (watermark_pixel % 2) != (math.ceil(elem) % 2):
                    final_elem = math.ceil(elem) - 1

               
                dct_block[0, 0] = final_elem * self.fact  
                final[roi_y:roi_y+self.block_size, roi_x:roi_x+self.block_size] = idct2(dct_block)

                if i< 50:
                   dct_block2 = dct2(final[roi_y:roi_y+self.block_size, roi_x:roi_x+self.block_size])
                   print(final_elem, dct_block2[0, 0]/self.fact+0.5)
            
            return final
            

    def decode(self, image: np.ndarray, watermark_shape: tuple) -> np.ndarray:

        cut_image = image[self.b_cut: -self.b_cut, self.b_cut: -self.b_cut]
        cut_shape = cut_image.shape

        [cut_hight, cut_width] = cut_shape
        [org_hight, org_width] = image.shape

        blocks_v = cut_hight // self.block_size
        blocks_h = cut_width // self.block_size

        imf = np.float32(image)

        # for i in range(0, org_hight, self.block_size):
        #     for j in range(0, org_width, self.block_size):
        #         dst = dct2(imf[i:i+self.block_size, j:j+self.block_size] / 1.0)
        #         imf[i:i+self.block_size, j:j+self.block_size] = idct2(dst)

        print('available blocks:', blocks_v * blocks_h)
        np.random.seed(self.seed)
        block_order = np.random.permutation(blocks_v * blocks_h) if self.permute else np.arange(blocks_v * blocks_h)
        print('order start:', block_order[0:10])

        watermark = np.zeros(watermark_shape, dtype=np.uint8)
        print('watermark shape:', watermark.size)
        for i in range(watermark.size):
            x = block_order[i]

            # coordinates of the block top left corner
            roi_y = (x//blocks_h) * self.block_size + self.b_cut
            roi_x = (x % blocks_h) * self.block_size + self.b_cut

            dct_block = dct2(imf[roi_y:roi_y+self.block_size, roi_x:roi_x+self.block_size])

            elem = math.floor(dct_block[0, 0]/self.fact+0.5)


            watermark[i // watermark_shape[1], i % watermark_shape[1]] = elem % 2

        return watermark


             


                


                 



