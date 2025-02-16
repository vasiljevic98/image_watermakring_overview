from embeders.Embeder import Watermarker
from utils import ensure_block_size, idct2, dct2
import numpy as np

class FrequencyFlippingWatermarker(Watermarker):
    def __init__(self, c1,c2, block_size):
        self.c1 = c1
        self.c2 = c2
        self.block_size = block_size
        

    def encode(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:

        [hight, width] = image.shape
        ensure_block_size(image, self.block_size)
        
        blocksV = hight // self.block_size
        blocksH = width // self.block_size


        if blocksV < watermark.shape[0] or blocksH < watermark.shape[1]:
            raise ValueError('Watermark image is too big')
        
        bits_extended = np.pad(watermark, 
                                ((0, blocksV - watermark.shape[0]), 
                                (0, blocksH - watermark.shape[1])))

        result_img = np.zeros((hight, width), dtype=np.uint8)
        
        for row in range(blocksV):
            for col in range(blocksH):
                bit = bits_extended[row, col]

                roi = image[row*self.block_size:(row+1)*self.block_size, 
                            col*self.block_size:(col+1)*self.block_size]
                dct_roi = dct2(roi)
                
                if bit == 1 and dct_roi[self.c1] < dct_roi[self.c2]:
                    dct_roi[self.c1], dct_roi[self.c2] = dct_roi[self.c2], dct_roi[self.c1]
                elif bit == 0 and dct_roi[self.c1] >= dct_roi[self.c2]:
                    dct_roi[self.c1], dct_roi[self.c2] = dct_roi[self.c2], dct_roi[self.c1]
                
                result_img[row*self.block_size:(row+1)*self.block_size, 
                        col*self.block_size:(col+1)*self.block_size] = idct2(dct_roi)

        return result_img
        

    def decode(self, image: np.ndarray) -> np.ndarray:
        [height, width] = image.shape
        ensure_block_size(image, self.block_size)
        
        blocksV = height // self.block_size
        blocksH = width // self.block_size

        bits = np.zeros((blocksV, blocksH), dtype=np.int32)

        for row in range(blocksV):
            for col in range(blocksH):

                roi = image[row*self.block_size:(row+1)*self.block_size, 
                            col*self.block_size:(col+1)*self.block_size]
                dct_roi = dct2(roi)
                bits[row,col] = 1 if dct_roi[self.c1] >= dct_roi[self.c2] else 0

   
        return bits