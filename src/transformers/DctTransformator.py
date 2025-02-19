import numpy as np
from transformers import Transformator
from utils import dct2, idct2


class DctTransformator(Transformator):
    def __init__(self,params):
        self.block_size = params['block_size'] if 'block_size' in params else 8


    def forwardTransform(self,image):
        [hight, width] = image.shape
        result_img = np.zeros((hight, width), dtype=np.uint8)

        for i in range(0, hight, self.block_size):
            for j in range(0, width, self.block_size):
                dst = dct2(image[i:i+self.block_size, j:j+self.block_size]/1.0)
                result_img[i:i+self.block_size, j:j+self.block_size] = dst

        return result_img
    
    def backwardTransform(self,image):
        [hight, width] = image.shape
        result_img = np.zeros((hight, width), dtype=np.uint8)

        for i in range(0, hight, self.block_size):
            for j in range(0, width, self.block_size):
                dst = idct2(image[i:i+self.block_size, j:j+self.block_size]/1.0)
                result_img[i:i+self.block_size, j:j+self.block_size] = dst

        return result_img