from embeders.Embeder import Watermarker
import numpy as np
from utils import idct2, dct2, recoverDiagonalMatrix
import math
import matplotlib.pyplot as plt
import pywt


class DctSvdWatermarker(Watermarker):

    

    def encode(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:

            [watermark_hight, watermark_width] = watermark.shape

            coefficients = pywt.wavedec2(image, wavelet='haar', level=1)
            shape_LL = coefficients[0].shape 
            
            Uc, Sc, Vc = np.linalg.svd(coefficients[0])
            Sc = recoverDiagonalMatrix(Sc, shape_LL)

            LLMinDim = min(shape_LL)

            if(watermark_hight > shape_LL[0] or watermark_width > shape_LL[1]):
                raise ValueError(f'Not enough space in the image. Image can store {shape_LL[0] * shape_LL[1]} bits, but {watermark_hight * watermark_width} bits were provided')
            
            padded_watermark = np.zeros(shape_LL)
            padded_watermark[:watermark_hight, :watermark_width] = watermark

            alpha=0.1
            Snew=np.zeros((LLMinDim,LLMinDim))
            
            for py in range(0,LLMinDim):
                for px in range(0,LLMinDim):
                    Snew[py][px]=Sc[py][px]+alpha*(padded_watermark[py][px])


            Uw, Sw, Vw = np.linalg.svd(Snew)
            LLnew=np.zeros((LLMinDim,LLMinDim))
            LLnew=Uc.dot(np.diag(Sw)).dot(Vc)
            LLnew=Uc.dot(Snew).dot(Vc)

            coefficients[0]=LLnew
            final = pywt.waverec2(coefficients, wavelet='haar')

            return final
            

    def decode(self, image: np.ndarray, original_image: np.ndarray) -> np.ndarray:
        
        coefficients = pywt.wavedec2(image, wavelet='haar', level=1)
        shape_LL = coefficients[0].shape 
        
        Uc, Sc, Vc = np.linalg.svd(coefficients[0])
        Sc = recoverDiagonalMatrix(Sc, shape_LL)

        LLMinDim = min(shape_LL)

        alpha=0.1
        Snew=np.zeros((LLMinDim,LLMinDim))
        
        for py in range(0,LLMinDim):
            for px in range(0,LLMinDim):
                Snew[py][px]=Sc[py][px]

        Uw, Sw, Vw = np.linalg.svd(Snew)
        LLnew=np.zeros((LLMinDim,LLMinDim))
        LLnew=Uc.dot(np.diag(Sw)).dot(Vc)
        LLnew=Uc.dot(Snew).dot(Vc)

        coefficients[0]=LLnew
        final = pywt.waverec2(coefficients, wavelet='haar')

        return final
             


                


                 



