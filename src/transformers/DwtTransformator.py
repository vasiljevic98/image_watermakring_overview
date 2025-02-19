import pywt
from transformers import Transformator


class DwtTransformator(Transformator):

    def forwardTransform(self,image):
        coefficients = pywt.wavedec2(image, wavelet='haar', level=1)
        return coefficients
    
    def backwardTransform(self,coefficients):
        return pywt.waverec2(coefficients, wavelet='haar')