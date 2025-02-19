import numpy as np
from transformers import Transformator
from utils import recoverDiagonalMatrix

class SvdTransformator:
    def forwardTransform(self,image):
        Uc, Sc, Vc = np.linalg.svd(image)
        Sc = recoverDiagonalMatrix(Sc, image.shape)
        return Uc, Sc, Vc
    
    def backwardTransform(self,Uc, Sc, Vc):
        return Uc.dot(Sc).dot(Vc)