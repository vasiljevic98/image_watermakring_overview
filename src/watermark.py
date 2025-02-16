from dataclasses import dataclass
import numpy as np
from enum import Enum

class WatermarkType(Enum):
    image = 1
    text = 2

@dataclass
class Watermark:
    watermarkType: WatermarkType
    data: np.ndarray | str

