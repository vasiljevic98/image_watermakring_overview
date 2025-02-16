
from math import log10, sqrt 
import cv2 
import numpy as np 
import png 
import qrcode
import uuid
from scipy import signal
import nltk
from scipy.fftpack import dct, idct


def PSNR(original, transformed): 
    mse = np.mean((original - transformed) ** 2) 
    if(mse == 0): 
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 
  

def MSE(original, transformed): 
    mse = np.mean((original - transformed) ** 2) 
    return mse

def generate_qr_code(data, correction='M', box_size=10):

    ecs = {
        'L': qrcode.constants.ERROR_CORRECT_L,
        'M': qrcode.constants.ERROR_CORRECT_M,
        'Q': qrcode.constants.ERROR_CORRECT_Q,
        'H': qrcode.constants.ERROR_CORRECT_H,
    }
    error_correction = ecs.get(correction, qrcode.constants.ERROR_CORRECT_L)

    qr = qrcode.QRCode(
    error_correction=error_correction,
    box_size=box_size
    )
    qr.add_data(data)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    return np.array(img)

def generate_tiled_image(img, targetSize):
    org_height, org_width = img.shape[:2]
    vertical_tiles = targetSize[0]//org_height + 1
    horizontal_tiles = targetSize[1]//org_width + 1
    tiled_img = np.tile(img, (vertical_tiles, horizontal_tiles))
    return tiled_img[:targetSize[0], :targetSize[1]]

def get_lorem_text_bytes(numberOfChars):
    with open('../assets/lorem.txt', 'rb') as file:
        text = file.read()
        if len(text) < numberOfChars:
            raise ValueError('Not enough text in file')
        return text[:numberOfChars]
    

def get_bit_layer(img, bit):
    return (img >> bit) & 1
        
def get_guid():
    return uuid.uuid4()

def correlate_images(img1, img2):
    return np.correlate(img1.flatten(), img2.flatten())

def get_noise_image(shape, noise_type):
    if noise_type == 'gaussian':
        return np.random.normal(0, 1, shape)
    elif noise_type == 'uniform':
        return np.random.uniform(0, 1, shape)
    elif noise_type == 'salt_and_pepper':
        return np.random.choice([0, 1], size=shape, p=[0.5, 0.5])
    else:
        raise ValueError('Invalid noise type')
    
def bytes_to_array_of_bits(bytes):
    bits_string = ''.join(format(i, '08b') for i in bytes)
    return np.array([int(b) for b in bits_string])

def array_of_bits_to_bytes(bits):
    bits_string = ''.join(str(b) for b in bits)
    return bytes(int(bits_string[i:i+8], 2) for i in range(0, len(bits_string), 8))
    

def map_bits_to_image(shape, watermark, zero_img, one_img):

    if(zero_img.shape != one_img.shape):
        raise ValueError('Images must have the same shape')
    if(shape[0] % zero_img.shape[0] != 0 or shape[1] % zero_img.shape[1] != 0):
        raise ValueError('Target shape must be a multiple of the image shape')
    

    vertical_tiles = shape[0]//zero_img.shape[0]
    horizontal_tiles = shape[1]//zero_img.shape[1]

    if vertical_tiles < watermark.shape[0] or horizontal_tiles < watermark.shape[1]:
        raise ValueError('Not enough space in the image')
    
    result = np.zeros(shape, dtype=np.uint8)

    for i in range(watermark.shape[0]):
        for j in range(watermark.shape[1]):
            x = j * zero_img.shape[1]
            y = i * zero_img.shape[0]
            result[y:y+zero_img.shape[0], x:x+zero_img.shape[1]] = zero_img if watermark[i, j] == 0 else one_img

    return result

def correlate_images_np_scaled(img1, img2):
    return np.correlate(img1.flatten()/255, img2.flatten()/255)

def correlate_images_np(img1, img2):
    return np.correlate(img1.flatten(), img2.flatten())

def correlate_images_scipy(img1, img2):
    corr_matrix = signal.correlate2d(img1, img2, mode='same')
    return np.mean(corr_matrix)

def correlate_images_scipy_scaled(img1, img2):
    corr_matrix = signal.correlate2d(img1/255, img2/255, mode='same')
    return np.mean(corr_matrix)


def correlate_images_cv2_matchTemplate(img1, img2):
    corr_matrix = cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)
    return np.mean(corr_matrix)

def correlate_images_abs_diff(img1, img2):
    return np.average(np.abs(img1 - img2))
    

def edit_distance(str1, str2):
    return nltk.edit_distance(str1, str2)

def ensure_block_size(img, block_size):
    [hight, width] = img.shape
    if hight % block_size != 0 or width % block_size != 0:
        raise ValueError('Image size must be a multiple of block size')
    return img

def dct2(a):
    return cv2.dct(np.float32(a))
    #return dct( dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return cv2.idct(a)
    #return idct( idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')


def fit_to_size(img, size, allow_crop=False):
    [height, width] = img.shape
    if height > size[0] or width > size[1]:
        if allow_crop:
            cropped_img = img[:min(height, size[0]), :min(width, size[1])]
            result = np.zeros(size, dtype=img.dtype)
            result[:cropped_img.shape[0], :cropped_img.shape[1]] = cropped_img
            return result
        else:
            raise ValueError('Image is too big')
    else:
        result = np.zeros(size, dtype=img.dtype)
        result[:height, :width] = img
        return result
    
def dct_write(img, bits, block_size=8,  c1 = (3,0), c2 = (1,2)):
    [hight, width] = img.shape
    ensure_block_size(img, block_size)
    
    blocksV = hight // block_size
    blocksH = width // block_size

    if blocksV * blocksH < len(bits):
        raise ValueError(f'Not enough space in the image. Image can store {blocksV * blocksH} bits, but {len(bits)} bits were provided')
    bits_extended = np.pad(bits, (0, blocksV * blocksH - len(bits)), 'constant')

    result_img = np.zeros((hight, width), dtype=np.uint8)
    
    for i, bit in enumerate(bits_extended):
        row = i // blocksH
        col = i % blocksH
        roi = img[row*block_size:(row+1)*block_size, col*block_size:(col+1)*block_size]
        dct_roi = dct2(roi)
        
        if bit == 1 and dct_roi[c1] < dct_roi[c2]:
            dct_roi[c1], dct_roi[c2] = dct_roi[c2], dct_roi[c1]
        elif bit == 0 and dct_roi[c1] >= dct_roi[c2]:
            dct_roi[c1], dct_roi[c2] = dct_roi[c2], dct_roi[c1]
        
        result_img[row*block_size:(row+1)*block_size, col*block_size:(col+1)*block_size] = idct2(dct_roi)

    return result_img

def dct_read(img, block_size=8, c1=(3, 0), c2=(1, 2)):
    [height, width] = img.shape
    ensure_block_size(img, block_size)
    
    blocksV = height // block_size
    blocksH = width // block_size

    bits = np.zeros(blocksV * blocksH, dtype=np.int32)

    for i in range(blocksV * blocksH):
        row = i // blocksH
        col = i % blocksH
        roi = img[row*block_size:(row+1)*block_size, col*block_size:(col+1)*block_size]
        dct_roi = dct2(roi)
        bits[i] = 1 if dct_roi[c1] >= dct_roi[c2] else 0

    return bits

def recoverDiagonalMatrix(S, shape_LL):
    Scdiag = np.zeros(shape_LL)
    row = min(shape_LL)
    Scdiag[:row, :row] = np.diag(S)
    return Scdiag
    