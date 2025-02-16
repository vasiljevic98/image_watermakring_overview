import cv2
import numpy as np

def contrast_stretch(img):
    stretched_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return stretched_img

def histogram_equalization(img):
    equalized_img = cv2.equalizeHist(img)
    return equalized_img


def drop_bit_layer(img, position):
    return img & ~(1 << position)


def resize_reset(img, scale):
    org_height, org_width = img.shape[:2]
    resized_img = cv2.resize(img, (int(org_width*scale), int(org_height*scale)))
    resized_img = cv2.resize(resized_img, (org_width, org_height))
    return resized_img

def crop_edge(img, border, resizeToOriginal=True):
    org_height, org_width = img.shape[:2]
    cropped_img = img[border:org_height-border, border:org_width-border]
    if not resizeToOriginal:
        cropped_img = cv2.resize(cropped_img, (org_width, org_height))
    return cropped_img

def scale_crop(img, scale):
    if scale < 1:
        return resize_reset(img, scale)
    
    org_height, org_width = img.shape[:2]
    resized_img = cv2.resize(img, (int(org_width*scale), int(org_height*scale)))
    border = (resized_img.shape[0]-org_height)//2
    cropped_img = resized_img[border:org_height+border, border:org_width+border]
    org_dim_img = cv2.resize(cropped_img, (org_width, org_height))
    return org_dim_img    


def add_gaussian_noise(img, sigma):
    img = img.astype(np.float32)
    noise = np.random.normal(0, sigma, img.shape)
    added_noise_img = img + noise
    cliped_img = np.clip(added_noise_img, 0, 255)
    return cliped_img.astype(np.uint8)

def jpg_compression(img, quality):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img, encode_param)
    encimg = cv2.imdecode(encimg, 1)
    return encimg[:, :, 0]

def rotation(img, angle, zoom_to_fit=True):
    org_height, org_width = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((org_width/2, org_height/2), angle, 1)
    rotated_img = cv2.warpAffine(img, rotation_matrix, (org_width, org_height))
    if zoom_to_fit:
        diag_angle = np.arctan(org_height/org_width)
        angle_rad = np.deg2rad(angle)
        scale= np.sin(angle_rad+diag_angle)/np.sin(diag_angle)
        rotated_img = scale_crop(rotated_img, scale)
    return rotated_img

def drew_random_lines_on_img(img, number_of_lines=100):
    height, width = img.shape[:2]
    for _ in range(number_of_lines):
        x1, x2 = np.random.randint(0, width, 2)
        y1, y2 = np.random.randint(0, height, 2)
        thickness = np.random.randint(1,3)
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), thickness)
    return img
