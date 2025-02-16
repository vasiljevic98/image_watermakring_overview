import argparse
import time
import os
import cv2

from transformations import add_gaussian_noise, contrast_stretch, crop_edge, drew_random_lines_on_img, drop_bit_layer, histogram_equalization, jpg_compression, resize_reset, rotation, scale_crop

parser = argparse.ArgumentParser(
                    prog='Tester',
                    description='Testing  all combinations of encodings and attacks',
                    epilog='Text at the bottom of help')

parser.add_argument('-i', '--image')    
parser.add_argument('-o', '--output_dir')
parser.add_argument('-t', '--transformations', nargs='+')
parser.add_argument('-ts', '--time_stamp', action='store_true')


def transform( transform_name, img, **kwargs):
    if transform_name == 'contrast_stretch':
        return contrast_stretch(img)
    elif transform_name == 'histogram_equalization':
        return histogram_equalization(img, **kwargs)
    elif transform_name == 'drop_bit_layer':
        return drop_bit_layer(img, **kwargs)
    elif transform_name == 'resize_reset':
        return resize_reset(img, **kwargs)
    elif transform_name == 'crop_resize':
        return crop_edge(img, **kwargs)
    elif transform_name == 'scale_crop':
        return scale_crop(img, **kwargs)
    elif transform_name == 'add_gaussian_noise':
        return add_gaussian_noise(img, **kwargs)
    elif transform_name == 'jpg_compression':
        return jpg_compression(img, **kwargs)
    elif transform_name == 'rotation':
        return rotation(img, **kwargs)
    elif transform_name == 'drew_random_lines_on_img':
        return drew_random_lines_on_img(img, **kwargs)
    else:
        raise ValueError(f"Unknown transform: {transform_name}")


args = parser.parse_args()

defaults = {
    "contrast_stretch": {},
    "histogram_equalization": {},
    "drop_bit_layer": {'position': 0},
    "resize_reset": {'scale': 0.5},
    "crop_resize": {'border': 10},
    "scale_crop": {'scale': 1.05},
    "add_gaussian_noise": {'sigma': 10},
    "jpg_compression": {'quality': 50},
    "rotation": {'angle': 1, 'zoom_to_fit': True},
    "drew_random_lines_on_img": {'number_of_lines': 100}
}

transformations = [
    "contrast_stretch",
    "histogram_equalization",
    "drop_bit_layer",
    "resize_reset",
    "crop_resize",
    "scale_crop",
    "add_gaussian_noise",
    "jpg_compression",
    "rotation",
    "drew_random_lines_on_img"
] if args.transformations is None else args.transformations
                  


print(args)

if args.image is None:
    raise ValueError("Image not provided")


image_path = args.image
image_filename = os.path.basename(image_path)
image_filename_no_ext = os.path.splitext(image_filename)[0]
image_parent_dir = os.path.dirname(image_path)

should_add_timestamp = args.time_stamp

current_time_unix = int(time.time())


if args.output_dir is None:
    output_dir = f"output-{image_filename_no_ext}"
else:
    output_dir = args.output_dir

if should_add_timestamp:
    output_dir += f"-{current_time_unix}"

print(f"Output dir: {output_dir}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for transform_name in transformations:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    transformed = transform(transform_name, img, **defaults[transform_name])
    
    output_path = f"{output_dir}/{transform_name}.png"
    cv2.imwrite(output_path, transformed)
    print(f"Saved {transform_name} to {output_path}")

    # example cmd: python src/test.py -i assets\imgs\lenna.png -o output -ts 