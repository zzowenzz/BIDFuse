import os
import argparse
from PIL import Image

def resize_img(input_path, output_path, resize_size=(128, 128)):
    """
    Resize images and saves it to the output path.

    Parameters:
    input_path (str): Path to the directory containing images.
    output_path (str): Path to the directory where cropped images will be saved.
    resize_size (tuple): The size (width, height) to resize the image.
    """

    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Process each image in the input directory
    for img_name in os.listdir(input_path):
        img_path = os.path.join(input_path, img_name)
        
        # Check if it's a file and not a directory
        if os.path.isfile(img_path):
            with Image.open(img_path) as img:
                # Resize and save the image
                img_resized = img.resize(resize_size)
                img_resized.save(os.path.join(output_path, img_name))

if __name__ == '__main__':
    # set arguments
    parser = argparse.ArgumentParser(description='Resize images')
    parser.add_argument('--dataset', type=str, default='',help='configuration of inference')
    parser.add_argument('--output', type=str, required=True, default='', help='model path')
    args = parser.parse_args()

    resize_img(args.dataset, args.output)
