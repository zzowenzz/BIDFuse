from PIL import Image
import os
import argparse

def center_crop_and_save(input_path, output_path, crop_size=(128, 128)):
    """
    Center crops an image and saves it to the output path.

    Parameters:
    input_path (str): Path to the directory containing images.
    output_path (str): Path to the directory where cropped images will be saved.
    crop_size (tuple): The size (width, height) to crop the image.
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
                # Compute the cropping box
                width, height = img.size
                left = (width - crop_size[0]) / 2
                top = (height - crop_size[1]) / 2
                right = (width + crop_size[0]) / 2
                bottom = (height + crop_size[1]) / 2

                # Crop and save the image
                img_cropped = img.crop((left, top, right, bottom))
                img_cropped.save(os.path.join(output_path, img_name))

if __name__ == '__main__':
    # set arguments
    parser = argparse.ArgumentParser(description='Center crop images')
    parser.add_argument('--dataset', type=str, default='',help='configuration of inference')
    parser.add_argument('--output', type=str, required=True, default='', help='model path')
    args = parser.parse_args()

    center_crop_and_save(args.dataset, args.output)
