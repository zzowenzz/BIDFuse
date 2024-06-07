from PIL import Image
import os
from collections import defaultdict

def reassemble_images(patches_folder, original_image_size, patch_width=160, patch_height=160):
    # Group patches by original image name
    image_patches = defaultdict(list)
    for filename in os.listdir(patches_folder):
        if filename.endswith(".png"):
            original_image_name, row_str, col_str = filename.rsplit('_', 2)
            image_patches[original_image_name].append(filename)

    # Reassemble each image
    reassembled_images = {}
    for original_image_name, patches in image_patches.items():
        reassembled_image = Image.new('RGB', original_image_size)
        for patch_name in patches:
            _, row_str, col_str = patch_name.rsplit('_', 2)
            row = int(row_str)
            col = int(col_str.split('.')[0])  # Remove file extension

            x = col * patch_width
            y = row * patch_height

            path = os.path.join(patches_folder, patch_name)
            patch = Image.open(path)
            reassembled_image.paste(patch, (x, y))

        reassembled_images[original_image_name] = reassembled_image

    return reassembled_images

# Example usage
original_image_size = (640, 480)  # Replace with the size of your original images
patches_folder = "/home/ubuntu/code/SFuse/log/idea_3_mask_attn_500e/msrs_test_crop_tiny_Fuse"
reassembled_images = reassemble_images(patches_folder, original_image_size)

# Save or display each reassembled image
for img_name, img in reassembled_images.items():
    img.save(f'{patches_folder}/{img_name}.png')
