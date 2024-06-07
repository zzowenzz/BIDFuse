from PIL import Image
import os

def crop_images_to_patches(ir_folder, vis_folder, patch_width=160, patch_height=160):
    # Ensure the output directories exist
    os.makedirs(f'{ir_folder}_patches', exist_ok=True)
    os.makedirs(f'{vis_folder}_patches', exist_ok=True)
    
    # Process IR folder
    for filename in os.listdir(ir_folder):
        if filename.endswith(".png"):  # assuming images are in png format, change if they have another format
            path = os.path.join(ir_folder, filename)
            img = Image.open(path)
            for i in range(0, img.width, patch_width):
                for j in range(0, img.height, patch_height):
                    row, col = j // patch_height, i // patch_width
                    box = (i, j, i + patch_width, j + patch_height)
                    patch = img.crop(box)
                    patch.save(os.path.join(f'{ir_folder}_patches', f'{filename[:-4]}_{row}_{col}.png'))

    # Process VIS folder
    for filename in os.listdir(vis_folder):
        if filename.endswith(".png"):  # assuming images are in png format, change if they have another format
            path = os.path.join(vis_folder, filename)
            img = Image.open(path)
            for i in range(0, img.width, patch_width):
                for j in range(0, img.height, patch_height):
                    row, col = j // patch_height, i // patch_width
                    box = (i, j, i + patch_width, j + patch_height)
                    patch = img.crop(box)
                    patch.save(os.path.join(f'{vis_folder}_patches', f'{filename[:-4]}_{row}_{col}.png'))

# Example usage
crop_images_to_patches('/home/ubuntu/code/SFuse/dataset/msrs_test_tiny/ir', '/home/ubuntu/code/SFuse/dataset/msrs_test_tiny/vi')
