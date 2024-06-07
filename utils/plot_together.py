# plot original ir-vis and reconstructed ir-vis together

import os
import cv2
import glob
import argparse
import numpy as np
import shutil
import matplotlib.pyplot as plt

def plot_together(ori_ir_path, ori_vis_path, recon_ir_path, recon_vis_path, save_path):
    ori_ir_images = sorted(glob.glob(ori_ir_path + '/*'))
    ori_vis_images = sorted(glob.glob(ori_vis_path + '/*'))
    recon_ir_images = sorted(glob.glob(recon_ir_path + '/*'))
    recon_vis_images = sorted(glob.glob(recon_vis_path + '/*'))

    for img in ori_ir_images:
        ori_ir = cv2.imread(img)
        ori_vis = cv2.imread(ori_vis_images[ori_ir_images.index(img)])
        ori_vis = cv2.cvtColor(ori_vis, cv2.COLOR_BGR2RGB)
        recon_ir = cv2.imread(recon_ir_images[ori_ir_images.index(img)])
        recon_vis = cv2.imread(recon_vis_images[ori_ir_images.index(img)])
        recon_vis = cv2.cvtColor(recon_vis, cv2.COLOR_BGR2RGB)
        # ori = np.concatenate((ori_ir, ori_vis), axis=1)
        # recon = np.concatenate((recon_ir, recon_vis), axis=1)
        # cv2.imwrite(save_path + '/' + os.path.basename(img), np.concatenate((ori, recon), axis=0))

        fig, axs = plt.subplots(2, 2)
        axs[0, 0].imshow(ori_ir)
        axs[0, 0].set_title('Original IR')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(ori_vis)
        axs[0, 1].set_title('Original VIS')
        axs[0, 1].axis('off')
        axs[1, 0].imshow(recon_ir)
        axs[1, 0].set_title('Reconstructed IR')
        axs[1, 0].axis('off')
        axs[1, 1].imshow(recon_vis)
        axs[1, 1].set_title('Reconstructed VIS')
        axs[1, 1].axis('off')
        plt.savefig(save_path + '/' + os.path.basename(img))
        plt.close()




def main():
    parser = argparse.ArgumentParser(description='Plot original ir-vis and reconstructed ir-vis together.')
    parser.add_argument('--ori_ir', help='Path to the original infrared images')
    parser.add_argument('--ori_vis', help='Path to the original visible images')
    parser.add_argument('--recon_ir', help='Path to the reconstructed infrared images')
    parser.add_argument('--recon_vis', help='Path to the reconstructed visible images')
    args = parser.parse_args()

    if os.path.exists(os.path.join(os.path.dirname(args.recon_ir), 'plot')):
        shutil.rmtree(os.path.join(os.path.dirname(args.recon_ir), 'plot'))
    os.makedirs(os.path.join(os.path.dirname(args.recon_ir), 'plot'))
    print('Plotting...')
    plot_together(args.ori_ir, args.ori_vis, args.recon_ir, args.recon_vis, os.path.join(os.path.dirname(args.recon_ir), 'plot'))
    print('Done!')

if __name__ == '__main__':
    main()
