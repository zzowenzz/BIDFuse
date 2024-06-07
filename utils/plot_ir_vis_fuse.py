# plot original ir-vis and reconstructed ir-vis together

import os
import cv2
import glob
import argparse
import numpy as np
import shutil
import matplotlib.pyplot as plt

def plot_together(ir_path, vis_path, fuse_path):
    ir_images = sorted(glob.glob(ir_path + '/*'))
    vis_images = sorted(glob.glob(vis_path + '/*'))
    fuse_images = sorted(glob.glob(fuse_path + '/*'))
    assert os.path.basename(ir_images[0]) == os.path.basename(vis_images[0]) == os.path.basename(fuse_images[0])

    for img in ir_images:
        ir = cv2.imread(img)
        vis = cv2.imread(vis_images[ir_images.index(img)])
        vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        fuse = cv2.imread(fuse_images[ir_images.index(img)])
        fuse = cv2.cvtColor(fuse, cv2.COLOR_BGR2RGB)

        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(ir)
        axs[0].set_title('IR')
        axs[0].axis('off')
        axs[1].imshow(vis)
        axs[1].set_title('VIS')
        axs[1].axis('off')
        axs[2].imshow(fuse)
        axs[2].set_title('Fused')
        axs[2].axis('off')
        plt.savefig(os.path.join(os.path.dirname(ir_path)+'_IR_VIS_Fuse', os.path.basename(img)))
        plt.close()




def main():
    parser = argparse.ArgumentParser(description='Plot original ir-vis and reconstructed ir-vis together.')
    parser.add_argument('--ir', help='Path to the original infrared images')
    parser.add_argument('--vis', help='Path to the original visible images')
    parser.add_argument('--fuse', help='Path to the fused images')
    args = parser.parse_args()

    if os.path.exists(os.path.dirname(args.ir) + '_IR_VIS_Fuse'):
        shutil.rmtree(os.path.dirname(args.ir) + '_IR_VIS_Fuse')
    os.mkdir(os.path.dirname(args.ir) + '_IR_VIS_Fuse')
    print('Plotting...')
    plot_together(args.ir, args.vis, args.fuse)
    print('Done!')

if __name__ == '__main__':
    main()