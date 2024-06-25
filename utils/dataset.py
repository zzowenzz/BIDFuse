import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils import data

from PIL import Image
import os

class FusionTrainDataset(Dataset):
    def __init__(self, dataset_folder, imgsz=224):
        super().__init__()
        self.ir_folder = os.path.join(dataset_folder, 'ir')
        self.vis_folder = os.path.join(dataset_folder, 'vi')
        assert len(os.listdir(self.ir_folder)) == len(os.listdir(self.vis_folder)), "The number of images in the two folders must be the same."
        self.image_names = os.listdir(self.ir_folder)
        self.ir_transform = transforms.Compose([ 
            transforms.Resize((imgsz, imgsz)), 
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
        self.vis_transform = transforms.Compose([   
            transforms.Resize((imgsz, imgsz)), 
            # transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        ir_image_path = os.path.join(self.ir_folder, img_name)
        vis_image_path = os.path.join(self.vis_folder, img_name)

        ir_image = self.ir_transform(self.load_and_transform(ir_image_path))
        vis_image = self.vis_transform(self.load_and_transform(vis_image_path).convert('RGB'))
        # vis_image = self.vis_transform(self.load_and_transform(vis_image_path))
        return img_name, ir_image, vis_image

    def load_and_transform(self, image_path):
        # Load image and apply transformations
        image = Image.open(image_path)
        return image

class FusionTestDataset(Dataset):
    def __init__(self, dataset_folder):
        super().__init__()
        self.ir_folder = os.path.join(dataset_folder, 'ir')
        self.vis_folder = os.path.join(dataset_folder, 'vi')
        assert len(os.listdir(self.ir_folder)) == len(os.listdir(self.vis_folder)), "The number of images in the two folders must be the same."
        self.image_names = os.listdir(self.ir_folder)
        self.ir_transform = transforms.Compose([ 
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
        self.vis_transform = transforms.Compose([   
            # transforms.Grayscale(1),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        ir_image_path = os.path.join(self.ir_folder, img_name)
        vis_image_path = os.path.join(self.vis_folder, img_name)

        ir_image = self.ir_transform(self.load_and_transform(ir_image_path))
        vis_image = self.vis_transform(self.load_and_transform(vis_image_path).convert('RGB'))
        # vis_image = self.vis_transform(self.load_and_transform(vis_image_path))
        return img_name, ir_image, vis_image

    def load_and_transform(self, image_path):
        # Load image and apply transformations
        image = Image.open(image_path)
        return image
