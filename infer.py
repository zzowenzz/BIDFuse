import torch
from torch.utils.data import DataLoader
import torchvision

from pathlib import Path
import argparse

from utils.dataset import FusionTestDataset
from utils.net import Restormer_Encoder, Restormer_Decoder, MultiHeadSelfAttention, BidirAttention

def main(args):
    # output dir (log_folder/images_dataset_checkpoint/)
    if not (Path(args.checkpoint).parent / Path("images_" + str(Path(args.dataset).parent.name) + "_epoch" + str(Path(args.checkpoint).stem.split("_")[-1]))).exists(): 
        (Path(args.checkpoint).parent / Path("images_" + str(Path(args.dataset).parent.name) + "_epoch" + str(Path(args.checkpoint).stem.split("_")[-1]))).mkdir(parents=True)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    dataset_path = args.dataset
    dataset = FusionTestDataset(dataset_path)
    test_loader = DataLoader(dataset,
                             batch_size=1,
                             num_workers=4) 

    # model
    IR_Encoder = Restormer_Encoder().to(device)
    VIS_Encoder = Restormer_Encoder(inp_channels=3).to(device)
    Decoder = Restormer_Decoder().to(device)
    MSA = MultiHeadSelfAttention(dim=64, num_heads=8, bias=False).to(device)
    BDA = BidirAttention(dim=64, num_heads=8, bias=False).to(device)

    # load pre-trained weights
    checkpoint = torch.load(args.checkpoint, map_location=device)
    IR_Encoder.load_state_dict(checkpoint['IR_Encoder'])
    VIS_Encoder.load_state_dict(checkpoint['VIS_Encoder'])
    Decoder.load_state_dict(checkpoint['Decoder'])
    MSA.load_state_dict(checkpoint['MSA'])
    BDA.load_state_dict(checkpoint['BDA'])

    print("Start inference...")
    with torch.no_grad():
        for i, (img_name, data_IR, data_VIS) in enumerate(test_loader):
            data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)

            # forward
            feature_IR = IR_Encoder(data_IR)
            feature_VIS = VIS_Encoder(data_VIS)
            # feature fusion
            attn_IR_IR, attn_VIS_VIS = MSA(feature_IR), MSA(feature_VIS)
            attn_IR_VIS = BDA(feature_IR, feature_VIS, feature_VIS)
            attn_VIS_IR = BDA(feature_VIS, feature_IR, feature_IR)
            attn_map_IR = torch.cat([attn_IR_IR, attn_IR_VIS], dim=1)
            attn_map_VIS = torch.cat([attn_VIS_VIS, attn_VIS_IR], dim=1)
            data_Fuse = Decoder(data_VIS, attn_map_IR, attn_map_VIS)

            # save image
            # import pdb; pdb.set_trace()
            save_path = Path(args.checkpoint).parent / Path("images_" + str(Path(args.dataset).parent.name) + "_epoch" + str(Path(args.checkpoint).stem.split("_")[-1])) / img_name[0]
            torchvision.utils.save_image(data_Fuse, save_path)

    print("Inference done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Infer a baseline model")
    parser.add_argument("--dataset", type=str, default='', required=True, help='path to dataset folder')
    parser.add_argument("--checkpoint", type=str, required=True, help='path to checkpoint file')

    args = parser.parse_args()
    main(args)