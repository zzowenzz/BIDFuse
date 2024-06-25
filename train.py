import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse
import time
import logging
import os
import numpy as np
import random
import wandb
import kornia

from utils.dataset import FusionTrainDataset
from utils.net import Restormer_Encoder, Restormer_Decoder, MultiHeadSelfAttention, BidirAttention
from utils.loss import Fusionloss, cc 

def main(args):
    # logger
    if not os.path.exists("./log/"+args.name):
        os.makedirs("./log/"+args.name)
    else:
        print(f"Log directory {args.name} already exists. Rename the experiment.")
        exit()
    log_filename = f'./log/{args.name}/training_log.log'
    logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    logging.info(f"Create log file at: {log_filename}")
    logging.info(f"Arguments are: {args}")

    # wandb
    if args.wandb:
        wandb.init(project=args.project, name=args.name)
        logging.info(f"Initialized wandb project: {args.project}, name: {args.name}")

    # seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    logging.info(f"Set random seed to {seed}")

    # device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # dataset
    dataset_path = args.dataset
    dataset = FusionTrainDataset(dataset_path,imgsz=args.imgsz)
    train_loader = DataLoader(dataset,
                             batch_size=args.batch_size,
                             num_workers=4) 
    logging.info(f"Loaded dataset from {dataset_path} with {len(dataset)} samples.")

    # model
    IR_Encoder = Restormer_Encoder().to(device)
    VIS_Encoder = Restormer_Encoder(inp_channels=3).to(device)
    Decoder = Restormer_Decoder().to(device)
    MSA = MultiHeadSelfAttention(dim=64, num_heads=8, bias=False).to(device)
    BDA = BidirAttention(dim=64, num_heads=8, bias=False).to(device)
    logging.info("Created model.")

    # resume training
    staring_epoch = 1
    def remove_module_prefix(state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        staring_epoch = checkpoint["Epoch"]
        IR_Encoder.load_state_dict(remove_module_prefix(checkpoint["IR_Encoder"]))
        VIS_Encoder.load_state_dict(remove_module_prefix(checkpoint["VIS_Encoder"]))
        Decoder.load_state_dict(remove_module_prefix(checkpoint["Decoder"]))
        BDA.load_state_dict(remove_module_prefix(checkpoint["BDA"]))
        MSA.load_state_dict(remove_module_prefix(checkpoint["MSA"]))
        logging.info(f"Resumed training from {args.resume} at epoch {staring_epoch}.")
    
    # set model to train mode
    IR_Encoder.train()
    VIS_Encoder.train()
    Decoder.train()
    MSA.train()
    BDA.train()
    logging.info("Set model to train mode.")

    # loss function (XX_loss:)
    fusion_loss = Fusionloss()
    mse_loss = nn.MSELoss() 
    l1_loss = nn.L1Loss()
    ssim_loss = kornia.losses.ssim.SSIMLoss(11, reduction='mean') # kornia.losses.ssim.SSIMLoss
    logging.info("Created loss functions.")

    # optimizer (xx_optim)
    IR_Encoder_optim = torch.optim.Adam(IR_Encoder.parameters(), lr=args.lr, weight_decay=args.wd)
    VIS_Encoder_optim = torch.optim.Adam(VIS_Encoder.parameters(), lr=args.lr, weight_decay=args.wd)
    Decoder_optim = torch.optim.Adam(Decoder.parameters(), lr=args.lr, weight_decay=args.wd)
    MSA_optim = torch.optim.Adam(MSA.parameters(), lr=args.lr, weight_decay=args.wd)
    BDA_optim = torch.optim.Adam(BDA.parameters(), lr=args.lr, weight_decay=args.wd)
    logging.info("Created optimizers.")

    # lr scheduler (xx_sche)
    IR_Encoder_sche = torch.optim.lr_scheduler.StepLR(IR_Encoder_optim, step_size=args.step_size, gamma=args.gamma)
    VIS_Encoder_sche = torch.optim.lr_scheduler.StepLR(VIS_Encoder_optim, step_size=args.step_size, gamma=args.gamma)
    Decoder_sche = torch.optim.lr_scheduler.StepLR(Decoder_optim, step_size=args.step_size, gamma=args.gamma)
    MSA_sche = torch.optim.lr_scheduler.StepLR(MSA_optim, step_size=args.step_size, gamma=args.gamma)
    BDA_sche = torch.optim.lr_scheduler.StepLR(BDA_optim, step_size=args.step_size, gamma=args.gamma)
    logging.info("Created lr schedulers.")

    logging.info("Start training ...")
    for epoch in range(staring_epoch, args.epochs+1):
        acc_loss_V, acc_loss_I, acc_loss_Gradient ,acc_loss_F, acc_loss_total = 0.0, 0.0, 0.0, 0.0, 0.0
        for i, (img_name, data_IR, data_VIS) in enumerate(train_loader):
            data_VIS, data_IR = data_VIS.to(device), data_IR.to(device)

            IR_Encoder_optim.zero_grad()
            VIS_Encoder_optim.zero_grad()
            Decoder_optim.zero_grad()
            MSA_optim.zero_grad()
            BDA_optim.zero_grad()

            if epoch < args.epoch_gap:
                # forward
                feature_IR = IR_Encoder(data_IR)
                feature_VIS = VIS_Encoder(data_VIS)
                data_IR_hat = Decoder(data_IR, feature_IR, feature_VIS)
                data_VIS_hat = Decoder(data_VIS, feature_IR, feature_VIS)

                # loss
                loss_V = 5 * ssim_loss(data_VIS, data_VIS_hat) + mse_loss(data_VIS, data_VIS_hat)
                loss_I = 5 * ssim_loss(data_IR, data_IR_hat) + mse_loss(data_IR, data_IR_hat)
                loss_Gradient = l1_loss(kornia.filters.SpatialGradient()(data_VIS),kornia.filters.SpatialGradient()(data_VIS_hat))
                loss_fusion = torch.tensor(0) # not used
                losses = args.vf_weight * loss_V + args.if_weight * loss_I + args.gradient_weight * loss_Gradient
                
                # backprop
                losses.backward()
                # clip gradient
                nn.utils.clip_grad_norm_(IR_Encoder.parameters(), max_norm=args.norm, norm_type=2)
                nn.utils.clip_grad_norm_(VIS_Encoder.parameters(), max_norm=args.norm, norm_type=2)
                nn.utils.clip_grad_norm_(Decoder.parameters(), max_norm=args.norm, norm_type=2)
                # update optimizers every batch
                IR_Encoder_optim.step()
                VIS_Encoder_optim.step()
                Decoder_optim.step()
                
            else:
                # freeze 2 encoders
                layers_to_freeze = [IR_Encoder, VIS_Encoder]
                for layer in layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
                
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

                # loss
                loss_V, loss_I, loss_Gradient = torch.tensor(0), torch.tensor(0), torch.tensor(0) # not used
                loss_fusion, _ , _ = fusion_loss(data_VIS, data_IR, data_Fuse) 
                losses = args.fusion_weight * loss_fusion

                # backprop
                losses.backward()
                # clip gradient
                nn.utils.clip_grad_norm_(Decoder.parameters(), max_norm=args.norm, norm_type=2)
                nn.utils.clip_grad_norm_(MSA.parameters(), max_norm=args.norm, norm_type=2)
                nn.utils.clip_grad_norm_(BDA.parameters(), max_norm=args.norm, norm_type=2)
                # update optimizers every batch
                Decoder_optim.step()
                MSA_optim.step()
                BDA_optim.step()
            # end this batch

        # update lr schedulers and enforce the lower bound of lr
        if epoch < args.epoch_gap:
            IR_Encoder_sche.step()
            VIS_Encoder_sche.step()
            Decoder_sche.step()
            if IR_Encoder_optim.param_groups[0]['lr'] < args.lr_lb:
                IR_Encoder_optim.param_groups[0]['lr'] = args.lr_lb
            if VIS_Encoder_optim.param_groups[0]['lr'] < args.lr_lb:
                VIS_Encoder_optim.param_groups[0]['lr'] = args.lr_lb
            if Decoder_optim.param_groups[0]['lr'] < args.lr_lb:
                Decoder_optim.param_groups[0]['lr'] = args.lr_lb
        else:
            Decoder_sche.step()
            MSA_sche.step()
            BDA_sche.step()
            if Decoder_optim.param_groups[0]['lr'] < args.lr_lb:
                Decoder_optim.param_groups[0]['lr'] = args.lr_lb
            if MSA_optim.param_groups[0]['lr'] < args.lr_lb:
                MSA_optim.param_groups[0]['lr'] = args.lr_lb
            if BDA_optim.param_groups[0]['lr'] < args.lr_lb:
                BDA_optim.param_groups[0]['lr'] = args.lr_lb
        # end this epoch

        # accumulate loss
        acc_loss_V += loss_V.item() / len(train_loader)
        acc_loss_I += loss_I.item() / len(train_loader)
        acc_loss_Gradient += loss_Gradient.item() / len(train_loader)
        acc_loss_F += loss_fusion.item() / len(train_loader)
        acc_loss_total += losses.item() / len(train_loader)

        # log training info
        logging.info(f"Epoch: {epoch+1}/{args.epochs}, loss_v: {acc_loss_V}, loss_i: {acc_loss_I}, loss_gradient: {acc_loss_Gradient}, loss_f: {acc_loss_F}, loss_total: {acc_loss_total}")
        if args.wandb:
            wandb.log({"Epoch":epoch, "loss_v": round(acc_loss_V,4), "loss_i": round(acc_loss_I,4), "loss_gradient": round(acc_loss_Gradient,4), "loss_f": round(acc_loss_F,4), "loss_total": round(acc_loss_total,4)})     

        # model saving
        if epoch % args.save_freq == 0:
            checkpoint = {
                "Epoch": epoch,
                "IR_Encoder": IR_Encoder.state_dict(),
                "VIS_Encoder": VIS_Encoder.state_dict(),
                "Decoder": Decoder.state_dict(),
                "MSA": MSA.state_dict(),
                "BDA": BDA.state_dict(),
            }
            torch.save(checkpoint, f"./log/{args.name}/checkpoint_{epoch}.pth")
            logging.info(f"Model saved at epoch {epoch}")
                
if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Train ICIP Fuse')
    # train
    parser.add_argument("--name", type=str, required=False, help="name of the experiment.")
    parser.add_argument("--project", type=str, default="BIDFuse", help="name of the project.")
    parser.add_argument('--dataset', type=str, default='', help='path to dataset')
    parser.add_argument("--imgsz", type=int, default=224, help="size of the image")
    parser.add_argument('--epochs', type=int, default=120, help='number of epochs')
    parser.add_argument("--epoch_gap", type=int, default=40, help="number of epochs for the first stage")
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument("--wandb", action="store_true", help="use wandb for logging")
    parser.add_argument("--save_freq", type=int, default=40, help="save every few epochs.")

    # model
    parser.add_argument("--resume", required=False, help="path to the checkpoint")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--lr_lb", type=float, default=1e-6, help="lower bound of learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--norm", type=float, default=1e-2, help="max norm of the gradients")
    parser.add_argument("--step_size", type=int, default=20, help="step size of lr scheduler")
    parser.add_argument("--gamma", type=float, default=0.5, help="gamma of lr scheduler")

    # loss
    parser.add_argument("--vf_weight", type=float, default=1, help="weight of visible image")
    parser.add_argument("--if_weight", type=float, default=1, help="weight of infrared image")
    parser.add_argument("--gradient_weight", type=float, default=1, help="weight of gradient loss")
    parser.add_argument("--fusion_weight", type=float, default=1, help="weight of fusion loss")

    args = parser.parse_args()
    main(args)