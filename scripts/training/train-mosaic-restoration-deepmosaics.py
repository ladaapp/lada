# SPDX-FileCopyrightText: DeepMosaics Authors
# SPDX-License-Identifier: GPL-3.0 AND AGPL-3.0
# Code vendored from: https://github.com/HypoX64/DeepMosaics/

import os
import argparse
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lada.models.deepmosaics.util import data
from lada.models.deepmosaics.util import image_processing as impro
from lada.models.deepmosaics.models import BVDNet,model_util
from skimage.metrics import structural_similarity
from torch.utils.tensorboard import SummaryWriter
from lada.models.deepmosaics.mosaic_video_dataset import MosaicVideoDataset


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--N',type=int,default=2, help='The input tensor shape is H×W×T×C, T = 2N+1')
parser.add_argument('--S',type=int,default=3, help='Stride of 3 frames')
# parser.add_argument('--T',type=int,default=7, help='T = 2N+1')
parser.add_argument('--M',type=int,default=20, help='How many frames read from each videos')
parser.add_argument('--lr',type=float,default=0.0001, help='')
parser.add_argument('--beta1',type=float,default=0.9, help='')
parser.add_argument('--beta2',type=float,default=0.999, help='')
parser.add_argument('--finesize',type=int,default=256, help='')
parser.add_argument('--batchsize',type=int,default=8, help='')
parser.add_argument('--no_gan', action='store_true', help='if specified, do not use gan')
parser.add_argument('--n_blocks',type=int,default=4, help='')
parser.add_argument('--n_layers_D',type=int,default=2, help='')
parser.add_argument('--num_D',type=int,default=3, help='')
parser.add_argument('--lambda_L2',type=float,default=100, help='')
parser.add_argument('--lambda_VGG',type=float,default=1, help='')
parser.add_argument('--lambda_GAN',type=float,default=0.01, help='')
parser.add_argument('--lambda_D',type=float,default=1, help='')
parser.add_argument('--load_thread',type=int,default=2, help='number of thread for loading data')
parser.add_argument('--n_epoch',type=int,default=200, help='')
parser.add_argument('--save_freq',type=int,default=200, help='')
parser.add_argument('--continue_train', action='store_true', help='')
parser.add_argument('--savename',type=str,default='mosaic_restoration_v2', help='')
parser.add_argument('--showresult_freq',type=int,default=200, help='')
parser.add_argument('--eval_freq',type=int,default=200, help='')
parser.add_argument('--showresult_num',type=int,default=6, help='')
parser.add_argument('--pretrained_G_model_path',type=str, help='')
parser.add_argument('--pretrained_D_model_path',type=str, help='')
parser.add_argument('--gpu_id',type=str, default="0")


def ImageQualityEvaluation(tensor1,tensor2):
    batch_len = len(tensor1)
    psnr,ssim = 0,0
    for i in range(len(tensor1)):
        img1,img2 = data.tensor2im(tensor1,rgb2bgr=False,batch_index=i), data.tensor2im(tensor2,rgb2bgr=False,batch_index=i)
        psnr += impro.psnr(img1,img2)
        ssim += structural_similarity(img1,img2,multichannel=True, channel_axis=2)
    return psnr/batch_len,ssim/batch_len

def ShowImage(tensor1,tensor2,tensor3,max_num):
    show_imgs = []
    tensor_batch_size = tensor1.shape[0]
    count = min(max_num, tensor_batch_size)
    for i in range(count):
        show_imgs += [  data.tensor2im(tensor1,rgb2bgr = False,batch_index=i),
                        data.tensor2im(tensor2,rgb2bgr = False,batch_index=i),
                        data.tensor2im(tensor3,rgb2bgr = False,batch_index=i)]
    show_img = impro.splice(show_imgs,  (count,3))
    return show_img

'''
--------------------------Init--------------------------
'''
opt = vars(parser.parse_args())
opt["T"] = 2*opt["N"]+1
if opt["showresult_num"] >opt["batchsize"]:
    opt["showresult_num"] = opt["batchsize"]

experiment_root_dir = os.path.join('../../experiments', opt["savename"])
dir_checkpoint = os.path.join(experiment_root_dir, 'checkpoints')
os.makedirs(dir_checkpoint, exist_ok=True)
tensorboard_savedir = os.path.join(experiment_root_dir, 'tensorboard')
os.makedirs(tensorboard_savedir, exist_ok=True)
writer = SummaryWriter(tensorboard_savedir)

'''
--------------------------Init Network--------------------------
'''
if opt["continue_train"]:
    if not os.path.isfile(opt["pretrained_G_model_path"]):
        opt["continue_train"] = False
        print('can not load generator model path, training on init weight.')
    if not os.path.isfile(opt["pretrained_d_model_path"]):
        opt["continue_train"] = False
        print('can not load discriminator model path, training on init weight.')
if opt["gpu_id"] != '-1' and len(opt["gpu_id"]) == 1:
    torch.backends.cudnn.benchmark = True

netG = BVDNet.define_G(opt["N"],opt["n_blocks"],gpu_id=opt["gpu_id"])
if opt["continue_train"]:
    netG.load_state_dict(torch.load(os.path.join(dir_checkpoint,opt["pretrained_G_model_path"])))
optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt["lr"], betas=(opt["beta1"], opt["beta2"]))
lossfun_L2 = nn.MSELoss()
lossfun_VGG = model_util.VGGLoss(opt["gpu_id"])
if not opt["no_gan"]:
    netD = BVDNet.define_D(n_layers_D=opt["n_layers_D"],num_D=opt["num_D"],gpu_id=opt["gpu_id"])
    if opt["continue_train"]:
        netD.load_state_dict(torch.load(os.path.join(dir_checkpoint, opt["pretrained_d_model_path"])))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt["lr"], betas=(opt["beta1"], opt["beta2"]))
    lossfun_GAND = BVDNet.GANLoss('D')
    lossfun_GANG = BVDNet.GANLoss('G')

'''
--------------------------Init DataLoader--------------------------
'''
train_set_options = {
    "dataroot_gt": "datasets/mosaic_removal_vid/train/img",
    "dataroot_lq": "datasets/mosaic_removal_vid/train/mosaic",
    "dataroot_meta": "datasets/mosaic_removal_vid/train/meta",
    "num_frame": opt["M"],
    "gt_size": opt["finesize"],
    "use_hflip": False,
    "use_rot": False,
    "lq_size": opt["finesize"],
    "dataloader_shuffle": True,
    "dataloader_num_workers": opt["load_thread"],
    "dataloader_batch_size": opt["batchsize"],
    "S": opt["S"],
    "T": opt["T"]
}
train_set = MosaicVideoDataset(train_set_options)
train_loader = DataLoader(train_set,
                          batch_size=train_set_options['dataloader_batch_size'],
                          shuffle=train_set_options['dataloader_shuffle'],
                          num_workers=train_set_options['dataloader_num_workers'],
                          drop_last=True,
                          pin_memory=True)



val_set_options = {
    "dataroot_gt": "datasets/mosaic_removal_vid/val/img",
    "dataroot_lq": "datasets/mosaic_removal_vid/val/mosaic",
    "dataroot_meta": "datasets/mosaic_removal_vid/val/meta",
    "num_frame": -1,
    "gt_size": opt["finesize"],
    "use_hflip": False,
    "use_rot": False,
    "lq_size": opt["finesize"],
    "dataloader_shuffle": False,
    "dataloader_num_workers": 1,
    "dataloader_batch_size": 1,
    "S": opt["S"],
    "T": opt["T"]
}
val_set = MosaicVideoDataset(val_set_options)
val_loader = DataLoader(val_set,
                          batch_size=val_set_options['dataloader_batch_size'],
                          shuffle=val_set_options['dataloader_shuffle'],
                          num_workers=val_set_options['dataloader_num_workers'],
                          drop_last=False,
                          pin_memory=False,
                          prefetch_factor=1)


'''
--------------------------Train--------------------------
'''
device = "cuda:0"
train_loader_size = len(train_loader)
print(f"train_loader size: {train_loader_size}")
for epoch in range(opt["n_epoch"]):
    for sample, (img_gts_batch, img_lqs_batch) in enumerate(train_loader):
        train_iter = (epoch * train_loader_size) + sample
        previous_prediction_frame = None
        for j in range(img_gts_batch.shape[1]):
            # train
            ori_stream, mosaic_stream = img_gts_batch[:,j].to(device), img_lqs_batch[:,j].to(device)
            if previous_prediction_frame is None:
                # start of clip, there is no previous prediction, lets take center frame
                previous_prediction_frame = ori_stream[:,:, ori_stream.shape[2] // 2 + 1].detach().clone().to(device)

            ############### Forward ####################
            # Fake Generator
            out = netG(mosaic_stream,previous_prediction_frame)
            # Discriminator
            if not opt["no_gan"]:
                dis_real = netD(torch.cat((mosaic_stream[:,:,opt["N"]],ori_stream[:,:,opt["N"]].detach()),dim=1))
                dis_fake_D = netD(torch.cat((mosaic_stream[:,:,opt["N"]],out.detach()),dim=1))
                loss_D = lossfun_GAND(dis_fake_D,dis_real) * opt["lambda_GAN"] * opt["lambda_D"]
            # Generator
            loss_L2 = lossfun_L2(out,ori_stream[:,:,opt["N"]]) * opt["lambda_L2"]
            loss_VGG = lossfun_VGG(out,ori_stream[:,:,opt["N"]]) * opt["lambda_VGG"]
            loss_G = loss_L2+loss_VGG
            if not opt["no_gan"]:
                dis_fake_G = netD(torch.cat((mosaic_stream[:,:,opt["N"]],out),dim=1))
                loss_GANG = lossfun_GANG(dis_fake_G) * opt["lambda_GAN"]
                loss_G = loss_G + loss_GANG

            ############### Backward Pass ####################
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            if not opt["no_gan"]:
                optimizer_D.zero_grad()
                loss_D.backward()
                optimizer_D.step()

            previous_prediction_frame = out.detach().clone().to(device)

            if not opt["no_gan"]:
                writer.add_scalar('loss/L2/train', loss_L2.item(), train_iter)
                writer.add_scalar('loss/VGG/train', loss_VGG.item(), train_iter)
                writer.add_scalar('loss/loss_D/train', loss_D.item(), train_iter)
                writer.add_scalar('loss/loss_G/train', loss_G.item(), train_iter)
                writer.add_scalar('lr/lr_D/train', optimizer_D.param_groups[-1]['lr'], train_iter)
                writer.add_scalar('lr/lr_G/train', optimizer_G.param_groups[-1]['lr'], train_iter)
            else:
                writer.add_scalar('loss/L2/train', loss_L2.item(), train_iter)
                writer.add_scalar('loss/VGG/train', loss_VGG.item(), train_iter)

            # save network
            if train_iter%opt["save_freq"] == 0 and train_iter != 0:
                model_util.save(netG, os.path.join(dir_checkpoint,str(train_iter)+'_G.pth'), opt["gpu_id"])
                if not opt["no_gan"]:
                    model_util.save(netD, os.path.join(dir_checkpoint,str(train_iter)+'_D.pth'), opt["gpu_id"])

            # Image quality evaluation
            if train_iter%(opt["showresult_freq"]) == 0:
                psnr, ssim = ImageQualityEvaluation(out,ori_stream[:,:,opt["N"]])
                writer.add_scalar('quality/psnr/train', psnr, train_iter)
                writer.add_scalar('quality/ssim/train', ssim, train_iter)

            # Show result
            if train_iter % (opt["showresult_freq"]) == 0:
                show_img = ShowImage(mosaic_stream[:,:,opt["N"]],out,ori_stream[:,:,opt["N"]],opt["showresult_num"])
                writer.add_image('prediction/train', show_img, train_iter, dataformats='HWC')

        '''
        --------------------------Eval--------------------------
        '''
        if train_iter%(opt["eval_freq"]) == 0 and train_iter != 0:
            eval_results = OrderedDict()
            eval_results['psnr'] = []
            eval_results['ssim'] = []
            eval_results['loss_L2'] = []
            eval_results['loss_VGG'] = []
            eval_results['loss_D'] = []
            eval_results['loss_G'] = []

            for val_sample_idx, val_sample in enumerate(val_loader):
                img_gts_batch, img_lqs_batch = val_sample
                print(f"iter:{train_iter}: evaluating validation sample #{val_sample_idx}, size: {img_gts_batch.nbytes / 1024 / 1024}MB, batch len: {img_gts_batch.shape[1]}")
                previous_prediction_frame = None
                for j in range(img_gts_batch.shape[1]):
                    # eval
                    ori_stream, mosaic_stream = img_gts_batch[:,j].to(device), img_lqs_batch[:,j].to(device)
                    if previous_prediction_frame is None:
                        # start of clip, there is no previous prediction, lets take center frame
                        previous_prediction_frame = ori_stream[:, :, ori_stream.shape[2] // 2 + 1].detach().clone().to(device)

                    with torch.no_grad():
                        # Fake Generator
                        out = netG(mosaic_stream, previous_prediction_frame)
                        # Discriminator
                        if not opt["no_gan"]:
                            dis_real = netD(
                                torch.cat((mosaic_stream[:, :, opt["N"]], ori_stream[:, :, opt["N"]].detach()), dim=1))
                            dis_fake_D = netD(torch.cat((mosaic_stream[:, :, opt["N"]], out.detach()), dim=1))
                            loss_D = lossfun_GAND(dis_fake_D, dis_real) * opt["lambda_GAN"] * opt["lambda_D"]
                            eval_results["loss_D"].append(loss_D.item())
                        # Generator
                        loss_L2 = lossfun_L2(out, ori_stream[:, :, opt["N"]]) * opt["lambda_L2"]
                        loss_VGG = lossfun_VGG(out, ori_stream[:, :, opt["N"]]) * opt["lambda_VGG"]
                        loss_G = loss_L2 + loss_VGG
                        if not opt["no_gan"]:
                            dis_fake_G = netD(torch.cat((mosaic_stream[:, :, opt["N"]], out), dim=1))
                            loss_GANG = lossfun_GANG(dis_fake_G) * opt["lambda_GAN"]
                            loss_G = loss_G + loss_GANG
                        eval_results["loss_L2"].append(loss_L2.item())
                        eval_results["loss_VGG"].append(loss_VGG.item())
                        eval_results["loss_G"].append(loss_G.item())

                        previous_prediction_frame = out.detach().clone().to(device)

                        # Image quality evaluation
                        psnr, ssim = ImageQualityEvaluation(out, ori_stream[:, :, opt["N"]])

                        eval_results["psnr"].append(psnr)
                        eval_results["ssim"].append(ssim)


            psnr = sum(eval_results['psnr']) / len(eval_results['psnr'])
            ssim = sum(eval_results['ssim']) / len(eval_results['ssim'])
            loss_L2 = sum(eval_results['loss_L2']) / len(eval_results['loss_L2'])
            loss_VGG = sum(eval_results['loss_VGG']) / len(eval_results['loss_VGG'])
            loss_G = sum(eval_results['loss_G']) / len(eval_results['loss_G'])
            loss_D = sum(eval_results['loss_D']) / len(eval_results['loss_D'])

            if not opt["no_gan"]:
                writer.add_scalar('loss/L2/eval', loss_L2, train_iter)
                writer.add_scalar('loss/VGG/eval', loss_VGG, train_iter)
                writer.add_scalar('loss/loss_D/eval', loss_D, train_iter)
                writer.add_scalar('loss/loss_G/eval', loss_G, train_iter)
            else:
                writer.add_scalar('loss/L2/eval', loss_L2, train_iter)
                writer.add_scalar('loss/VGG/eval', loss_VGG, train_iter)

            writer.add_scalar('quality/psnr/eval', psnr, train_iter)
            writer.add_scalar('quality/ssim/eval', ssim, train_iter)

            # Show result
            show_img = ShowImage(mosaic_stream[:,:,opt["N"]],out,ori_stream[:,:,opt["N"]],opt["showresult_num"])
            writer.add_image('prediction/eval', show_img, train_iter, dataformats='HWC')
            print(f"iter:{train_iter:d}: AVG val: L2:{loss_L2:.4f}  vgg:{loss_VGG:.4f}  psnr:{psnr:.2f}  ssim:{ssim:.3f}")

writer.flush()
writer.close()