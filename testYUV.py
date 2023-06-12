import argparse 
import math
import random
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import Kodak24Dataset
from torch import nn
from lightning_model_DLEC import LightningModel
from gained_mean_scale_hyperprior_DLEC import GainedMeanScaleHyperprior
from torchvision.utils import save_image
import os

def ycbcr_to_rgb(ycbcr_images):
    # YCbCr to RGB 변환
    rgb_images = []
    for ycbcr_image in ycbcr_images:
        ycbcr_image = transforms.ToPILImage()(ycbcr_image)  # Tensor를 PIL 이미지로 변환
        rgb_image = ycbcr_image.convert("RGB")  # YCbCr을 RGB로 변환
        rgb_image = transforms.ToTensor()(rgb_image)  # PIL 이미지를 Tensor로 변환
        rgb_images.append(rgb_image)
    rgb_images = torch.stack(rgb_images, dim=0)  # 이미지들을 하나의 텐서로 결합
    return rgb_images
        
        
def rgb_to_ycbcr(images):
    # RGB to YCbCr 변환
    ycbcr_images = []
    for image in images:
        image = transforms.ToPILImage()(image)  # Tensor를 PIL 이미지로 변환
        ycbcr_image = image.convert("YCbCr")  # RGB를 YCbCr로 변환
        ycbcr_image = transforms.ToTensor()(ycbcr_image)  # PIL 이미지를 Tensor로 변환
        ycbcr_images.append(ycbcr_image)
    ycbcr_images = torch.stack(ycbcr_images, dim=0)  # 이미지들을 하나의 텐서로 결합
    return ycbcr_images


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--seed", type=float, default=123, help="Set random seed for reproducibility"
    )
    parser.add_argument(
        "--write_stream", type=float, default=1, help="if write stream"
    )
    parser.add_argument(
        "--write_recon_image", type=float, default=0, help="if write recon image"
    )
    parser.add_argument(
        "--dataset", type=str, default='')
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args



class RateDistortionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        N, _, H, W = target.size()
        out = {}
        num_pixels = N * H * W

        out["bpp_loss"] = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in output["likelihoods"].values()
        )
        return out


def build_dataset(args):
    # Warning, the order of the transform composition should be kept.
    kodak_transform = transforms.Compose(
        [transforms.ToTensor()]
    )

    kodak_dataset = Kodak24Dataset(
        args.dataset,
        transform=kodak_transform,
    )

    kodak_dataloader = DataLoader(
        kodak_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    return kodak_dataloader


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def comrpess_and_decompress(model, test_dataloader, args, device):
   
    model.update()

    flag = True
    #for j in range(10):
    # for j in range(8):
    for j in range(4):
        criterion = RateDistortionLoss()
        psnr = AverageMeter()
        bpp = AverageMeter()        
        for i, x in enumerate(test_dataloader):
            x = x.to(device)
            luma = x[:,0,:,:].unsqueeze(dim=1)
            chroma = x[:,1:,:,:]
            

            if args.write_stream:
                # compress
                compressed = model.compress(x, j) #, 0)  # {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
                strings = compressed['strings']
                shape = compressed['shape']

                # decompress
                decompressed = model.decompress(strings, shape, j) #, 0)
                x_hat = decompressed['x_hat']

                bpp_y = (len(strings[0][0])) * 8 / (x.shape[2] * x.shape[3])
                bpp_z = (len(strings[1][0])) * 8 / (x.shape[2] * x.shape[3])
                bpp_ = bpp_y + bpp_z
                
                x_hat_rgb = ycbcr_to_rgb(x_hat)
                if args.write_recon_image and j == 0:
                    os.makedirs('./dlec_recon_images0/', exist_ok=True)
                    save_image(x_hat_rgb, f'./dlec_recon_images0/kodim{str(i+1).zfill(2)}.png')
                if args.write_recon_image and j == 1:
                    os.makedirs('./dlec_recon_images1/', exist_ok=True)
                    save_image(x_hat_rgb, f'./dlec_recon_images1/kodim{str(i+1).zfill(2)}.png')
                if args.write_recon_image and j == 2:
                    os.makedirs('./dlec_recon_images2/', exist_ok=True)
                    save_image(x_hat_rgb, f'./dlec_recon_images2/kodim{str(i+1).zfill(2)}.png')
                if args.write_recon_image and j == 3:
                    os.makedirs('./dlec_recon_images3/', exist_ok=True)
                    save_image(x_hat_rgb, f'./dlec_recon_images3/kodim{str(i+1).zfill(2)}.png')
                
            else:
                out_net = model(x)
                out_criterion = criterion(out_net, x)
                x_hat = out_net['x_hat'].clamp_(0, 1)
                bpp_ = out_criterion['bpp_loss'].detach().cpu()
            
            a = nn.MSELoss()
            
            # 이부분은 YCBCR PSNR 구할 때 사용
            
            ycbcr_original = rgb_to_ycbcr(x).to("cuda")
            luma_recon = x_hat[:,0,:,:].unsqueeze(dim=1)
            chroma_recon = x_hat[:,1:,:,:]
            luma_original = ycbcr_original[:,0,:,:].unsqueeze(dim=1)
            chroma_original = ycbcr_original[:,1:,:,:]

            distortion_loss_luma = a(luma_recon, luma_original)
            distortion_loss_chroma = a(chroma_recon, chroma_original)

            #alphaaa = [1, 0.25, 0.1, 0.025]
            #distortion_loss = ((1 * distortion_loss_luma) + (2 * alphaaa[j] * distortion_loss_chroma)) / (1 + 2 * alphaaa[j])
            distortion_loss = ((4 * distortion_loss_luma) + (2 * distortion_loss_chroma)) / (6)

            psnr_ = 10 * (torch.log(1 * 1 / distortion_loss) / math.log(10))
            #psnr_ = 10 * (torch.log(1 * 1 / distortion_loss_luma) / math.log(10))
            

            
            # 이부분은 RGB PSNR 구할 때 사용
            """
            recon_rgb = ycbcr_to_rgb(x_hat).to("cuda")
            x_rgb = ycbcr_to_rgb(x).to("cuda")
            mse_ = a(recon_rgb, x_rgb)
            psnr_ = 10 * (torch.log(1 * 1 / mse_) / math.log(10))                        
            """
            # print(
            #     f"{i} \tPSNR: {psnr_:.3f} |"
            #     f"\tBPP: {bpp_:.3f}"
            # )
            
            print(i, " : ", psnr_, "bpp : ", bpp_)
            
            bpp.update(bpp_)
            psnr.update(psnr_.detach().cpu())
            
        print(
            f"{j}"
            f"\tTest PSNR: {psnr.avg:.3f} |"
            f"\tTest BPP: {bpp.avg:.3f}"
        )

def main(argv):
    args = parse_args(argv)
    torch.backends.cudnn.deterministic = True

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GainedMeanScaleHyperprior(N=192, M=320)    
    lightning_model = LightningModel(model).eval().to(device)

    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=('cpu'))
        lightning_model.load_state_dict(checkpoint["state_dict"])

    test_dataloader = build_dataset(args)
    with torch.no_grad():
        comrpess_and_decompress(lightning_model, test_dataloader, args, device)


if __name__ == "__main__":
    main(sys.argv[1:])
