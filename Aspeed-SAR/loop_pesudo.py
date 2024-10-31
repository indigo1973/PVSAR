import shutup
shutup.please()

import torch
import argparse
import json
import os
import pathlib
import wandb
from pytorch3d.implicitron.dataset.utils import bbox_xywh_to_xyxy

# from model import Vit_b
from model_pvtSAR import Vit_b
from torchvision import transforms
from tqdm import tqdm
import kornia.augmentation as K
from data_loader import SpeedPlusDataset
import cv2
import scipy
import numpy as np
import speedscore
import kornia

from src.randomsunflare import RandomSunFlare
from src.coarsedropout import CoarseDropout
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def load_config(config_path):
    with open(config_path) as json_file:
        return json.load(json_file)
# ----------------------------- #
#     Initialization            #
# ----------------------------- #
parser = argparse.ArgumentParser(description="Spacecraft Pose Estimation: PVTv2 + SAR + PnP")
parser.add_argument("-c", "--cfg", metavar="DIR", help="Path to the configuration file", required=True)
args = parser.parse_args()
config = load_config(args.cfg)
device = config["device"]

# Create the direcctories for the weight checkpoints
path_checkpoints = os.path.join(config["path_results"],args.cfg,"ckpt")
pathlib.Path(path_checkpoints).mkdir(parents=True, exist_ok=True)

# model = Vit_b().to(device)
# model = torch.load("/home/indigo/Local/Aspeed-SAR/model/PVT_SAR_v2/PVT_SAR_v1p5_Album_epoch_29.pt").to(device)
# model = torch.load("/home/indigo/Local/Aspeed-SAR/model/lightbox_pesudo/PVT_SAR_v1_pesudo_lightbox_epoch_4.pth").to(device)
model = torch.load("/home/indigo/Local/Aspeed-SAR/model/sunlamp_pesudo/PVT_SAR_v1_sunlamp_epoch_4.pth").to(device)

def get_tango_kpts(config,gpu):
    world_kpts = scipy.io.loadmat(config["root_dir"] + "tangoPoints.mat")
    world_kpts = world_kpts["tango3Dpoints"].astype(np.float32).T
    world_kpts = torch.from_numpy(world_kpts).unsqueeze(0).repeat([config["batch_size"],1,1]).cuda(gpu).type(torch.float)
    return world_kpts

with open('/home/indigo/dataset/speedpp/camera.json') as f:
    cam = json.load(f)
cameraMatrix = np.array(cam['cameraMatrix'], dtype=np.float32)
distCoeffs = np.array(cam['distCoeffs'], dtype=np.float32)
kpts_3d_gt = get_tango_kpts(config, device)[0].cpu().numpy()

print("\n--------------  Training started  -------------------\n")
# print("  -- Using config from:\t", args.cfg)
# print("  -- Using weights from:\t", config["path_pretrain"])
# print("  -- Saving weights to:\t", path_checkpoints)
# print("\n-----------------------------------------------------\n")



# ----------------------------- # -----------------------------
#           Transforms          #
# ----------------------------- #
tforms = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((config["rows"], config["cols"]))
    ])

album_val =A.Compose(
    [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),ToTensorV2()],
        A.BboxParams(format='albumentations')       # [xmin, ymin, xmax, ymax] (normalized)
)


# ----------------------------- # -----------------------------
#           Loaders             #
# ----------------------------- #
# train_iter = torch.utils.data.DataLoader(SpeedPlusDataset(split='train', transform_input=tforms), batch_size=config["batch_size"], shuffle=True)
# val_iter = torch.utils.data.DataLoader(SpeedPlusDataset(split='validation', transform_input=tforms), batch_size=config["batch_size_test"], shuffle=False)
# test_iter = torch.utils.data.DataLoader(SpeedPlusDataset(split='lightbox', transform_input=tforms), batch_size=config["batch_size_test"], shuffle=False)


train_iter = torch.utils.data.DataLoader(SpeedPlusDataset(split=config['split_submission'], transform_input=album_val, pesudo=True), batch_size=config["batch_size"],
                                         shuffle=True,num_workers=4 ,pin_memory=True)
val_iter = torch.utils.data.DataLoader(SpeedPlusDataset(split=config['split_submission'], transform_input=album_val, pesudo=False), batch_size=config["batch_size_test"],
                                         shuffle=False,num_workers=0 ,pin_memory=True)



# ----------------------------- # -----------------------------
#        Optimizer/Loss         #
# ----------------------------- #
optim_params = [ {"params": model.parameters(),
                  "lr": config["lr"]}]

optimizer = torch.optim.AdamW(optim_params)
mse_loss = torch.nn.MSELoss()

img_normal_factor = 2/5
# ----------------------------- # -----------------------------
#        Train/Val Loop         #
# ----------------------------- #
for epoch in range(config["start_epoch"], config["total_epochs"]):
    # ----------------------------- #
    #        Train Epoch            #
    # ----------------------------- #
    model.train()
    loss_mean_batch_list = []
    loss_max_batch_list = []


    for params in optimizer.param_groups:
        # if epoch >= 4:
        #     params['lr'] -= 3e-6
        print("Epoch: ", epoch, "       ", 'Learing Rate is ',params['lr'])

    for i, data in enumerate(tqdm(train_iter)):
        optimizer.zero_grad()

        img = data["image"].to(device)
        # img = aug_intensity(img)    #N x 3 x 480 x 768

        q_gt = data["q0"].to(device)        # Qxyzw
        t_gt = data["r0"].to(device)
        kpts_gt = (data["kpts_2Dim"]).to(device)  #in train got torch.Size([8, 22])
        kpts_gt = kpts_gt.reshape([kpts_gt.shape[0], 11, 2])  # N * 11 *2


        # Obtain the prediction
        logit, keypoint = model(img)

        bs, k, h, w = logit.size()
        assert k == keypoint.size(1) and keypoint.size(2) == 2

        # get cls score
        cls_score = logit.reshape(bs*k, h*w)
        # get reg score
        pred_keypoints = keypoint.reshape(bs*k, 2, h*w).permute(0, 2, 1)


        kpts_gt = kpts_gt.reshape(bs * k, 2)[:, None, :]
        kpts_gt[..., 0] /= (w/img_normal_factor)
        kpts_gt[..., 1] /= (h/img_normal_factor)

        dist_mat = torch.abs(pred_keypoints - kpts_gt)
        dist_mat = dist_mat * 16.0
        # dist_mat = dist_mat * 64.0
        reg_score = torch.exp(-dist_mat.sum(dim=2))

        norm_cls_score = cls_score / cls_score.sum(dim=1, keepdim=True)
        # norm_cls_score = cls_score

        normcls2reg_loss = torch.sum(norm_cls_score * reg_score, dim=1)
        normcls2reg_loss = -torch.log(normcls2reg_loss + 1e-6)
        loss = normcls2reg_loss.mean()

        loss_mean_batch_list.append(loss.item())   #dxdydz #l is ESA Score
        loss_max_batch_list.append(normcls2reg_loss.max().item())

        loss.backward()
        optimizer.step()
        # import pdb;pdb.set_trace()
    loss_mean = (sum(loss_mean_batch_list)/len(loss_mean_batch_list))
    loss_max  = (max(loss_max_batch_list))
    print(f'Train got loss_mean {loss_mean} & loss_max {loss_max}')

    # torch.save(model, f"/home/indigo/Local/Aspeed-SAR/model/lightbox_pesudo/PVT_SAR_v1_pesudo_lightbox_epoch_{epoch}.pth")
    torch.save(model, f"/home/indigo/Local/Aspeed-SAR/model/sunlamp_pesudo/PVT_SAR_v1_sunlamp_epoch_{epoch}.pth")

    # ----------------------------- #
    #        Val Epoch              #
    # ----------------------------- #
    speed_total = 0
    speed_t_total = 0
    speed_r_total = 0

    if epoch >=4:
        with torch.no_grad():
            model.eval()

            for i, data in enumerate(tqdm(val_iter)):

                img = data["image"].to(device)
                # img = aug_intensity_val(img)

                q_gt = data["q0"].to(device)        # Qxyzw
                t_gt = data["r0"].to(device)
                kpts_gt = (data["kpts_2Dim"]).to(device)  #in train got torch.Size([8, 22])
                kpts_gt = kpts_gt.reshape([config["batch_size_test"], 11, 2])  # N * 11 *2

                # Obtain the prediction

                logit, keypoint = model(img)

                bs, k, h, w = logit.shape       #   [1, 11, 480, 768]
                logits = logit.reshape(bs*k, h*w)
                logits = logits / logits.sum(dim=1, keepdim=True)
                keypoints = keypoint.reshape(bs*k, 2, h*w).permute(0, 2, 1)
                maxvals, maxinds = logits.max(dim=1)
                coords = keypoints[torch.arange(bs*k, dtype=torch.long).to(keypoints.device), maxinds]
                coords[..., 0] *= w
                coords[..., 1] *= h
                preds = coords.reshape(bs, k, 2).cpu().numpy()  #N x 11 x2
                maxvals = maxvals.reshape(bs, k).cpu().numpy()  #N x 11

                # kpts_gt = kpts_gt[0].cpu().data.numpy().squeeze()
                kpts_pre = preds[0] / img_normal_factor
                aux, rvecs, t_est, inliers = cv2.solvePnPRansac(kpts_3d_gt, kpts_pre, cameraMatrix, distCoeffs,
                                                                confidence=0.99, reprojectionError=20.0,flags=cv2.USAC_MAGSAC)

                q_est = kornia.geometry.axis_angle_to_quaternion(torch.tensor(rvecs.T)).to(device)
                t_est = torch.tensor(t_est.T).to(device)

                speed, speed_t, speed_r = speedscore.compute_ESA_score(t_est, q_est, t_gt, q_gt, applyThresh=True)
                speed_total += speed
                speed_t_total += speed_t
                speed_r_total += speed_r
            print('Val get score: ',speed_total / len(val_iter), speed_t_total / len(val_iter), speed_r_total / len(val_iter))
            print(len(val_iter))
            torch.save(model,
                       # f"/home/indigo/Local/Aspeed-SAR/model/lightbox_pesudo/PVTSAR_v1_lightbox_{speed_total / len(val_iter)}.pth")
                       f"/home/indigo/Local/Aspeed-SAR/model/sunlamp_pesudo/PVT_SAR_v1_sunlamp_{speed_total / len(val_iter)}.pth")
