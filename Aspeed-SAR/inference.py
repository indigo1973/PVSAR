import torch
import argparse
import json
import os
import pathlib
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
# model_weights = torch.load("/home/indigo/Local/Aspeed-SAR/model/log/PVT_SAR_epoch_35.pt")
# model.load_state_dict(model_weights)

model = torch.load("/home/indigo/Local/Aspeed-SAR/model/PVSAR_no_aug/PVSAR_no_aug_epoch_20.pth")

# model = torch.load("/home/indigo/Local/Aspeed-SAR/model/PVT_SAR_v2/PVT_SAR_v2_Album_epoch_34.pt").to(device)
# model = torch.load("/home/indigo/Local/Aspeed-SAR/model/PVT_SAR_v1/PVT_SAR_v1test_Album_epoch_61.pth").to(device) #good ori
# model = torch.load("/home/indigo/Local/Aspeed-SAR/model/lightbox_pesudo/PVTSAR_lightbox_008027.pth").to(device) #good ori


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

print("\n--------------  Validation started  -----------------\n")
print("  -- Using config from:\t", args.cfg)
print("  -- Using weights from:\t", config["path_pretrain"])
print("\n-----------------------------------------------------\n")


# ----------------------------- # -----------------------------
#           Transforms          #
# ----------------------------- #

album_val =A.Compose(
    [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),ToTensorV2()],
        A.BboxParams(format='albumentations')       # [xmin, ymin, xmax, ymax] (normalized)
)

# ----------------------------- # -----------------------------
#           Loaders             #
# ----------------------------- #
val_iter = torch.utils.data.DataLoader(SpeedPlusDataset(split='validation', transform_input=album_val), batch_size=config["batch_size_test"],
                                       shuffle=False,num_workers=4,pin_memory=True)

test_iter = torch.utils.data.DataLoader(SpeedPlusDataset(split='lightbox', transform_input=album_val), batch_size=config["batch_size_test"],
                                        shuffle=False,num_workers=0,pin_memory=True)

sun_iter = torch.utils.data.DataLoader(SpeedPlusDataset(split='sunlamp', transform_input=album_val), batch_size=config["batch_size_test"],
                                        shuffle=False,num_workers=0,pin_memory=True)
img_normal_factor = 2/5

speed_total_lightbox = 0
speed_t_total_lightbox = 0
speed_r_total_lightbox = 0
speed_total = 0
speed_t_total = 0
speed_r_total = 0

aux_len = 0
speed_aux = 0

with torch.no_grad():
    model.eval()
    # for i, data in enumerate(tqdm(val_iter)):
    #     img = data["image"].to(device)
    #     # img = aug_intensity_val(img)
    #
    #     q_gt = data["q0"].to(device)  # Qxyzw
    #     t_gt = data["r0"].to(device)
    #     kpts_gt = (data["kpts_2Dim"]).to(device)  # in train got torch.Size([8, 22])
    #     kpts_gt = kpts_gt.reshape([config["batch_size_test"], 11, 2])  # N * 11 *2
    #
    #     # Obtain the prediction
    #
    #     logit, keypoint = model(img)
    #
    #     bs, k, h, w = logit.shape  # [1, 11, 480, 768]
    #     logits = logit.reshape(bs * k, h * w)
    #     logits = logits / logits.sum(dim=1, keepdim=True)
    #     keypoints = keypoint.reshape(bs * k, 2, h * w).permute(0, 2, 1)
    #     maxvals, maxinds = logits.max(dim=1)
    #     coords = keypoints[torch.arange(bs * k, dtype=torch.long).to(keypoints.device), maxinds]
    #     coords[..., 0] *= w
    #     coords[..., 1] *= h
    #     preds = coords.reshape(bs, k, 2).cpu().numpy()  # N x 11 x2
    #     maxvals = maxvals.reshape(bs, k).cpu().numpy()  # N x 11
    #
    #     # kpts_gt = kpts_gt[0].cpu().data.numpy().squeeze()
    #     kpts_pre = preds[0] / img_normal_factor
    #     aux, rvecs, t_est, inliers = cv2.solvePnPRansac(kpts_3d_gt, kpts_pre, cameraMatrix, distCoeffs,
    #                                                     confidence=0.99, reprojectionError=20.0, flags=cv2.USAC_MAGSAC)
    #
    #     q_est = kornia.geometry.axis_angle_to_quaternion(torch.tensor(rvecs.T)).to(device)
    #     t_est = torch.tensor(t_est.T).to(device)
    #
    #     speed, speed_t, speed_r = speedscore.compute_ESA_score(t_est, q_est, t_gt, q_gt, applyThresh=True)
    #     speed_total += speed
    #     speed_t_total += speed_t
    #     speed_r_total += speed_r
    # print('Val get score: ', speed_total / len(val_iter), speed_t_total / len(val_iter), speed_r_total / len(val_iter))


    for i, data in enumerate(tqdm(test_iter)):
        img = data["image"].to(device)

        q_gt = data["q0"].to(device)  # Qxyzw
        t_gt = data["r0"].to(device)
        kpts_gt = (data["kpts_2Dim"]).to(device)  # in train got torch.Size([8, 22])
        kpts_gt = kpts_gt.reshape([config["batch_size_test"], 11, 2])  # N * 11 *2

        # Obtain the prediction

        logit, keypoint = model(img)

        bs, k, h, w = logit.shape  # [1, 11, 480, 768]
        logits = logit.reshape(bs * k, h * w)
        logits = logits / logits.sum(dim=1, keepdim=True)
        keypoints = keypoint.reshape(bs * k, 2, h * w).permute(0, 2, 1)
        maxvals, maxinds = logits.max(dim=1)
        coords = keypoints[torch.arange(bs * k, dtype=torch.long).to(keypoints.device), maxinds]
        coords[..., 0] *= w
        coords[..., 1] *= h
        preds = coords.reshape(bs, k, 2).cpu().numpy()  # N x 11 x2
        maxvals = maxvals.reshape(bs, k).cpu().numpy()  # N x 11

        kpts_gt = kpts_gt[0].cpu().data.numpy().squeeze()
        kpts_pre = preds[0] / img_normal_factor
        aux, rvecs, t_est, inliers = cv2.solvePnPRansac(kpts_3d_gt, kpts_pre, cameraMatrix, distCoeffs,
                                                        confidence=0.99, reprojectionError=25,flags=cv2.USAC_MAGSAC)

        # aux, rvecs, t_est, inliers = cv2.solvePnPRansac(kpts_3d_gt, kpts_pre, cameraMatrix, distCoeffs,
        #                                                 confidence=0.99, reprojectionError=20.0, flags=cv2.SOLVEPNP_EPNP)

        if rvecs is not None:
            q_ESA = kornia.geometry.axis_angle_to_quaternion(torch.tensor(rvecs.T)).to(device)
            t_ESA = torch.tensor(t_est.T).to(device)

        speed, speed_t, speed_r = speedscore.compute_ESA_score(t_ESA, q_ESA, t_gt, q_gt, applyThresh=True)
        # import pdb;pdb.set_trace()
        # if len(inliers) > 8:
        #     aux_len += 1
        #     speed_aux += speed

        # print(kpts_pre-kpts_gt, aux, speed, len(inliers))

        speed_total_lightbox += speed
        speed_t_total_lightbox += speed_t
        speed_r_total_lightbox += speed_r


    print('Lightbox get EASscore: ', speed_total_lightbox / len(test_iter), speed_t_total_lightbox / len(test_iter),
          speed_r_total_lightbox / len(test_iter))
    # print(aux_len, speed_aux/aux_len)

    speed_total_sunlamp = 0
    speed_t_total_sunlamp = 0
    speed_r_total_sunlamp = 0
    aux_len = 0
    speed_aux = 0

    for i, data in enumerate(tqdm(sun_iter)):
        img = data["image"].to(device)
        # img = aug_intensity_test(img)

        q_gt = data["q0"].to(device)  # Qxyzw
        t_gt = data["r0"].to(device)
        kpts_gt = (data["kpts_2Dim"]).to(device)  # in train got torch.Size([8, 22])
        kpts_gt = kpts_gt.reshape([config["batch_size_test"], 11, 2])  # N * 11 *2

        # Obtain the prediction

        logit, keypoint = model(img)

        bs, k, h, w = logit.shape  # [1, 11, 480, 768]
        logits = logit.reshape(bs * k, h * w)
        logits = logits / logits.sum(dim=1, keepdim=True)
        keypoints = keypoint.reshape(bs * k, 2, h * w).permute(0, 2, 1)
        maxvals, maxinds = logits.max(dim=1)
        coords = keypoints[torch.arange(bs * k, dtype=torch.long).to(keypoints.device), maxinds]
        coords[..., 0] *= w
        coords[..., 1] *= h
        preds = coords.reshape(bs, k, 2).cpu().numpy()  # N x 11 x2
        maxvals = maxvals.reshape(bs, k).cpu().numpy()  # N x 11

        kpts_gt = kpts_gt[0].cpu().data.numpy().squeeze()
        kpts_pre = preds[0] / img_normal_factor
        aux, rvecs, t_est, inliers = cv2.solvePnPRansac(kpts_3d_gt, kpts_pre, cameraMatrix, distCoeffs,
                                                        confidence=0.99, reprojectionError=25.0,
                                                        flags=cv2.USAC_MAGSAC)
        if rvecs is not None:
            q_ESA = kornia.geometry.axis_angle_to_quaternion(torch.tensor(rvecs.T)).to(device)
            t_ESA = torch.tensor(t_est.T).to(device)

        speed, speed_t, speed_r = speedscore.compute_ESA_score(t_ESA, q_ESA, t_gt, q_gt, applyThresh=True)
        # if len(inliers) > 7:
        #     aux_len += 1
        #     speed_aux += speed
        # aux_len += 1
        # print(aux_len,'.jpg  ', speed, speed_t, speed_r)
        # import pdb;pdb.set_trace()


        speed_total_sunlamp += speed
        speed_t_total_sunlamp += speed_t
        speed_r_total_sunlamp += speed_r

    print('Sunlamp get EASscore: ', speed_total_sunlamp / len(sun_iter), speed_t_total_sunlamp / len(sun_iter),
          speed_r_total_sunlamp / len(sun_iter))
    # print(aux_len, speed_aux/aux_len)



    #     npts = 0
    #     idxs = []
    #     for idx in range(11):
    #         # if (responses_0[idx] >= 85):
    #         if (maxvals[0][idx]> 0.3):
    #             idxs.append(idx)
    #             npts += 1
    #
    #     # aux, rvecs, t_est, inliers = cv2.solvePnPRansac(kpts_3d_gt, kpts_pre, cameraMatrix, distCoeffs,
    #     #                                                 confidence=0.99, reprojectionError=1.0, flags=cv2.SOLVEPNP_EPNP)
    #
    #     if npts > 5:
    #         aux, rvecs, t_est, inliers  = cv2.solvePnPRansac(kpts_3d_gt[idxs], kpts_pre[idxs], cameraMatrix,
    #                                                                 distCoeffs, confidence=0.99,
    #                                                                 reprojectionError=2.0, flags=cv2.SOLVEPNP_EPNP)
    #     # print(aux)
    #     # import pdb;pdb.set_trace()
    #
    #
    #         if aux:
    #             aux_len +=1
    #
    #
    #             q_est = kornia.geometry.axis_angle_to_quaternion(torch.tensor(rvecs.T)).to(device)
    #             t_est = torch.tensor(t_est.T).to(device)
    #
    #             speed, speed_t, speed_r = speedscore.compute_ESA_score(t_est, q_est, t_gt, q_gt, applyThresh=True)
    #             # import pdb;
    #             # pdb.set_trace()
    #
    #             speed_total_lightbox += speed
    #             speed_t_total_lightbox += speed_t
    #             speed_r_total_lightbox += speed_r
    #

    # print('Test get EASscore: ', speed_total_lightbox / aux_len, speed_t_total_lightbox / aux_len,
    #       speed_r_total_lightbox / aux_len)
    # print(aux_len, ' / ',len(test_iter))

