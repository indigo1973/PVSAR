import shutup
shutup.please()

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
# model_weights = torch.load("/home/indigo/Local/Aspeed-SAR/model/log/PVT_SAR_epoch_3.pt")
# model.load_state_dict(model_weights)
# model = torch.load("/home/indigo/Local/Aspeed-SAR/model/PVT_SAR_v2/PVT_SAR_v2_Album_epoch_34.pt").to(device)
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

def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """
    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3), dtype=np.float32)
    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2
    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2
    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1
    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm

def project_keypoints(q_vbs2tango, r_Vo2To_vbs, cameraMatrix, distCoeffs, keypoints):
    ''' Project keypoints.
    Arguments:
        q_vbs2tango:  (4,) numpy.ndarray - unit quaternion from VBS to Tango frame
        r_Vo2To_vbs:  (3,) numpy.ndarray - position vector from VBS to Tango in VBS frame (m)
        cameraMatrix: (3,3) numpy.ndarray - camera intrinsics matrix
        distCoeffs:   (5,) numpy.ndarray - camera distortion coefficients in OpenCV convention
        keypoints:    (3,N) or (N,3) numpy.ndarray - 3D keypoint locations (m)
    Returns:
        points2D: (2,N) numpy.ndarray - projected points (pix)
    '''
    # Size check (3,N)
    if keypoints.shape[0] != 3:
        keypoints = np.transpose(keypoints)

    # Keypoints into 4 x N homogenous coordinates
    keypoints = np.vstack((keypoints, np.ones((1, keypoints.shape[1]))))

    # transformation to image frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q_vbs2tango)),
                          np.expand_dims(r_Vo2To_vbs, 1)))
    xyz      = np.dot(pose_mat, keypoints) # [3 x N]

    closest_index = np.argmin(xyz[2][0:8])
    # import pdb;pdb.set_trace()
    x0, y0   = xyz[0,:] / xyz[2,:], xyz[1,:] / xyz[2,:] # [1 x N] each

    # apply distortion
    r2 = x0*x0 + y0*y0
    cdist = 1 + distCoeffs[0]*r2 + distCoeffs[1]*r2*r2 + distCoeffs[4]*r2*r2*r2
    x  = x0*cdist + distCoeffs[2]*2*x0*y0 + distCoeffs[3]*(r2 + 2*x0*x0)
    y  = y0*cdist + distCoeffs[2]*(r2 + 2*y0*y0) + distCoeffs[3]*2*x0*y0

    # apply camera matrix
    points2D = np.vstack((cameraMatrix[0,0]*x + cameraMatrix[0,2],
                          cameraMatrix[1,1]*y + cameraMatrix[1,2]))

    return points2D, closest_index


print("\n------------  Generate labels started  ----------------\n")
# print("  -- Using config from:\t", args.cfg)
# print("\n-----------------------------------------------------\n")


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
# val_iter = torch.utils.data.DataLoader(SpeedPlusDataset(split='validation', transform_input=album_val), batch_size=config["batch_size_test"],
#                                        shuffle=False,num_workers=4,pin_memory=True)

# test_iter = torch.utils.data.DataLoader(SpeedPlusDataset(split='lightbox', transform_input=album_val), batch_size=config["batch_size_test"],
#                                         shuffle=False,num_workers=0,pin_memory=True)

sun_iter = torch.utils.data.DataLoader(SpeedPlusDataset(split='sunlamp', transform_input=album_val), batch_size=config["batch_size_test"],
                                        shuffle=False,num_workers=0,pin_memory=True)

img_normal_factor = 2/5

speed_total_lightbox = 0
speed_t_total_lightbox = 0
speed_r_total_lightbox = 0

aux_len = 0
speed_aux = 0

outcsvfile = os.path.join(config['root_dir'], config['split_submission'], 'labels_pesudo.csv')
csv = open(outcsvfile, 'w')
print('New labels is Writing to {}'.format(outcsvfile))

kp_arr = np.array([[-0.37  , -0.37  ,  0.37  ,  0.37  , -0.37  , -0.37  ,  0.37  ,
         0.37  , -0.5427,  0.5427,  0.305 ],
       [-0.385 ,  0.385 ,  0.385 , -0.385 , -0.264 ,  0.304 ,  0.304 ,
        -0.264 ,  0.4877,  0.4877, -0.579 ],
       [ 0.3215,  0.3215,  0.3215,  0.3215,  0.    ,  0.    ,  0.    ,
         0.    ,  0.2535,  0.2591,  0.2515]])


with torch.no_grad():
    model.eval()

    # for i, data in enumerate(tqdm(test_iter)):
    for i, data in enumerate(tqdm(sun_iter)):
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

        # kpts_gt = kpts_gt[0].cpu().data.numpy().squeeze()
        kpts_pre = preds[0] / img_normal_factor

        # aux, rvecs, t_est, inliers = cv2.solvePnPRansac(kpts_3d_gt, kpts_pre, cameraMatrix, distCoeffs,
        #                                                 confidence=0.99, reprojectionError=5.0, flags=cv2.USAC_MAGSAC)
        aux, rvecs, t_est, inliers = cv2.solvePnPRansac(kpts_3d_gt, kpts_pre, cameraMatrix, distCoeffs,
                                                        confidence=0.99, flags=cv2.USAC_MAGSAC)


        q_est = kornia.geometry.axis_angle_to_quaternion(torch.tensor(rvecs.T)).to(device)
        t_est = torch.tensor(t_est.T).to(device)

        speed, speed_t, speed_r = speedscore.compute_ESA_score(t_est, q_est, t_gt, q_gt, applyThresh=True)
        # import pdb;pdb.set_trace()
        if len(inliers) > 8:
            aux_len += 1
            speed_aux += speed
            q_vbs2tango = q_est.cpu().data.numpy().squeeze()
            r_Vo2To_vbs = t_est.cpu().data.numpy().squeeze()
            keypoints_loc, closest_index = project_keypoints(q_vbs2tango=q_vbs2tango, r_Vo2To_vbs=r_Vo2To_vbs,
                                                             cameraMatrix=cameraMatrix,
                                                             distCoeffs=distCoeffs, keypoints=kp_arr)
            bbox_gt = [max(keypoints_loc[0].min(), 0) / 1920, max(keypoints_loc[1].min(), 0) / 1200,
                       min(keypoints_loc[0].max(), 1920) / 1920, min(keypoints_loc[1].max(), 1200) / 1200]

            keypoints_loc = keypoints_loc.T  # [11, 2]
            row = (data["y"] + q_vbs2tango.tolist() + r_Vo2To_vbs.tolist())

            for kp_id in range(keypoints_loc.shape[0]):
                row = row + keypoints_loc[kp_id].tolist()
            row += bbox_gt
            row += str(closest_index)

            row = ', '.join([str(e) for e in row])
            # Write csv
            csv.write(row + '\n')


        # print(kpts_pre-kpts_gt, aux, speed, len(inliers))

        speed_total_lightbox += speed
        speed_t_total_lightbox += speed_t
        speed_r_total_lightbox += speed_r

    csv.close()
    #
    # print('Lightbox get EASscore: ', speed_total_lightbox / len(test_iter), speed_t_total_lightbox / len(test_iter),
    #       speed_r_total_lightbox / len(test_iter))
    print('Sunlamp get EASscore: ', speed_total_lightbox / len(sun_iter), speed_t_total_lightbox / len(sun_iter),
          speed_r_total_lightbox / len(sun_iter))
    print(aux_len, speed_aux/aux_len)





