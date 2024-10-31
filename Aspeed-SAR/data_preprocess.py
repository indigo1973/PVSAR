import json
import argparse
import os
import numpy as np
from tqdm import tqdm
from transforms3d import affines, quaternions
import scipy
from scipy.spatial.transform import  Rotation as R
import cv2
import torch
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion, quaternion_to_matrix, matrix_to_rotation_6d
from trimesh.proximity import closest_point

PROJROOTDIR = '/home/indigo/Local/Aspeed-SAR'
DATAROOTDIR = '/home/indigo/dataset/'


parser = argparse.ArgumentParser('Generating CSV files')
parser.add_argument('--projroot',   type=str, default=PROJROOTDIR)
parser.add_argument('--dataroot',   type=str, default=DATAROOTDIR)
parser.add_argument('--dataname',   type=str, default='speedplusv2')

parser.add_argument('--num_keypoints',   type=int,   default=11)
parser.add_argument('--num_neighbors',   type=int,   default=5)
parser.add_argument('--keypts_3d_model', type=str, default='/src/utils/tangoPoints.mat')

# parser.add_argument('--domain',   type=str, default='synthetic')
# parser.add_argument('--domain',   type=str, default='lightbox')
parser.add_argument('--domain',   type=str, default='sunlamp')

# parser.add_argument('--domain_split',   type=str, default='train')
# parser.add_argument('--domain_split',   type=str,  default='validation')
parser.add_argument('--domain_split',   type=str, default='test')

# parser.add_argument('--domain_split',   type=str, default='train')
# parser.add_argument('--jsonfile', type=str, default=f'{domain_split}.json')
# parser.add_argument('--csvfile',  type=str, default=f'{domain_split}_labels.csv')

parser.add_argument('--model_input_img_size',  default=[768, 480])       #!!!
cfg = parser.parse_args()

if cfg.domain_split == 'test':
    output_labels_path = cfg.domain
else: output_labels_path = cfg.domain_split





# Read camera

def load_camera_intrinsics(camera_json):
    with open(camera_json) as f:
        cam = json.load(f)
    cameraMatrix = np.array(cam['cameraMatrix'], dtype=np.float32)
    distCoeffs   = np.array(cam['distCoeffs'],   dtype=np.float32)

    return cameraMatrix, distCoeffs


camerafile = os.path.join(cfg.dataroot, cfg.dataname, 'camera.json')
cameraMatrix, distCoeffs = load_camera_intrinsics(camerafile)

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


def json2csv(cfg,):

    jsonfile = os.path.join(cfg.dataroot, cfg.dataname, cfg.domain, f'{cfg.domain_split}.json')
    print('Reading from {} ...'.format(jsonfile))

    with open(jsonfile, 'r') as f:
        labels = json.load(f) # list


    imagedir = os.path.join(cfg.dataroot,  "speedpp", output_labels_path,'images_resized')
    if not os.path.exists(imagedir): os.makedirs(imagedir)
    print(f'Resized images will be saved to {imagedir}')

    # Open CSV file
    outcsvfile = os.path.join(cfg.dataroot, "speedpp", output_labels_path, 'labels.csv')
    csv = open(outcsvfile, 'w')
    print('New labels is Writing to {}'.format(outcsvfile))

    kp_mat = scipy.io.loadmat(PROJROOTDIR + cfg.keypts_3d_model)
    kp_arr = np.array(kp_mat['tango3Dpoints'])
    # import pdb;pdb.set_trace()

    # kp_arr = np.array([[0],[0],[0]])

    for idx in tqdm(range(len(labels))):
    # for idx in tqdm(range(10)):
        # Filename & pose labels

        q_vbs2tango = np.array(labels[idx]['q_vbs2tango_true'], dtype=np.float32)   #Q
        r_Vo2To_vbs = np.array(labels[idx]['r_Vo2To_vbs_true'], dtype=np.float32)   #Txyz
        keypoints_loc, closest_index = project_keypoints(q_vbs2tango=q_vbs2tango, r_Vo2To_vbs=r_Vo2To_vbs, cameraMatrix=cameraMatrix,
                                          distCoeffs= distCoeffs, keypoints=kp_arr)
        bbox_gt = [max(keypoints_loc[0].min(),0)/1920, max(keypoints_loc[1].min(),0)/1200,
                   min(keypoints_loc[0].max(),1920)/1920, min(keypoints_loc[1].max(),1200)/1200]


        keypoints_loc = keypoints_loc.T     #[11, 2]

    # row = [labels[idx]['filename']]  + q_vbs2tango.tolist() + r_Vo2To_vbs.tolist() + q_obj_view_r6d.tolist() + r_obj_view_dxdy.tolist()
    #     row = [labels[idx]['filename']]  + q_vbs2tango.tolist() + r_Vo2To_vbs.tolist() + keypoints_loc.tolist()
        row = ([labels[idx]['filename']]  + q_vbs2tango.tolist() + r_Vo2To_vbs.tolist())

        for kp_id in range(keypoints_loc.shape[0]):
            row = row +  keypoints_loc[kp_id].tolist()
        row += bbox_gt
        row += str(closest_index)

        row = ', '.join([str(e) for e in row])
    # Write csv
        csv.write(row + '\n')


        # # resize images _______________no need reuse every time
        # image_filename    = os.path.join(cfg.dataroot, cfg.dataname, cfg.domain, 'images', labels[idx]['filename'])
        # image    = cv2.imread(image_filename, cv2.IMREAD_COLOR)
        # # image_ori    = cv2.imread(image_filename, cv2.IMREAD_COLOR)
        # # image    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # image    = cv2.resize(image, cfg.model_input_img_size)
        # cv2.imwrite(os.path.join(imagedir, labels[idx]['filename']), image)


        # print(imagedir)
        # print(keypoints_loc.T)
        # # import pdb;pdb.set_trace()
        # cv2.circle(image, (int(keypoints_loc[0][0]/2.5), int(keypoints_loc[0][1]/2.5)), 5, (255, 0, 255), -1)
        # cv2.circle(image, (int(keypoints_loc[1][0]/2.5), int(keypoints_loc[1][1]/2.5)), 5, (255, 0, 255), -1)
        # cv2.circle(image, (int(keypoints_loc[2][0]/2.5), int(keypoints_loc[2][1]/2.5)), 5, (255, 0, 255), -1)
        # cv2.circle(image, (int(keypoints_loc[3][0]/2.5), int(keypoints_loc[3][1]/2.5)), 5, (255, 0, 255), -1)
        # cv2.circle(image, (int(keypoints_loc[4][0]/2.5), int(keypoints_loc[4][1]/2.5)), 5, (255, 0, 255), -1)
        # cv2.circle(image, (int(keypoints_loc[5][0]/2.5), int(keypoints_loc[5][1]/2.5)), 5, (255, 0, 255), -1)
        # cv2.circle(image, (int(keypoints_loc[6][0]/2.5), int(keypoints_loc[6][1]/2.5)), 5, (255, 0, 255), -1)
        # cv2.circle(image, (int(keypoints_loc[7][0]/2.5), int(keypoints_loc[7][1]/2.5)), 5, (255, 0, 255), -1)
        # cv2.circle(image, (int(keypoints_loc[8][0]/2.5), int(keypoints_loc[8][1]/2.5)), 5, (255, 0, 255), -1)
        # cv2.circle(image, (int(keypoints_loc[9][0]/2.5), int(keypoints_loc[9][1]/2.5)), 5, (255, 0, 255), -1)
        # cv2.circle(image, (int(keypoints_loc[10][0]/2.5), int(keypoints_loc[10][1]/2.5)), 5, (255, 0, 255), -1)
        # cv2.imshow('test', image)
        # cv2.waitKey(0)
        # cv2.destroyWindow()
    csv.close()
    print(f'{cfg.domain_split}_labels_rebuild_done\n')

if __name__=='__main__':

    json2csv(cfg)