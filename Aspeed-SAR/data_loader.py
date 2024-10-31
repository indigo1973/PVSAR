import torch
import torch.nn as nn
import os
import pandas as pd
import random

import numpy as np
import cv2
# import albumentations as A
from pyglet import image

device = torch.device('cuda:0')

face_indexs = [[[0,1,5,4],[0,4,7,3]],
                [[1,5,6,2],[1,5,4,0]],
                [[2,1,5,6],[2,6,7,3]],
                [[3,2,6,7],[3,7,4,0]],
                [[4,0,1,5],[4,7,3,0],[4,5,6,7]],
                [[5,1,0,4],[5,1,2,6],[5,6,7,4]],
                [[6,2,1,5],[6,7,3,2],[6,7,4,5]],
                [[7,3,2,6],[7,3,0,4],[7,4,5,6]]
            ]

# album = A.Compose([
#     A.InvertImg(p=0.5),
#     A.RandomBrightnessContrast(brightness_limit=0.7, contrast_limit=0.2, p=0.8),
# ])


def read_data_speedplus(split='train', pesudo=False):
    data_dir = f'/home/indigo/dataset/speedpp/{split}/'
    if pesudo:
        csv_fname = os.path.join(data_dir, 'labels_pesudo.csv')
    else:
        csv_fname = os.path.join(data_dir, 'labels.csv')
    # csv_data = pd.read_csv(csv_fname, header=None , nrows = 2000 if is_train else 500)
    csv_data = pd.read_csv(csv_fname, header=None)

    # import pdb; pdb.set_trace()
    csv_data = csv_data.set_index(0)

    images_dir, targets= [], []
    for img_name, target in csv_data.iterrows():
        label_arr = np.array([target[1], target[2], target[3], target[4],   #Qxyzw x 4
                              target[5], target[6], target[7],              #Txyz x3
                              target[8],target[9],
                              target[10],target[11],
                              target[12],target[13],
                              target[14], target[15],
                              target[16], target[17],
                              target[18], target[19],
                              target[20], target[21],
                              target[22], target[23],
                             target[24], target[25],
                             target[26], target[27],
                             target[28], target[29],
                              target[30], target[31],target[32], target[33],
                              target[34],
                              ])

        # print(r6d_arr.T)
        images_dir.append(os.path.join(data_dir, 'images_resized', f'{img_name}'))
        # images.append(torchvision.io.read_image(os.path.join(data_dir, 'train_images_768x480_RGB' if is_train else 'validation_images_768x480_RGB', f'{img_name}')))
        targets.append(list(label_arr))
    return images_dir, torch.tensor(targets)


class SpeedPlusDataset(torch.utils.data.Dataset):
    # def __init__(self, is_train):
    def __init__(self, split='train', transform_input=None, pesudo = False):

        if split not in {'train', 'validation', 'sunlamp', 'lightbox','roe1','roe2'}:
            raise ValueError('Invalid split, has to be either \'train\', \'validation\', \'sunlamp\' or \'lightbox\'')
        self.split = split
        self.img_dir, self.labels = read_data_speedplus(split, pesudo)
        print('read ' + str(len(self.img_dir)) + (f' train examples from {split}' if split=='train' else f' val examples from {split}' ))
        self.transform_input = transform_input

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        # data = torchvision.io.read_image(self.img_dir[idx])
        # return (data.float()/255, self.labels[idx])

        sample_id = self.img_dir[idx]
        img_name = os.path.join(f'/home/indigo/dataset/speedpp/{self.split}/','images_resized', sample_id)
        pil_image = cv2.imread(img_name)


        bb = [[self.labels[idx][29],self.labels[idx][30],self.labels[idx][31],self.labels[idx][32],'Tango']]


        kpts_loc_xy = np.array(self.labels[idx][7:29]/2.5).astype(np.int32).reshape(11,2)
        # import pdb;pdb.set_trace()

        if self.split == 'train':
        # if self.split == 'aaaaaa':
            faces = face_indexs[int(self.labels[idx][33])]
            mask_name = os.path.join(f'/home/indigo/dataset/speedpp/speedplus_masks/synthetic/masks/', sample_id.split('/')[-1])
            mask_tango = cv2.imread(mask_name,cv2.IMREAD_GRAYSCALE)
            mask_tango  = cv2.resize(mask_tango, (0,0),fx = 2/5,fy =2/5)
            # import pdb;pdb.set_trace()

            for i in range(len(faces)):
                rand = random.random()
                if rand < 0.5:
                    # print(sample_id.split('/')[-1])
                    face_points = np.array([kpts_loc_xy[faces[i][0]], kpts_loc_xy[faces[i][1]],
                                            kpts_loc_xy[faces[i][2]], kpts_loc_xy[faces[i][3]]])
                    mask = np.zeros(pil_image.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [face_points], (255))
                    # import pdb;pdb.set_trace()
                    mask = cv2.bitwise_and(mask, mask_tango)

                    # inverse_image = cv2.bitwise_not(pil_image)
                    inverse_image = pil_image + random.randint(0,255)
                    # inverse_image = album(image= pil_image)["image"]

                    result = cv2.bitwise_and(inverse_image, inverse_image, mask=mask)
                    pil_image = cv2.bitwise_or(cv2.bitwise_and(pil_image, pil_image ,mask=cv2.bitwise_not(mask)),result  )
                # cv2.imshow(img_name, pil_image)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

        torch_image = self.transform_input(image=pil_image, bboxes=bb)



        # if self.split == 'train' :
        #     torch_image = self.transform_input(image=pil_image, bboxes=bb)
        # else:
        #     torch_image = self.transform_input(image=pil_image)

        # torch_image = pil_image.transforms.ToTensor()

        sample = dict()
        if self.split =='train':
            sample["image"] = torch_image['image']
            sample["q0"] = np.array(self.labels[idx][0:4]).astype(np.float32)
            sample["r0"] = np.array(self.labels[idx][4:7]).astype(np.float32)
            sample["kpts_2Dim"] = np.array(self.labels[idx][7:29]).astype(np.float32)
            sample["y"]  = sample_id.split('/')[-1]

        else:
            sample["image"] = torch_image['image']
            sample["q0"] = np.array(self.labels[idx][0:4]).astype(np.float32)
            sample["r0"] = np.array(self.labels[idx][4:7]).astype(np.float32)
            sample["kpts_2Dim"] = np.array(self.labels[idx][7:29]).astype(np.float32)
            sample["y"]  = sample_id.split('/')[-1]


        return sample

