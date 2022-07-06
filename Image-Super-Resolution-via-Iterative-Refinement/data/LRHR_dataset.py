from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import h5py, os
import numpy as np
import torch
import copy

def neibor_16_mul(num, size=32):
    a = num // size
    b = num % size
    if b >= 0.5 * size:
        return size * (a + 1)
    else:
        return size * a

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False, other_params=None):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        self.down_sample = other_params['down_sample'] if "down_sample" in other_params.keys() else None


        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'h5':
            hf = h5py.File(os.path.join(dataroot, other_params['now_hdf5']), 'r')
            self.data = hf.get("data")
            self.target = hf.get("label")
            self.data_len = self.data.shape[0]

        elif datatype == "haze_img":
            self.sr_path = Util.get_paths_from_images("{}/HR_hazy".format(dataroot))
            self.hr_path = Util.get_paths_from_images("{}/HR".format(dataroot))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

        elif datatype == "coco_img":
            self.sr_path = Util.get_paths_from_images(dataroot)
            self.hr_path = self.sr_path
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)

            self.mae_num = other_params['mae_num'] if "mae_num" in other_params.keys() else None
            self.mae_size = other_params['mae_size'] if "mae_size" in other_params.keys() else None

        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")

        elif self.datatype == 'h5':
            LR_patch = self.data[index, :, :, :]
            HR_patch = self.target[index, :, :, :]

            LR_patch = np.clip(LR_patch, 0, 1)  # we might get out of bounds due to noise
            HR_patch = np.clip(HR_patch, 0, 1)  # we might get out of bounds due to noise
            LR_patch = np.asarray(LR_patch, np.float32)
            HR_patch = np.asarray(HR_patch, np.float32)

            flip_channel = random.randint(0, 1)
            if flip_channel != 0:
                LR_patch = np.flip(LR_patch, 2)
                HR_patch = np.flip(HR_patch, 2)
            # randomly rotation
            rotation_degree = random.randint(0, 3)
            img_SR = np.rot90(LR_patch, rotation_degree, (1, 2)).transpose(1, 2, 0).copy()
            img_HR = np.rot90(HR_patch, rotation_degree, (1, 2)).transpose(1, 2, 0).copy()
            if self.need_LR:
                img_LR = img_SR.copy()

        elif self.datatype == "coco_img":
            img_HR = Image.open(self.hr_path[index]).convert("RGB")

            H, W, C = np.shape(img_HR)
            if H > self.r_res + 10 and W > self.r_res + 10:
                start_x = np.random.randint(0, H-self.r_res)
                start_y = np.random.randint(0, W-self.r_res)
                box = (start_y, start_x, start_y + self.r_res, start_x + self.r_res)
                img_HR = img_HR.crop(box)
            else:
                img_HR = img_HR.resize((self.r_res, self.r_res))

            # pretrain task
            # random_factor = np.random.randint(0, 3)
            # if random_factor == 0:
            img_SR = img_HR
            if self.mae_num is not None:
                img_SR = self._get_mae_enhancement(copy.deepcopy(np.asarray(img_HR)))
                img_SR = Image.fromarray(img_SR)

            if self.need_LR:
                img_LR = img_SR

        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.sr_path[index]).convert("RGB")

            if self.down_sample is not None:
                img_HR = self.resize(img_HR)
                img_SR = self.resize(img_SR)
                img_LR = self.resize(img_LR)

        if self.need_LR:
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))

            # print("{} img_SR:".format(self.split))
            # print(torch.mean(img_SR), torch.std(img_SR), torch.max(img_SR), torch.min(img_SR))
            # print("{} img_HR:".format(self.split))
            # print(torch.mean(img_HR), torch.std(img_HR), torch.max(img_HR), torch.min(img_HR))
            # exit()

            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index}
        else:
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))

            # print("{} img_SR:".format(self.split))
            # print(torch.mean(img_SR), torch.std(img_SR), torch.max(img_SR), torch.min(img_SR))
            # print("{} img_HR:".format(self.split))
            # print(torch.mean(img_HR), torch.std(img_HR), torch.max(img_HR), torch.min(img_HR))


            return {'HR': img_HR, 'SR': img_SR, 'Index': index}

    def resize(self, input_image):
        H, W, C = np.shape(input_image)
        resize_H, resize_W, resize_C = neibor_16_mul(int(H / self.down_sample)), neibor_16_mul(int(W / self.down_sample)), C
        out_image = input_image.resize((resize_W, resize_H))
        return out_image

    def _get_mae_enhancement(self, LR_patch):
        LR_patch.flags.writeable = True
        H_src, W_src, C_src = np.shape(LR_patch)

        for i in range(self.mae_num):

            h_src_rand = random.randint(0, H_src - self.mae_size)
            w_src_rand = random.randint(0, W_src - self.mae_size)

            LR_patch[h_src_rand:h_src_rand + self.mae_size, w_src_rand:w_src_rand + self.mae_size, :] = 0

        return LR_patch