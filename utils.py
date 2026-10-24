import torch
from torch.utils.data import Dataset
import numpy as np
import logging 
import time 
import os
import cv2
import math 
import albumentations
from sklearn.metrics import mean_squared_error as MSE
import os
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
import scipy.io as sio

class TrainData(Dataset):
    def __init__(self, args):
        self.img_path = args.train_data_path  # .mat 文件路径
        self.size_h, self.size_w = args.meas_size   # 裁剪大小
        self.size_gh, self.size_gw = args.size      # 原图大小（用于放到大图里）
        num_patch = 4
        mat_names = [f for f in os.listdir(self.img_path) if f.endswith('.mat')]
        self.img_names = mat_names * num_patch
        self.transform = A.Compose([
            A.RandomCrop(height=self.size_h, width=self.size_w, p=1),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5)
        ])
    def __getitem__(self, index):
        mat_data = sio.loadmat(os.path.join(self.img_path, self.img_names[index]))
        volume = mat_data['sliced_image']  # H×W×C  volume
        transformed = self.transform(image=volume)
        gt0 = transformed['image']  # H×W×C
        gt0 = gt0.astype(np.float32)
        gt0 /= 255.0
        return gt0
    def __len__(self):
        return len(self.img_names)

class TestData(Dataset):
    def __init__(self,args):
        self.size_h, self.size_w = args.meas_size
        self.size_gh, self.size_gw = args.size
        self.data_path = args.benchmark_path
        self.data_list = os.listdir(self.data_path)
    def __getitem__(self,index):
        mat_data = sio.loadmat(os.path.join(self.data_path, self.data_list[index]))
        volume = mat_data['sliced_image']  # H×W×C  volume
        max_row_start = int(self.size_gh-self.size_h)
        max_col_start = int(self.size_gw-self.size_w)
        # 随机选择一个起始位置
        row_start = np.random.randint(0, max_row_start + 1)
        col_start = np.random.randint(0, max_col_start + 1)
        # 将小矩阵放置到大矩阵的指定位置
        gt0 = volume[row_start:row_start + self.size_h, col_start:col_start + self.size_w, :]
        gt0 = gt0.astype(np.float32)
        # depth0 = depth.astype(np.float32)
        gt0 /= 255.
        # depth0 /=255.0
        return gt0#, depth0
    def __len__(self,):
        return len(self.data_list)


class TrainData_image(Dataset):
    def __init__(self,args):
        self.img_path = args.train_data_path
        self.size_h, self.size_w = args.meas_size
        self.size_gh, self.size_gw = args.size
        num_patch = 2
        img_names = os.listdir(self.img_path)
        # print('img_names:', img_names)
        self.img_names = img_names * num_patch

    def __getitem__(self,index):
        gt0 = np.zeros([self.size_gh, self.size_gw],dtype=np.float32)
        image = cv2.imread(os.path.join(self.img_path,self.img_names[index]))
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)[:, :, 0]
        except cv2.error as e:
            print(f"处理图像失败: {e}")
        # print('image1:', image.shape)
        image = cv2.resize(image, (self.size_h, self.size_w), interpolation=cv2.INTER_LINEAR)
        image_h, image_w = image.shape[:2]
        if image_h > image_w:
            image = cv2.flip(image, 1)
            image = cv2.transpose(image)
            image_h, image_w = image.shape[:2]
        transform = albumentations.Compose([
            # albumentations.RandomCrop(height=crop_h,width=crop_w,p=1),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Resize(self.size_h,self.size_w)
        ])
        gt = transform(image=image)['image']
        max_row_start = self.size_gh - self.size_h
        max_col_start = self.size_gh - self.size_h
        # 随机选择一个起始位置
        row_start = np.random.randint(0, max_row_start + 1)
        col_start = np.random.randint(0, max_col_start + 1)
        # 将小矩阵放置到大矩阵的指定位置
        gt0[row_start:row_start + self.size_h, col_start:col_start + self.size_w] = gt
        nrows,ncols = gt.shape
        if nrows != self.size_h or ncols != self.size_w:
            raise ValueError("Training image dimension is not the desired one.")
        gt0 = gt0.astype(np.float32)
        gt0 /= 255.
        return gt0

    def __len__(self,):
        return len(self.img_names)

class TestData_image(Dataset):
    def __init__(self,args):
        self.size_h, self.size_w = args.meas_size
        self.size_gh, self.size_gw = args.size
        self.data_path = args.benchmark_path
        self.data_list = os.listdir(self.data_path)

    def __getitem__(self,index):
        gt0 = np.zeros([self.size_gh, self.size_gw], dtype=np.float32)
        pic = cv2.imread(os.path.join(self.data_path,self.data_list[index]))
        pic = cv2.cvtColor(pic,cv2.COLOR_BGR2YCrCb)[:,:,0]
        pic = cv2.resize(pic, (self.size_h, self.size_w), interpolation=cv2.INTER_LINEAR)

        max_row_start = int(self.size_gh-self.size_h)
        max_col_start = int(self.size_gw-self.size_w)

        # 随机选择一个起始位置
        row_start = np.random.randint(0, max_row_start + 1)
        col_start = np.random.randint(0, max_col_start + 1)

        # 将小矩阵放置到大矩阵的指定位置
        gt0[row_start:row_start + self.size_h, col_start:col_start + self.size_w] = pic

        gt0 = pic.astype(np.float32)
        gt0 /= 255.
        return gt0
    def __len__(self,):
        return len(self.data_list)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def compare_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))


def compare_psnr(img1, img2, shave_border=0):
    height, width = img1.shape[:2]
    img1 = img1[shave_border:height - shave_border, shave_border:width - shave_border]
    img2 = img2[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = img1 - img2
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)


def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename


def Logger(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(filename)s [line: %(lineno)s] - %(message)s')

    localtime = time.strftime('%Y_%m_%d_%H_%M_%S')
    logfile = os.path.join(log_dir,localtime+'.log')
    fh = logging.FileHandler(logfile,mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger 


def checkpoint(epoch, model, optimizer, model_out_path):
    torch.save({'pretrain_epoch':epoch,
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict()}, model_out_path)

def load_checkpoint(model, pretrained_dict, optimizer=None):
    model_dict = model.state_dict()
    pretrained_model_dict = pretrained_dict['state_dict']
    # load_dict = {k: p for k, p in pretrained_model_dict.items() if k in model_dict.keys()} # filtering parameters
    load_dict = {k: v for k, v in pretrained_model_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
    for k, p in pretrained_model_dict.items():
        if k not in model_dict:
            # k不在当前模型中，无法加载
            print(f"参数未加载（不存在于当前模型）: {k}, 预训练参数形状: {p.shape}")
        elif k in model_dict and p.shape != model_dict[k].shape:
            # k存在但形状不匹配，无法加载
            print(f"参数未加载（形状不匹配）: {k}, 预训练形状: {p.shape}, 当前模型形状: {model_dict[k].shape}")
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    if optimizer is not None:
        optimizer.load_state_dict(pretrained_dict['optimizer']) #loading pretrained optimizer when network is not changed.
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict), len(load_dict)))

def load_checkpoint_withoutHW(model, pretrained_dict, optimizer=None):
    model_dict = model.state_dict()
    pretrained_model_dict = pretrained_dict['state_dict']
    load_dict = {k: p for k, p in pretrained_model_dict.items() if k in model_dict.keys() and k != 'H' and k != 'W'} # filtering parameters  and k != 'H' and k != 'W'
    model_dict.update(load_dict)
    model.load_state_dict(model_dict)
    if optimizer is not None:
        optimizer.load_state_dict(pretrained_dict['optimizer']) #loading pretrained optimizer when network is not changed.
    print('Model parameter number: {}, Pretrained parameter number: {}, Loaded parameter number: {}'\
        .format(len(model_dict), len(pretrained_model_dict), len(load_dict)))

def compute_ortho(A,B):
    h1, w1 = A.shape
    h2, w2 = B.shape
    AT = np.transpose(A)
    BT = np.transpose(B)
    return np.sqrt(MSE(np.matmul(A,AT), np.identity(h1))) + np.sqrt(MSE(np.matmul(B,BT), np.identity(h2)))