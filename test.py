
from torch.utils.data import DataLoader
import torch
import os
import os.path as osp
import scipy.io as scio
import numpy as np
import einops
from opts import parse_args
from model.Network import PGDUN
from utils import Logger, load_checkpoint, TestData, compare_ssim, compare_psnr, load_checkpoint_withoutHW
import cv2
from skimage.metrics import structural_similarity as ski_ssim

torch.set_float32_matmul_precision('highest')

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def test(args, network, logger, test_dir, epoch=1):
    network = network.eval()
    test_data = TestData(args)
    test_data_loader = DataLoader(test_data, shuffle=False, batch_size=1)
    psnr_dict,ssim_dict = {},{}
    psnr_list,ssim_list = [],[]
    dmae_list,dsim_list = [],[]
    for iter, data in enumerate(test_data_loader):
        print('iter：', iter)
        cc = 1  ## 20
        aa = 800
        gt0 = data
        with torch.no_grad():
            x0 = gt0.float().to(args.device)
            gt, out_slices, out_list, _= network(x0)
        out = out_list.cpu().numpy()
        out_slices = out_slices[-1].cpu().numpy()
        gt = gt.cpu().numpy()
        out = out.squeeze(1)
        out_slices = out_slices.squeeze(0)
        gt1 = gt.squeeze(1)

        batch, _, _ = gt1.shape
        delta = 0
        for k in range(batch):
            delta +=  abs(gt1[k,:,:]*255 - out_slices[k,:,:]*255)
        delta = delta / batch

        dmae_list.append(np.round(delta, 4))
        gt0 = np.sum(gt, 0)
        batch,_,_ = out.shape
        psnr_t = 0
        ssim_t = 0
        for k in range(batch):
            psnr_t += compare_psnr(gt0[k,:,:]*255,out[k,:,:]*255)
            ssim_t += ski_ssim(gt0[k,:,:]*255,out[k,:,:]*255,data_range=255)
        psnr = psnr_t / batch
        ssim = ssim_t / batch
        psnr_list.append(np.round(psnr,4))
        ssim_list.append(np.round(ssim,4))

        image_name = os.path.join(test_dir, "epoch_"+str(epoch)+"iter_"+str(iter)+"_image.png")
        gt_list = gt0.reshape((cc*aa, 1*aa))
        rec_list = out.reshape((cc*aa, 1*aa))*1

        result_img = np.zeros((aa*2, aa*cc))
        for i in range(cc):
            # 直接将 gt_list 对应部分填充到 result_img 上半部分
            result_img[:aa, (i * aa):((i + 1) * aa)] = gt_list[(i * aa):((i + 1) * aa), :] * 380
            # 直接将 rec_list 对应部分填充到 result_img 下半部分
            result_img[aa:, (i * aa):((i + 1) * aa)] = rec_list[(i * aa):((i + 1) * aa), :] * 380
        result_img = result_img.astype(np.float32)
        cv2.imwrite(image_name,result_img)

        cc = 10  ## 20
        aa = 800  ## 800
        image_name = os.path.join(test_dir, "epoch_" + str(epoch) + "iter_" + str(iter) + "_slices.png")
        # gt0 = np.sum(gt0, 0)
        gt_list = gt.reshape((cc * aa, 1 * aa))
        rec_list = out_slices.reshape((cc * aa, 1 * aa)) * 1
        result_img = np.zeros((aa * 2, aa * cc))
        for i in range(cc):
            # 直接将 gt_list 对应部分填充到 result_img 上半部分
            result_img[:aa, (i * aa):((i + 1) * aa)] = gt_list[(i * aa):((i + 1) * aa), :] * 255
            # 直接将 rec_list 对应部分填充到 result_img 下半部分
            result_img[aa:, (i * aa):((i + 1) * aa)] = rec_list[(i * aa):((i + 1) * aa), :] * 255
        result_img = result_img.astype(np.float32)
        cv2.imwrite(image_name, result_img)

    if logger is not None:
        logger.info("psnr_mean: {:.4f}.".format(np.mean(psnr_list)))
        logger.info("ssim_mean: {:.4f}.".format(np.mean(ssim_list)))
        logger.info("dmae_mean: {:.4f}.".format(np.mean(dmae_list)))
    return psnr_dict, ssim_dict

if __name__=="__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    args = parse_args()
    network = PGDUN(imag_size=args.size,
                    meas_size=args.meas_size,
                    img_channels=args.color_channels,
                    channels=args.channels,
                    mid_blocks=args.mid_blocks,
                    enc_blocks=args.enc_blocks,
                    dec_blocks=args.dec_blocks,
                    stages=args.stages,
                    matrix_train = args.matrix_train).to(args.device)
    log_dir = os.path.join("test_results","log")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)
    if args.test_weight_path is not None:
        logger.info('Loading pretrained model...')
        pretrained_dict = torch.load(args.test_weight_path)
        load_checkpoint(network, pretrained_dict)
        # load_checkpoint_withoutHW(network, pretrained_dict)
    else:
        raise ValueError('Please input a weight path for testing.')
    test_path = 'test_results' + '/' + '{}'.format(args.decoder_type) + '/' + '0115'
    if not os.path.exists(test_path):
        os.makedirs(test_path, exist_ok=True)
    psnr_dict, ssim_dict = test(args, network, logger, test_path, epoch=1)
    logger.info("psnr: {}.".format(psnr_dict))
    logger.info("ssim: {}.".format(ssim_dict))
    logger.info("test finish!")
