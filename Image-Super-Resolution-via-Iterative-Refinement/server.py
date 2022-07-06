import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from brisque import BRISQUE
import cv2, shutil
from flask import Flask, request, json

app = Flask(__name__)  # Flask 初始化

def get_stand_return(flag, message):
    if (isinstance(message, dict) and isinstance(message.get('code'), int) and message.get('code') != 0):
        return_json = message
    else:
        return_json = {
            'code': 0 if flag else 400,
            "message": message if not flag else "Success",
            "data": message if flag else None
        }
    return return_json


def calc_mean_rgb(img):
    H, W, C = np.shape(img)
    img = np.reshape(img, (H*W, C))
    return np.mean(img, axis=0)

def fix_img(img, img_ref):
    sr_R, sr_G, sr_B = calc_mean_rgb(img)
    hr_R, hr_G, hr_B = calc_mean_rgb(img_ref)

    R,G,B = sr_R-hr_R, sr_G-hr_G, sr_B-hr_B
    R = np.array(img[:, :, 0]) - R
    G = np.array(img[:, :, 1]) - G
    B = np.array(img[:, :, 2]) - B

    R = np.expand_dims(R, axis=-1)
    G = np.expand_dims(G, axis=-1)
    B = np.expand_dims(B, axis=-1)

    return np.array(np.concatenate((R,G,B), axis=-1), dtype=np.uint8)

@app.route("/optim_ddpm/", methods=["POST"])
def infer():
    request_data = json.loads(request.data.decode('utf-8'))

    os.makedirs("tmp", exist_ok=True)
    os.makedirs("tmp/input", exist_ok=True)
    os.makedirs("tmp/output", exist_ok=True)
    shutil.copy(request_data['src_path'], "tmp/input/tmp.jpg")
    opt['path']['results'] = "tmp/output"
    opt['datasets']['val']['dataroot'] = "tmp/input"


    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)

    logger.info('Begin Model Inference.')
    current_step = 0
    idx = 0

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    for _, val_data in enumerate(val_loader):

        idx += 1

        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals(need_LR=False)

        hr_img = Metrics.tensor2img(visuals['HR'])  # uint8
        fake_img = Metrics.tensor2img(visuals['INF'])  # uint8

        sr_img_mode = 'grid'
        if sr_img_mode == 'single':
            # single img series
            sr_img = visuals['SR']  # uint8
            sample_num = sr_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(sr_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            sr_img = Metrics.tensor2img(visuals['SR'])  # uint8

            h, w, c = np.shape(hr_img)
            sr_img[-h - 2:-2, -w - 2:-2, :] = hr_img

            Metrics.save_img(
                sr_img, '{}/{}_{}_sr_process.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                Metrics.tensor2img(visuals['SR'][-1]), '{}/{}_{}_sr.png'.format(result_path, current_step, idx))

        Metrics.save_img(
            hr_img, '{}/{}_{}_hr.png'.format(result_path, current_step, idx))
        Metrics.save_img(
            fake_img, '{}/{}_{}_inf.png'.format(result_path, current_step, idx))

    shutil.copy("tmp/output/{}".format([i for i in os.listdir("tmp/output/") if "sr.png" in i][0]), request_data['save_path'])
    return get_stand_return(True, "save to: {}".format(request_data['save_path']))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/rs_sr_server.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-enable_wandb', action='store_true')
    parser.add_argument('-log_infer', action='store_true')
    parser.add_argument('-color_fix', default=False)
    
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    

    app_host = '0.0.0.0'
    app_port = 5000

    app.run(host=app_host, port=app_port)



