# **
# * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
#
# @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
#         2022/05/19

import logging
import datetime

# get current time
def get_now_time():
    now = datetime.datetime.now()
    return "{}-{}-{}-{}-{}-{}".format(now.year, now.month, now.day, now.hour, now.minute, now.second)

# logger
def get_logger(save_path=None):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)  # 设置打印级别
    formatter = logging.Formatter('%(asctime)s %(message)s')

    # 设置屏幕打印的格式
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # 设置log保存
    if save_path != None:
        fh = logging.FileHandler(save_path, encoding='utf8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".h5"])

def is_pkl(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])
