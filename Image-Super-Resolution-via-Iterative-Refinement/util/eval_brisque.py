# **
# * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
#
# @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
#         2022/06/30
import os
from brisque import BRISQUE

def eval_brisque(path, status):
    files = os.listdir(path)

    all_brisque = 0
    idx = 0
    for i, f in enumerate(files):
        if status != None and status not in f:
            continue

        sub_path = os.path.join(path, f)
        brisque = BRISQUE(sub_path, url=False).score()
        all_brisque += brisque
        idx += 1
        print("{}/{}, {}:{}".format(i, len(files), sub_path, brisque))
    return all_brisque / (idx + 1)



if __name__ == "__main__":
    path = "/data/ImageDehazing/HazeRD-PROCESS/Results_vspga_s_bn/"
    status = None
    brisque = eval_brisque(path, status)
    print("Ave: {}".format(brisque))