# **
# * Copyright @2022 AI, AIRCAS. (mails.ucas.ac.cn)
#
# @author yuanzhiqiang <yuanzhiqiang19@mails.ucas.ac.cn>
#         2022/07/05

import json
import requests


class InferTest(object):

    def optim(self):
        """
        token: str 用户验证信息
        infer_name: str 推理名称
        train_id: int trainID
        model_name: str model_name
        prefix_cmd: str run command

        :return: bool 成功标志
        """

        data = {
            "src_path": "/data/server_test_data/input/0_1_inf.png",
            "save_path": "/data/server_test_data/output/airport_3.png",
        }
        url = 'http://0.0.0.0:5000/optim_ddpm/'

        r = requests.post(url, data=json.dumps(data))
        print(r)

if __name__ == "__main__":
    it = InferTest()
    it.optim()