import torch
import os
import torchvision.utils as tvu
import numpy as np
import cv2
from torchvision import transforms
resize = transforms.Resize([256,256])

name = 'lsun_bedroom1'
[mask, img] = torch.load("../colab_demo/{}.pth".format(name))

# print(mask)

print(img)

# tvu.save_image(mask, 'original_input.png')


mask_ = torch.from_numpy(np.zeros((3,256,256)))
img = cv2.imread("./aa.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
src = resize(torch.from_numpy(np.array(img)/255.0).float().permute(2, 0, 1))
# print(np.shape(src))
print(src)

torch.save([mask_, src], "haze.pth")