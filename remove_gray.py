import os
import numpy as np
import pandas as pd
import torch
from torchvision.io import read_image

x = pd.read_csv('../../data/post_index.csv')
index = x['post_index']

new_index = []
for i in index:
    src_test = read_image('../../data/poster/' + str(i) + '.jpg')

    if np.shape(src_test) != torch.Size([3, 268, 182]):
        os.remove('../../data/poster/' + str(i) + '.jpg')
        continue
    new_index.append(i)

test = pd.DataFrame(columns=['post_index'], data=new_index)
test.to_csv('../../data/rgb_index2.csv', encoding='utf-8')
