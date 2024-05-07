# 预测图与GT图对比的误检和漏检表示
import os
import cv2
import torch.utils.data
from skimage import io
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image

# BGR
# full_to_colour = {1: (255, 255, 255), 2: (0, 0, 0), 3: (255, 0, 0), 4: (0, 0, 255)}
# full_to_colour = {1: (255, 255, 255), 2: (0, 0, 0), 3: (0, 0, 255), 4: (255, 130, 0)}  # MPVIT
# full_to_colour = {1: (255, 255, 255), 2: (0, 0, 0), 3: (0, 0, 255), 4: (0, 255, 0)}
full_to_colour = {1: (255, 255, 255), 2: (0, 0, 0), 3: (0, 0, 255), 4: (255, 0, 0)}  # 蓝、红


# full_to_colour = {1: (255, 255, 255), 2: (0, 0, 0), 3: (0, 0, 255), 4: (0, 255, 0)}

def color_label(img1, img2):
    w, h, _ = img1.shape
    # 需要重新赋值,因为图片只读
    img = np.array(img2)

    fp = np.array([255, 255, 255])
    fn = np.array([1, 1, 1])
    for i in range(0, w):
        for j in range(0, h):
            p1 = img1[i][j]
            p2 = img2[i][j]
            # false positive，根据自己需求修改对应颜色
            if ((p2 - p1) == fp).all():
                img[i][j] = [255, 0, 0]
            # false negative
            if ((p2 - p1) == fn).all():
                img[i][j] = [0, 0, 255]
    return img


pred_dir = os.path.join('Color_Map', 'pred_EORSSD')
label_dir = os.path.join('Color_Map', 'GT_EORSSD')

data_list = os.listdir(label_dir)
valid_list = []
for it in data_list:
    if it[-4:] == '.png':
        valid_list.append(it)


for it in valid_list:
    imgA_path = os.path.join(pred_dir, it)
    imgB_path = os.path.join(label_dir, it)

    preds = io.imread(imgA_path)
    # labels = io.imread(imgB_path)
    #preds = Image.open(imgA_path)
    labels = Image.open(imgB_path)
    preds= cv2.threshold(preds, 127, 255, cv2.THRESH_BINARY)[1]
    preds=Image.fromarray(preds)
    #  灰度值转rgb
    img = np.asarray(preds.convert('RGB'))
    img_B = np.asarray(labels.convert('RGB'))
    color_img = color_label(img_B, img)
    # h, w = labels.shape
    # labels_np = labels / 255
    # preds_np = cv2.threshold(preds, 127, 255, cv2.THRESH_BINARY)[1]
    # preds_np = preds / 255
    #
    # tp = np.array((labels_np == 1) & (preds_np == 1)).astype(np.int8)
    # tn = np.array((labels_np == 0) & (preds_np == 0)).astype(np.int8)
    # fp = np.array((labels_np == 0) & (preds_np == 1)).astype(np.int8)  # 背景当目标
    # fn = np.array((labels_np == 1) & (preds_np == 0)).astype(np.int8)  # 目标当背景
    #
    # img = tp * 1 + tn * 2 + fp * 3 + fn * 4
    # print(np.unique(img))

    # img_colour = torch.zeros(1, 3, h, w)
    # img_r = torch.zeros(1, h, w)
    # img_g = torch.zeros(1, h, w)
    # img_b = torch.zeros(1, h, w)
    # img = img.reshape(1, 1, h, -1)

    # for k, v in full_to_colour.items():
    #     img_r[(img == k)] = v[0]
    #     img_g[(img == k)] = v[1]
    #     img_b[(img == k)] = v[2]
    #     img_colour = torch.cat((img_r, img_g, img_b), 0)
    #     img_colour = img_colour.data.cpu().numpy()
    #     img_colour = np.transpose(img_colour, (1, 2, 0))

    # cv2.imwrite("oo/S2Looking/"+i+paths, img_colour.astype(np.uint8))
    # final_path = "1.png"
    # print(final_path)
    # cv2.imwrite(final_path, color_img.astype(np.uint8))
    filename=it[:-4]
    image_pil = Image.fromarray(np.array(color_img, dtype=np.uint8))
    image_pil.save('visualization/Color_Map/Color_EORSSD/'+filename+'.png')

