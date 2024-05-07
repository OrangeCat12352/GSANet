import os
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F

import numpy as np
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import imageio

from model.MyNet import MyNet
from data import test_dataset

import os
# pip install pysodmetrics
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure


def SOD_Eval(epoch):
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=352, help='testing size')
    parser.add_argument('--dataset_path', type=str, default=r'D:\DataSets\SOD\EORSSD_aug', help='DataSet Path')
    parser.add_argument('--dataName', type=str, default='EORSSD', help='Test DataSet')
    opt = parser.parse_args()

    model = MyNet()

    model.load_state_dict(torch.load('./result/weight/MyNet.pth.{}'.format(epoch)))

    model.cuda()
    model.eval()

    save_path = 'Eval/pred/{}/'.format(opt.dataName)
    # save_path = 'eval_utils/pred/MyNet/ORS-4199/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = opt.dataset_path + '/' + 'test/image/'
    gt_root = opt.dataset_path + '/' + 'test/GT/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        res, _ = model(image)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res = res * 255
        res = res.astype(np.uint8)
        imageio.imsave(save_path + name, res)

    # Eval
    method = 'MyNet'
    data_name = opt.dataName
    mask_root = gt_root
    pred_root = save_path
    mask_name_list = sorted(os.listdir(mask_root))
    FM = Fmeasure()
    # WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for mask_name in tqdm(mask_name_list, total=len(mask_name_list)):
        mask_path = os.path.join(mask_root, mask_name)
        pred_path = os.path.join(pred_root, mask_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        # WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    # wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        # "Smeasure": sm,
        # "wFmeasure": wfm,
        # "MAE": mae,
        # "adpEm": em["adp"],
        # "meanEm": em["curve"].mean(),
        # "maxEm": em["curve"].max(),
        # "adpFm": fm["adp"],
        # "meanFm": fm["curve"].mean(),
        # "maxFm": fm["curve"].max(),

        "Smeasure": "{:.4f}".format(sm),
        "MAE": "{:.4f}".format(mae),
        "adpEm": "{:.4f}".format(em["adp"]),
        "meanEm": "{:.4f}".format(em["curve"].mean()),
        "maxEm": "{:.4f}".format(em["curve"].max()),
        "adpFm": "{:.4f}".format(fm["adp"]),
        "meanFm": "{:.4f}".format(fm["curve"].mean()),
        "maxFm": "{:.4f}".format(fm["curve"].max()),
    }

    print(results)
    file = open("Eval/eval_results.txt", "a")
    file.write(method + ' ' + data_name + ' ' + str(results) + '\n')

    return sm, mae, em["curve"].max(), fm["curve"].max()


if __name__ == '__main__':
    SOD_Eval(1)
