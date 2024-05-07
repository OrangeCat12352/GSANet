import argparse
import os

import time

from torch.utils.tensorboard import SummaryWriter

import pytorch_iou
from Eval.eval import SOD_Eval
from data import get_loader

from model.MyNet import MyNet

import datetime

from utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=45, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')  # 352 384 512
parser.add_argument('--val_interval', type=int, default=1, help='validation interval ')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
opt = parser.parse_args()

# build models
model = MyNet()
model.cuda()

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
# DataSet Path
train_image_root = ''
train_gt_root = ''

train_loader = get_loader(train_image_root, train_gt_root, batchsize=opt.batchsize, size=opt.trainsize,
                          is_train=True)

train_total_step = len(train_loader)
print(train_total_step)

bce_loss = torch.nn.BCEWithLogitsLoss()
iou_loss = pytorch_iou.IOU(size_average=True)


def train_one_epoch(train_loader, model, optimizer, epoch):
    model.train()
    mean_loss = []
    for i, pack in enumerate(train_loader, start=1):

        images, gts = pack
        images = images.cuda()
        gts = gts.cuda()  # [8,1,352,352]

        sal, sal_sig = model(images)
        loss_ce = bce_loss(sal, gts)
        loss_iou = iou_loss(sal_sig, gts)
        loss = bce_loss(sal, gts) + iou_loss(sal_sig, gts)

        optimizer.zero_grad()

        loss.backward()
        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        mean_loss.append(loss.data)
        if i % 20 == 0 or i == train_total_step:
            print(
                'Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}  '
                'Loss_ce: {:.4f}, Loss_iou: {:.4f}'.
                format(epoch, opt.epoch, i, train_total_step,
                       opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss_ce.data, loss_iou.data
                       ))

    train_mean_loss = sum(mean_loss) / len(mean_loss)
    return train_mean_loss, opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch)


print("strart train")
if __name__ == '__main__':
    current_Sm = 0.0
    # 添加tensorboard
    writer = SummaryWriter("./logs_train")  # tensorboard --logdir=logs_train

    start_time = time.time()
    for epoch in range(1, opt.epoch + 1):
        adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        train_mean_loss, lr = train_one_epoch(train_loader, model, optimizer, epoch)
        # save weight
        save_path = 'result/weight/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if epoch % 1 == 0:
            torch.save(model.state_dict(), save_path + 'MyNet.pth' + '.%d' % epoch,
                       _use_new_zipfile_serialization=False)

        # evaluate
        if epoch % opt.val_interval == 0:
            Sm_info, MAE_info, maxEm_info, maxFm_info = SOD_Eval(epoch)

            # 可视化
            writer.add_scalar("train_loss", train_mean_loss, epoch)
            writer.add_scalar("val_Sm", Sm_info, epoch)
            writer.add_scalar("val_mae", MAE_info, epoch)
            writer.add_scalar("val_maxEm", maxEm_info, epoch)
            writer.add_scalar("val_maxFm", maxFm_info, epoch)

            # save_best
            if current_Sm <= Sm_info:
                torch.save(model.state_dict(), 'result/weight/' + 'MyNet_best.pth' + '.%d' % epoch,
                           _use_new_zipfile_serialization=False)
                current_Sm = Sm_info

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))
