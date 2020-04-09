import time

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.autograd import Variable
import sys
sys.path.insert(0,'.')
sys.path.append('../')
import numpy as np
np.seterr(divide='ignore',invalid='ignore')

from pkgs.dataset.total_text import TotalText
from pkgs.network.loss import TextLoss
from pkgs.network.textnet import TextNet
from pkgs.util.augmentation import BaseTransform, Augmentation
from pkgs.util.config import config as cfg, update_config, print_config
from pkgs.util.misc import AverageMeter
from pkgs.util.misc import mkdirs, to_device
from pkgs.util.option import BaseOptions
from pkgs.util.visualize import visualize_network_output
from pkgs.util.evalution import evalution
import torch.nn as nn
from tensorboardX import SummaryWriter
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def save_model(model, epoch, lr,optimizer):

    save_dir = os.path.join(cfg.save_dir, cfg.exp_name)
    if not os.path.exists(save_dir):
        mkdirs(save_dir)

    save_path = os.path.join(save_dir, 'textcohesion_best_model_9.pth')
    print('Saving to {}.'.format(save_path))
    state_dict = {
        'lr': lr,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state_dict, save_path)

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = cfg.lr * (0.8 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(model, train_loader, criterion, scheduler, optimizer, epoch):

    start = time.time()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    model.train()

    adjust_learning_rate(optimizer, epoch)

    for i, (image, train_mask,rectangular_box, tr_mask, tcl_mask , tcl_weight , dege_mask, dege_mask_weight ,up_mask, down_mask, left_mask, right_mask, meta\
            ) in enumerate(train_loader):
        data_time.update(time.time() - end)

        img, train_mask, rectangular_box,tr_mask, tcl_mask, tcl_weight,up_mask, down_mask, left_mask, right_mask = to_device(
            image, train_mask,rectangular_box, tr_mask, tcl_mask, tcl_weight , up_mask, down_mask, left_mask, right_mask)


        up_mask = Variable(up_mask.cuda()).float()
        down_mask = Variable(down_mask.cuda()).float()
        left_mask = Variable(left_mask.cuda()).float()
        right_mask = Variable(right_mask.cuda()).float()
        rectangular_box = Variable(rectangular_box.cuda()).float()
        tcl_weight = Variable(tcl_weight.cuda()).float()

        output = model(img)
        loss_tr, loss_tcl, loss_up, loss_down, loss_left, loss_right,loss_rectangular_box = \
            criterion(output, tr_mask, tcl_mask,tcl_weight, up_mask, down_mask, left_mask, right_mask, train_mask,rectangular_box)
        loss = loss_tr + loss_tcl * 3 + loss_up + loss_down + loss_left + loss_right + loss_rectangular_box

        # backward
        scheduler.step()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item())
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if i % cfg.display_freq == 0:
            print('Epoch: [ {} ][ {:03d} / {:03d} ] - Loss: {:.4f} - tr_loss: {:.4f} - tcl_loss: {:.4f} - loss_up: {:.4f} - loss_down: {:.4f} - loss_left: {:.4f} - loss_right: {:.4f} - loss_rectangular_box: {:.4f}'.format(
                epoch, i, len(train_loader), loss.item(), loss_tr.item(), loss_tcl.item(), loss_up.item(), loss_down.item(), loss_left.item(), loss_right.item(),loss_rectangular_box.item())
            )
    print('Training Loss: {}'.format(losses.avg))


def validation(model, valid_loader, criterion,scheduler, optimizer,epoch,f_score):

    model.eval()
    losses = AverageMeter()

    # evaluation index
    global_accumulative_recall = 0
    global_accumulative_precision = 0
    total_num_gt = 0
    total_num_det = 0
    for i, (image, train_mask,rectangular_box, tr_mask, tcl_mask,tcl_weight, dege_mask, dege_mask_weight,up_mask, down_mask, left_mask, right_mask,\
    meta) in enumerate(valid_loader):

        img, train_mask,rectangular_box, tr_mask, tcl_mask, tcl_weight,up_mask, down_mask, left_mask, right_mask = to_device(
            image, train_mask,rectangular_box, tr_mask, tcl_mask, tcl_weight , up_mask, down_mask, left_mask, right_mask)

        up_mask = Variable(up_mask.cuda()).float()
        down_mask = Variable(down_mask.cuda()).float()
        left_mask = Variable(left_mask.cuda()).float()
        right_mask = Variable(right_mask.cuda()).float()
        tcl_weight = Variable(tcl_weight.cuda()).float()
        rectangular_box = Variable(rectangular_box.cuda()).float()

        output = model(img)
        loss_tr, loss_tcl, loss_up, loss_down, loss_left, loss_right,loss_rectangular_box = \
            criterion(output, tr_mask, tcl_mask,tcl_weight, up_mask, down_mask, left_mask, right_mask, train_mask,rectangular_box)
        loss = loss_tr + loss_tcl * 3 + loss_up + loss_down + loss_left + loss_right + loss_rectangular_box

        losses.update(loss.item())

        # if cfg.viz and i < cfg.vis_num:
        #     visualize_network_output(output, tr_mask, tcl_mask, prefix='val_{}'.format(i))

        if i % cfg.display_freq == 0:
            print(
                'Validation:[ {:03d} / {:03d} ] - Loss: {:.4f} - loss_tr: {:.4f} - loss_tcl: {:.4f} - loss_up: {:.4f} - loss_down: {:.4f} - loss_left: {:.4f}- loss_right: {:.4f}- loss_rectangular_box: {:.4f}'.format(
                    i, len(valid_loader),loss.item(), loss_tr.item(), loss_tcl.item(), loss_up.item(),
                    loss_down.item(), loss_left.item(), loss_right.item(),loss_rectangular_box.item())
            )

        if epoch % cfg.evaluation_freq == 0 and epoch > 20:
            tcl_thresh = 0.4
            # computate the evalution index
            global_accumulative_recall, global_accumulative_precision, total_num_gt, total_num_det\
                    =evalution(img,output,meta,global_accumulative_recall,global_accumulative_precision,total_num_gt,total_num_det,tcl_thresh)


    print('Validation Loss: {}'.format(losses.avg))

    f_score_ = 0

    if epoch % cfg.evaluation_freq == 0 and epoch > 20:
        precision =  0 if total_num_det == 0  else float(global_accumulative_precision) / total_num_det
        recall = global_accumulative_recall / total_num_gt
        print('precision:  {} , recall:   {}'.format(precision,recall))

        if precision + recall > 0.1:
            f_score_ = 2 * precision * recall / (precision + recall)
            print('f_score:  {}'.format(f_score_))

        if f_score_ > f_score:
            f_score = f_score_
            save_model(model, epoch, scheduler.get_lr(), optimizer)

    # writer.add_scalar('f_score', f_score_, epoch)

    return f_score
def main():

    trainset = TotalText(
        data_root='/data/data_weijia/CTW1500_Total_ICDAR2019/',
        ignore_list='/data/data_weijia/Total_Text/ignore_list.txt',
        is_training=True,
        transform=Augmentation(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
    )

    valset = TotalText(
        data_root='/data/data_weijia/CTW1500_Total_ICDAR2019/',
        ignore_list=None,
        is_training=False,
        transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
    )

    train_loader = data.DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = data.DataLoader(valset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # Model
    model = TextNet()
    model.cuda()
    model = nn.DataParallel(model)


    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    criterion = TextLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.94)


    model_path = os.path.join(cfg.save_dir, cfg.exp_name, \
                              'textcohesion_{}_{}.pth'.format(cfg.backbone_name, cfg.checkepoch))


    # init or resume
    if cfg.resume and os.path.isfile(model_path):
        print("TextCohesion  <==> Loading checkpoint '{}' <==> Begin".format(model_path))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("TextCohesion  <==> Loading checkpoint '{}' <==> Done".format(model_path))
    else:
        start_epoch = 0

    # writer = SummaryWriter(log_dir='logs')


    print('Start training TextCohesion.')

    f_score = 0.81
    for epoch in range(start_epoch, cfg.max_epoch):
        train(model, train_loader, criterion, scheduler, optimizer, epoch )
        f_score = validation(model, val_loader, criterion,scheduler, optimizer,epoch,f_score)


    print("final f_score: ", f_score)
    print('End.')

if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    main()