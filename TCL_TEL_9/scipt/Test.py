#!/usr/bin/env python 
# -*- coding:utf-8 -*-

import sys
sys.path.insert(0, '.')
sys.path.append('../')

import os
import torch

import torch.backends.cudnn as cudnn
import torch.utils.data as data
import numpy as np

import  torch.nn as nn
from pkgs.dataset.total_text import TotalText
from pkgs.network.textnet import TextNet
from pkgs.util.augmentation import BaseTransform
from pkgs.util.config import config as cfg, update_config, print_config
from pkgs.util.misc import to_device,write_to_txt
from pkgs.util.option import BaseOptions
from pkgs.util.visualize import visualize_network_output,visualize_detection,visualize_prediction
from pkgs.util.TextCohesion_decode import decode_batch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from pkgs.util.evalution import evalution
import cv2
from tqdm import tqdm
def load_model(model, model_path):
    print('Loading from {}'.format(model_path))
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict['model'])

def evaluate(model, test_loader):
    model.eval()

    # evaluation index
    global_accumulative_recall = 0
    global_accumulative_precision = 0
    total_num_gt = 0
    total_num_det = 0

    for tcl_thresh in range(50,65):
        tcl_thresh = tcl_thresh * 0.01

        # tcl_thresh = 0.55
        for i, (image, train_mask, rectangular_box, tr_mask, tcl_mask, tcl_weight, dege_mask, dege_mask_weight, up_mask, down_mask,
        left_mask, right_mask,meta) in tqdm(enumerate(test_loader)):

            img, train_mask, tr_mask, tcl_mask, up_mask, down_mask, left_mask, right_mask = to_device(
                image, train_mask, tr_mask, tcl_mask, up_mask, down_mask, left_mask, right_mask)

            # inference
            output = model(img)

            global_accumulative_recall, global_accumulative_precision, total_num_gt, total_num_det \
                = evalution(img, output, meta, global_accumulative_recall, global_accumulative_precision, total_num_gt,
                            total_num_det,tcl_thresh)

        print("tcl_thresh:",tcl_thresh)
        precision = global_accumulative_precision / total_num_det
        recall = global_accumulative_recall / total_num_gt
        print('precision:  {} , recall:   {}'.format(precision, recall))
        f_score_ = 2 * precision * recall / (precision + recall)
        print('f_score:  {}'.format(f_score_))

        break
        # raise NameError


def inference(model, test_loader):
    model.eval()

    # evaluation index
    tcl_thresh = 0.5
    for i, (image, train_mask, rectangular_box, tr_mask, tcl_mask, tcl_weight, dege_mask, dege_mask_weight, up_mask, down_mask,
    left_mask, right_mask,meta) in enumerate(test_loader):

        img, train_mask, rectangular_box, tr_mask, tcl_mask, tcl_weight, up_mask, down_mask, left_mask, right_mask = to_device(
            image, train_mask, rectangular_box, tr_mask, tcl_mask, tcl_weight, up_mask, down_mask, left_mask,
            right_mask)

        # inference
        output = model(img)
        # visualize_network_output(output, tr_mask, tcl_mask, up_mask, down_mask, left_mask, right_mask, dege_mask, meta)

        for idx in range(img.size(0)):
            tr_pred = output[idx, 0:2].softmax(dim=0).data.cpu().numpy()
            tcl_pred = output[idx, 2:4].softmax(dim=0).data.cpu().numpy()

            up_pred = output[idx, 4].data.cpu().numpy()
            down_pred = output[idx, 5].data.cpu().numpy()
            left_pred = output[idx, 6].data.cpu().numpy()
            right_pred = output[idx, 7].data.cpu().numpy()
            rectangular_box_pred = output[idx, 8:10].softmax(dim=0).data.cpu().numpy()
            tr_mask_ = np.array(tr_mask[idx] * train_mask[idx])
            # tr_mask_ = cv2.resize(tr_mask_, (meta['Width'][idx], meta['Height'][idx]))

            _, contours_tcl, _ = cv2.findContours(tr_mask_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

            # visualization
            img_show = img[idx].permute(1, 2, 0).cpu().numpy()
            img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)
            # img_show = cv2.resize(img_show, (meta['Width'][idx], meta['Height'][idx]))

            # visualize_detection(img_show,tr_pred[1],tcl_pred[1],up_pred,down_pred,left_pred,right_pred,meta['image_id'][idx])
            all_Text_Instance = decode_batch(img_show, tr_pred[1], tcl_pred[1], up_pred, down_pred, left_pred,\
                                             right_pred,rectangular_box_pred[1], meta['image_id'][idx],tcl_thresh)
            prefix1 = meta['image_id'][idx]
            # print(prefix1)
            name = prefix1.split('.')[0]
            # txt_path = os.path.join(cfg.txt_dir, '{}.txt'.format(name))
            # shape = [meta['Width'][idx], meta['Height'][idx]]
            # write_to_txt(all_Text_Instance, txt_path, shape)
            #
            img_path = os.path.join(cfg.vis_dir, '{}.jpg'.format(name))
            visualize_prediction(all_Text_Instance, img_path, img_show, contours_tcl)
            # break
        # break


def main():

    testset = TotalText(
        data_root='/data/data_weijia/CTW1500_Total_ICDAR2019/',
        ignore_list=None,
        is_training=False,
        transform=BaseTransform(size=cfg.input_size, mean=cfg.means, std=cfg.stds)
    )
    test_loader = data.DataLoader(testset, batch_size=2, shuffle=True, num_workers=cfg.num_workers)

    # Model
    model = TextNet()
    model_path = os.path.join(cfg.save_dir, cfg.exp_name,'textcohesion_best_model.pth')

    model = nn.DataParallel(model)
    load_model(model, model_path)

    # copy to cuda
    model = model.to(cfg.device)
    if cfg.cuda:
        cudnn.benchmark = True

    print('Start testing TextCohesion.')

    # inference(model, test_loader)

    evaluate(model, test_loader)

    print('End.')


if __name__ == "__main__":
    # parse arguments
    option = BaseOptions()
    args = option.initialize()

    update_config(cfg, args)
    print_config(cfg)

    main()