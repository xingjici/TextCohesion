import torch
import numpy as np
import cv2
import os
from ..util.config import config as cfg
from functools import reduce
from ..util.TextCohesion_decode import filters_TCL
from .misc import filters_points,fill_hole
from pkgs.util.evalution import gt_reading_mod
from  PIL import Image
def visualize_prediction(all_Text_Instance,img_path,img_show,prefix1):


    for i in range(len(all_Text_Instance)):
        cnt = all_Text_Instance[i]
        image, contours, hierarchy = cv2.findContours(cnt, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        epsilon = 0.01 * cv2.arcLength(contours, True)
        to_txt_str = cv2.approxPolyDP(contours, epsilon, True)
        to_txt_str = [to_txt_str]
        img_show = cv2.drawContours(img_show, to_txt_str, -1, (0, 255, 0), 2)

    # img_show = cv2.drawContours(img_show, prefix1, -1, (0, 0, 255), 2)
    # cv2.imwrite(img_path, img_show)
    img_show = Image.fromarray(img_show)
    img_show.save(img_path)

def to_txt(contours, txt_path , shape, tr_targ):
    with open((txt_path), 'w') as f:
        for i in range(len(contours)):
            # to_txt_str = filters_points(contours[i])  # filters tcl detections

            # filters coordinate
            cnt = contours[i]
            epsilon = 0.01 * cv2.arcLength(cnt, True)
            to_txt_str = cv2.approxPolyDP(cnt,epsilon, True)

            cv2.drawContours(tr_targ, to_txt_str , -1, (200, 200, 150), thickness = 5)
            # font = cv.InitFont(cv.CV_FONT_HERSHEY_SCRIPT_SIMPLEX, 1, 1, 0, 3, 8)
            # cv.PutText(tr_targ, i, to_txt_str[2], font, (0, 150, 0))
            cv2.putText(tr_targ, str(i), (to_txt_str[0][0][0],to_txt_str[0][0][1]), cv2.FONT_HERSHEY_PLAIN, 2.0, (155, 155, 255), 2)

            to_txt_str = to_txt_str.flatten()

            to_txt_str = to_txt_str / ([512, 512] * (int(len(to_txt_str) / 2)))
            to_txt_str = to_txt_str * (shape * (int(len(to_txt_str) / 2)))
            to_txt_str = [int(to_txt_str[i]) for i in range(len(to_txt_str))]

            to_txt_str = str(to_txt_str)

            to_txt_str = to_txt_str.replace('[', '')
            to_txt_str = to_txt_str.replace('\n', '')
            to_txt_str = to_txt_str.replace(']', '')
            to_txt_str = to_txt_str.replace('  ', ' ')

            to_txt_str = to_txt_str.strip()
            f.writelines(to_txt_str)
            f.write('\r\n')

    return tr_targ
def visualize_network_output(output, tr_mask, tcl_mask, up_mask, down_mask, left_mask, right_mask,dege_mask,meta):

    tr_pred = output[:, :2]
    tr_score, tr_predict = tr_pred.max(dim=1)
    tcl_pred = output[:, 2:4]
    tcl_score, tcl_predict = tcl_pred.max(dim=1)

    tr_predict = tr_predict.cpu().numpy()
    tcl_predict = tcl_predict.cpu().numpy()



    tr_target = tr_mask.cpu().numpy()
    dege_mask = dege_mask.cpu().numpy()
    tcl_target = tcl_mask.cpu().numpy()

    up_pred = output[:, 4].data.cpu().numpy()
    down_pred = output[:, 5].data.cpu().numpy()
    left_pred = output[:, 6].data.cpu().numpy()
    right_pred = output[:, 7].data.cpu().numpy()


    for i in range(len(tr_pred)):

        tr_pred = (tr_predict[i] * 255).astype(np.uint8)
        tr_targ = (tr_target[i] * 255).astype(np.uint8)
        dege_mask_ = (dege_mask[i] * 255).astype(np.uint8)


        tcl_pred = (tcl_predict[i] * 255).astype(np.uint8)
        tcl_targ = (tcl_target[i] * 255).astype(np.uint8)


        prefix1 = meta['image_id'][i]

        # four direction
        up_pred_1 = up_pred[i] * 255
        down_pred_1 = down_pred[i] * 255
        left_pred_1 = left_pred[i] * 255
        right_pred_1 = right_pred[i] * 255

        up_mask_1 = up_mask[i].data.cpu().numpy() * 255
        down_mask_1 = down_mask[i].data.cpu().numpy() * 255
        left_mask_1 = left_mask[i].data.cpu().numpy() * 255
        right_mask_1 = right_mask[i].data.cpu().numpy() * 255

        # get coordinates
        image, contours, hierarchy = cv2.findContours(tr_targ, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(tr_targ, contours, -1, (125, 125, 125), thickness=3)
        name = prefix1.split('.')[0]
        txt_path = os.path.join(cfg.vis_dir, '{}.txt'.format(name))
        # tr_targ = to_txt(contours, txt_path , shape1 ,tr_targ)


        path = os.path.join(cfg.vis_dir, 'tcl_targ_{}_{}.jpg'.format(prefix1, i))
        path1 = os.path.join(cfg.vis_dir, 'up_down_left_right_pred_{}_{}.jpg'.format(prefix1, i))
        path2 = os.path.join(cfg.vis_dir, 'dege_mask_{}_{}.jpg'.format(prefix1, i))
        path3 = os.path.join(cfg.vis_dir, 'tr_targ_{}_{}.jpg'.format(prefix1, i))



        tr_show = np.concatenate([up_pred_1, down_pred_1], axis=1)
        tc_show = np.concatenate([left_pred_1, right_pred_1], axis=1)
        show = np.concatenate([tr_show, tc_show], axis=0)
        show = cv2.resize(show, (512, 512))
        cv2.imwrite(path1, show)

        cv2.imwrite(path2, dege_mask_)
        cv2.imwrite(path, tcl_targ)
        cv2.imwrite(path3,tr_targ)

        tr_show = np.concatenate([up_mask_1, down_mask_1], axis=1)
        tc_show = np.concatenate([left_mask_1, right_mask_1], axis=1)
        show = np.concatenate([tr_show, tc_show], axis=0)
        show = cv2.resize(show, (512, 512))







def visualize_detection(img_show,tr_pred,tcl_pred,up_pred,down_pred,left_pred,right_pred, image_id):
    vis_dir = '/home/weijia.wu/workspace/Sence_Text_detection/Paper-ICDAR/TCL_TEL_2/pkgs/Test_result/visualize_detection/'
    image_show = img_show.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])


    # threshold value
    tr_pred = np.array(tr_pred) > cfg.tr_confi_thresh
    tcl_pred = np.array(tcl_pred) > cfg.tcl_confi_thresh
    up_pred = np.array(up_pred) > cfg.up_confi_thresh
    down_pred = np.array(down_pred) > cfg.down_confi_thresh
    left_pred = np.array(left_pred) > cfg.left_confi_thresh
    right_pred = np.array(right_pred) > cfg.right_confi_thresh


    up_pred = cv2.cvtColor((up_pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    down_pred = cv2.cvtColor((down_pred * 255).astype(np.uint8) , cv2.COLOR_GRAY2BGR)
    left_pred = cv2.cvtColor((left_pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    right_pred = cv2.cvtColor((right_pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    tr_show = np.concatenate([up_pred, down_pred], axis=1)
    tc_show = np.concatenate([left_pred, right_pred], axis=1)
    show = np.concatenate([tr_show, tc_show], axis=0)
    show = cv2.resize(show, (512, 512))
    path1 = os.path.join(vis_dir, 'up_{}.png'.format(image_id))
    cv2.imwrite(path1, show)



    tr_pred = cv2.cvtColor((tr_pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    tcl_pred = cv2.cvtColor((tcl_pred * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


    image_show = np.concatenate([image_show, tr_pred, tcl_pred], axis=1)
    path = os.path.join(vis_dir, image_id)
    cv2.imwrite(path, image_show)
