from os import listdir
from scipy import io
import numpy as np
from skimage.draw import polygon
from pkgs.util.polygon_wrapper import iou
from pkgs.util.polygon_wrapper import iod
from pkgs.util.polygon_wrapper import area_of_intersection
from pkgs.util.polygon_wrapper import area
from pkgs.util.config import config as cfg
import cv2
from pkgs.util.TextCohesion_decode import decode_batch
from pkgs.util.config import config as cfg
import os

def pixel_blok_to_box(all_Text_Instance,shape):
    coordinates = []
    for i in range(len(all_Text_Instance)):
        cnt = all_Text_Instance[i]
        image, contours, hierarchy = cv2.findContours(cnt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0]
        epsilon = 0.01 * cv2.arcLength(contours, True)
        coordinate = cv2.approxPolyDP(contours, epsilon, True)
        coordinate = coordinate.flatten()

        coordinate = coordinate / ([512, 512] * (int(len(coordinate) / 2)))
        coordinate = coordinate * (shape * (int(len(coordinate) / 2)))
        coordinate = [int(coordinate[i]) for i in range(len(coordinate))]

        coordinates.append(coordinate)
    return coordinates


def gt_reading_mod(gt_dir, gt_id):
    """This helper reads groundtruths from mat files"""
    gt_id = gt_id.split('.')[0]
    gt = io.loadmat('%s/poly_gt_%s.mat' % (gt_dir, gt_id))
    gt = gt['polygt']
    return gt


def detection_filtering(detections, groundtruths, threshold=0.5):
    for gt_id, gt in enumerate(groundtruths):
        if (gt[5] == '#') and (gt[1].shape[1] > 1):
            gt_x = [int(np.squeeze(gt[1])[i]) for i in range(len(np.squeeze(gt[1])))]
            gt_y = [int(np.squeeze(gt[3])[i]) for i in range(len(np.squeeze(gt[3])))]
            for det_id, detection in enumerate(detections):
                # detection = detection.split(', ')
                # detection = [int(detection[i]) for i in range(len(detection))]
                det_y = detection[1::2]  # get x abscissa
                det_x = detection[0::2]  # get y ordinate
                det_gt_iou = iod(det_x, det_y, gt_x, gt_y)  # get intersection area over detection area
                if det_gt_iou > threshold:
                    detections[det_id] = []

            detections[:] = [item for item in detections if item != []]
    return detections


def sigma_calculation(det_x, det_y, gt_x, gt_y):
    """
    sigma = inter_area / gt_area    >= 0.7
    """
    return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) / area(gt_x, gt_y)), 2)

def tau_calculation(det_x, det_y, gt_x, gt_y):
    """
    tau = inter_area / det_area      >= 0.6
    """
    return np.round((area_of_intersection(det_x, det_y, gt_x, gt_y) / area(det_x, det_y)), 2)


def one_to_one(local_sigma_table, local_tau_table, local_accumulative_recall,
               local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
               gt_flag, det_flag,num_gt):
    for gt_id in range(num_gt):
        qualified_sigma_candidates = np.where(local_sigma_table[gt_id, :] > cfg.tr)
        num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]
        qualified_tau_candidates = np.where(local_tau_table[gt_id, :] > cfg.tp)
        num_qualified_tau_candidates = qualified_tau_candidates[0].shape[0]


        if (num_qualified_sigma_candidates == 1) and (num_qualified_tau_candidates == 1):
            global_accumulative_recall = global_accumulative_recall + 1.0
            global_accumulative_precision = global_accumulative_precision + 1.0
            local_accumulative_recall = local_accumulative_recall + 1.0
            local_accumulative_precision = local_accumulative_precision + 1.0

            gt_flag[0, gt_id] = 1
            matched_det_id = np.where(local_sigma_table[gt_id, :] > cfg.tr)
            det_flag[0, matched_det_id] = 1
    return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag


def one_to_many(local_sigma_table, local_tau_table, local_accumulative_recall,
               local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
               gt_flag, det_flag,num_gt):
    for gt_id in range(num_gt):
        #skip the following if the groundtruth was matched
        if gt_flag[0, gt_id] > 0:
            continue

        non_zero_in_sigma = np.where(local_sigma_table[gt_id, :] > 0)
        num_non_zero_in_sigma = non_zero_in_sigma[0].shape[0]

        if num_non_zero_in_sigma >= cfg.k:
            ####search for all detections that overlaps with this groundtruth
            qualified_tau_candidates = np.where((local_tau_table[gt_id, :] >= cfg.tp) & (det_flag[0, :] == 0))
            num_qualified_tau_candidates = qualified_tau_candidates[0].shape[0]

            if num_qualified_tau_candidates == 1:
                if ((local_tau_table[gt_id, qualified_tau_candidates] >= cfg.tp) and (local_sigma_table[gt_id, qualified_tau_candidates] >= cfg.tr)):
                    #became an one-to-one case
                    global_accumulative_recall = global_accumulative_recall + 1.0
                    global_accumulative_precision = global_accumulative_precision + 1.0
                    local_accumulative_recall = local_accumulative_recall + 1.0
                    local_accumulative_precision = local_accumulative_precision + 1.0

                    gt_flag[0, gt_id] = 1
                    det_flag[0, qualified_tau_candidates] = 1
            elif (np.sum(local_sigma_table[gt_id, qualified_tau_candidates]) >= cfg.tr):
                gt_flag[0, gt_id] = 1
                det_flag[0, qualified_tau_candidates] = 1

                global_accumulative_recall = global_accumulative_recall + cfg.fsc_k
                global_accumulative_precision = global_accumulative_precision + num_qualified_tau_candidates * cfg.fsc_k

                local_accumulative_recall = local_accumulative_recall + cfg.fsc_k
                local_accumulative_precision = local_accumulative_precision + num_qualified_tau_candidates * cfg.fsc_k

    return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag


def many_to_many(local_sigma_table, local_tau_table, local_accumulative_recall,
               local_accumulative_precision, global_accumulative_recall, global_accumulative_precision,
               gt_flag, det_flag,num_det):
    for det_id in range(num_det):
        # skip the following if the detection was matched
        if det_flag[0, det_id] > 0:
            continue

        non_zero_in_tau = np.where(local_tau_table[:, det_id] > 0)
        num_non_zero_in_tau = non_zero_in_tau[0].shape[0]

        if num_non_zero_in_tau >= cfg.k:
            ####search for all detections that overlaps with this groundtruth
            qualified_sigma_candidates = np.where((local_sigma_table[:, det_id] >= cfg.tp) & (gt_flag[0, :] == 0))
            num_qualified_sigma_candidates = qualified_sigma_candidates[0].shape[0]

            if num_qualified_sigma_candidates == 1:
                if ((local_tau_table[qualified_sigma_candidates, det_id] >= cfg.tp) and (local_sigma_table[qualified_sigma_candidates, det_id] >= cfg.tr)):
                    #became an one-to-one case
                    global_accumulative_recall = global_accumulative_recall + 1.0
                    global_accumulative_precision = global_accumulative_precision + 1.0
                    local_accumulative_recall = local_accumulative_recall + 1.0
                    local_accumulative_precision = local_accumulative_precision + 1.0

                    gt_flag[0, qualified_sigma_candidates] = 1
                    det_flag[0, det_id] = 1
            elif (np.sum(local_tau_table[qualified_sigma_candidates, det_id]) >= cfg.tp):
                det_flag[0, det_id] = 1
                gt_flag[0, qualified_sigma_candidates] = 1

                global_accumulative_recall = global_accumulative_recall + num_qualified_sigma_candidates * cfg.fsc_k
                global_accumulative_precision = global_accumulative_precision + cfg.fsc_k

                local_accumulative_recall = local_accumulative_recall + num_qualified_sigma_candidates * cfg.fsc_k
                local_accumulative_precision = local_accumulative_precision + cfg.fsc_k
    return local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, gt_flag, det_flag


def get_recall_precision(groundtruths,detections,global_accumulative_recall, \
                         global_accumulative_precision, total_num_gt, total_num_det,input_id):
    local_accumulative_recall = 0
    local_accumulative_precision = 0

    dc_id = np.where(groundtruths[:, 5] == '#')
    groundtruths = np.delete(groundtruths, (dc_id), (0))
    local_sigma_table = np.zeros((groundtruths.shape[0], len(detections)))
    local_tau_table = np.zeros((groundtruths.shape[0], len(detections)))


    for gt_id, gt in enumerate(groundtruths):
        if len(detections) > 0:
            for det_id, detection in enumerate(detections):
                # detection = detection.split(', ')
                # detection = [int(detection[i]) for i in range(len(detection))]
                det_y = detection[1::2]
                det_x = detection[0::2]
                gt_x = [int(np.squeeze(gt[1])[i]) for i in range(len(np.squeeze(gt[1])))]
                gt_y = [int(np.squeeze(gt[3])[i]) for i in range(len(np.squeeze(gt[3])))]

                local_sigma_table[gt_id, det_id] = sigma_calculation(det_x, det_y, gt_x, gt_y)
                local_tau_table[gt_id, det_id] = tau_calculation(det_x, det_y, gt_x, gt_y)

    num_gt = local_sigma_table.shape[0]
    num_det = local_sigma_table.shape[1]

    total_num_gt = total_num_gt + num_gt
    total_num_det = total_num_det + num_det


    gt_flag = np.zeros((1, num_gt))
    det_flag = np.zeros((1, num_det))

    #######first check for one-to-one case##########
    local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
    gt_flag, det_flag = \
    one_to_one(local_sigma_table, local_tau_table,
                local_accumulative_recall, local_accumulative_precision,
                global_accumulative_recall, global_accumulative_precision,
                gt_flag, det_flag,num_gt)

    #######then check for one-to-many case##########
    local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
    gt_flag, det_flag = one_to_many(local_sigma_table, local_tau_table,
                                   local_accumulative_recall, local_accumulative_precision,
                                   global_accumulative_recall, global_accumulative_precision,
                                   gt_flag, det_flag,num_gt)

    #######then check for many-to-many case##########
    local_accumulative_recall, local_accumulative_precision, global_accumulative_recall, global_accumulative_precision, \
    gt_flag, det_flag = many_to_many(local_sigma_table, local_tau_table,
                                    local_accumulative_recall, local_accumulative_precision,
                                    global_accumulative_recall, global_accumulative_precision,
                                    gt_flag, det_flag,num_det)

    try:
        local_precision = local_accumulative_precision / num_det
    except ZeroDivisionError:
        local_precision = 0

    try:
        local_recall = local_accumulative_recall / num_gt
    except ZeroDivisionError:
        local_recall = 0
        
    # if local_precision <=0.5 or local_recall<=0.5:
    # print("local_precision:",local_precision)
    # print("local_recall:",local_recall)
    # print(input_id)

    return global_accumulative_recall, global_accumulative_precision,total_num_gt,total_num_det




# evaluation function
def evalution(img,output,meta,global_accumulative_recall,\
              global_accumulative_precision,total_num_gt,total_num_det,tcl_thresh):
    for idx in range(img.size(0)):
        tr_pred = output[idx, 0:2].softmax(dim=0).data.cpu().numpy()
        tcl_pred = output[idx, 2:4].softmax(dim=0).data.cpu().numpy()
        rectangular_box_pred = output[idx, 8:10].softmax(dim=0).data.cpu().numpy()
        up_pred = output[idx, 4].data.cpu().numpy()
        down_pred = output[idx, 5].data.cpu().numpy()
        left_pred = output[idx, 6].data.cpu().numpy()
        right_pred = output[idx, 7].data.cpu().numpy()


        # visualization
        img_show = img[idx].permute(1, 2, 0).cpu().numpy()
        img_show = ((img_show * cfg.stds + cfg.means) * 255).astype(np.uint8)

        # get Text instance
        all_Text_Instance = decode_batch(img_show, tr_pred[1], tcl_pred[1], up_pred, down_pred, left_pred,\
                                             right_pred,rectangular_box_pred[1], meta['image_id'][idx],tcl_thresh)

        shape = [meta['Width'][idx], meta['Height'][idx]]
        # get Text coordinates
        detections =  pixel_blok_to_box(all_Text_Instance,shape)

        input_id = meta['image_id'][idx]
        # computate the recall , precision  and f-score
        groundtruths = gt_reading_mod(cfg.gt_dir, input_id)

        detections = detection_filtering(detections, groundtruths)  # filters detections overlapping with DC area

        global_accumulative_recall, global_accumulative_precision,total_num_gt,total_num_det\
            = get_recall_precision(groundtruths,detections,global_accumulative_recall, global_accumulative_precision \
                                   , total_num_gt, total_num_det,input_id)

    return global_accumulative_recall,global_accumulative_precision,total_num_gt,total_num_det
