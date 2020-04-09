import cv2
import numpy as np
from pkgs.util.config import config as cfg
import os
from PIL import Image
from skimage.draw import polygon as drawpoly
def find_contours(mask, method = None):
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype = np.uint8)
    mask = mask.copy()
    try:
        contours, _ = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL,
                                   method = method)
    except:
        _, contours, _ = cv2.findContours(mask, mode = cv2.RETR_EXTERNAL,
                                  method = method)
    return contours

def min_area_rect(cnt):
    """
    Args:
        xs: numpy ndarray with shape=(N,4). N is the number of oriented bboxes. 4 contains [x1, x2, x3, x4]
        ys: numpy ndarray with shape=(N,4), [y1, y2, y3, y4]
            Note that [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] can represent an oriented bbox.
    Return:
        the oriented rects sorrounding the box, in the format:[cx, cy, w, h, theta].
    """
    rect = cv2.minAreaRect(cnt)
    cx, cy = rect[0]
    w, h = rect[1]
    theta = rect[2]
    box = [cx, cy, w, h, theta]
    return w, h


def filters_TCL(tcl_predict):

    cnts = find_contours(tcl_predict)
    for i , cnt in enumerate(cnts):
        area = cv2.contourArea(cnt)
        max_side = max(min_area_rect(cnt))
        if area <= 50:
            # print('1',area)
            cv2.fillPoly(tcl_predict, [cnt.astype(np.int32)], color=(0,))
        elif (area < 20) and max_side <= 10:
            # print('2', area,max_side)
            cv2.fillPoly(tcl_predict, [cnt.astype(np.int32)], color=(0,))
        # print(cnt,'area:',area)
    return tcl_predict
#up_pred ： 1
#right_pred ： 2
#down_pred ：4
#left_pred ： 10
#
def visualization_(tr_pred,tcl_pred,up_pred,down_pred,left_pred,right_pred,rectangular_box_pred,image_id):
    tcl_ = np.ones(tr_pred.shape[:2], np.uint8)
    tcl_ = tcl_ - tcl_pred
    tr_pred_1 = tcl_ * tr_pred


    tcl_pred = np.expand_dims(tr_pred * tcl_pred * 1, axis=2)
    up_pred = np.expand_dims(tr_pred_1 * up_pred  * 1,axis=2)
    down_pred = np.expand_dims(tr_pred_1 * down_pred  * 1,axis=2)
    left_pred = np.expand_dims(tr_pred_1 * left_pred  * 1, axis=2)
    right_pred = np.expand_dims(tr_pred_1 * right_pred  * 1, axis=2)
    rectangular_box_pred = np.expand_dims(rectangular_box_pred, axis=2)  * 255

    tcl_pred = np.concatenate((tcl_pred*255,tcl_pred*255,tcl_pred*255),axis=-1)
    up_pred = np.concatenate((up_pred, up_pred, up_pred*255), axis=-1)            # red
    down_pred = np.concatenate((down_pred*250, down_pred*150, down_pred*150), axis=-1)    # white
    left_pred = np.concatenate((left_pred*100 , left_pred*200, left_pred*150), axis=-1)   # green
    right_pred = np.concatenate((right_pred, right_pred*255 , right_pred ), axis=-1) #cyan


    all_imgae = up_pred + down_pred + left_pred + right_pred

    path = os.path.join(cfg.vis_dir, 'up_pred_'+ image_id)
    path1 = os.path.join(cfg.vis_dir,'down_pred_' + image_id)
    path2 = os.path.join(cfg.vis_dir,'left_pred_' + image_id)
    path3 = os.path.join(cfg.vis_dir,'right_pred_' + image_id)
    path4 = os.path.join(cfg.vis_dir,'all_imgae_' +image_id)
    path5 = os.path.join(cfg.vis_dir, 'final_tcl_pred' + image_id)
    path6 = os.path.join(cfg.vis_dir, 'final_tr_pred' + image_id)
    path7 = os.path.join(cfg.vis_dir, 'final_rectangular_box_pred' + image_id)
    #
    cv2.imwrite(path, up_pred)
    cv2.imwrite(path1, down_pred)
    cv2.imwrite(path2, left_pred)
    cv2.imwrite(path3, right_pred)
    cv2.imwrite(path4, all_imgae)
    cv2.imwrite(path5, tcl_pred)

    tr_pred = tr_pred * 255
    # rectangular_box_pred = rectangular_box_pred * 255
    # cv2.imwrite(path6 , tr_pred)
    # cv2.imwrite(path7, rectangular_box_pred)


def decode_batch(img_show,tr_pred,tcl_pred,up_pred,down_pred,left_pred,right_pred,rectangular_box_pred, image_id,tcl_thresh):
    image_show = img_show.copy()
    image_show = np.ascontiguousarray(image_show[:, :, ::-1])

    tcl_pred_ = tcl_pred * tr_pred
    # threshold value
    rectangular_box_pred = (np.array(rectangular_box_pred) > cfg.tr_confi_thresh).astype(np.uint8) * 1
    tr_pred = (np.array(tr_pred) > cfg.tr_confi_thresh).astype(np.uint8) * 1
    tcl_pred = (np.array(tcl_pred_) > cfg.tcl_confi_thresh).astype(np.uint8) * 1
    # tcl_pred = (np.array(tcl_pred) > tcl_thresh).astype(np.uint8) * 1
    up_pred = (np.array(up_pred) > cfg.up_confi_thresh).astype(np.uint8) * 1
    down_pred = (np.array(down_pred) > cfg.down_confi_thresh).astype(np.uint8) * 1
    left_pred = (np.array(left_pred) > cfg.left_confi_thresh).astype(np.uint8) * 1
    right_pred = (np.array(right_pred) > cfg.right_confi_thresh).astype(np.uint8) * 1

    tcl_pred = filters_TCL(tcl_pred)  # filters tcl detections
    tcl_pred = tcl_pred * rectangular_box_pred
    confidence = tcl_pred_ * tcl_pred
    all_Text_Instance = decode_image_by_join_1(tr_pred, tcl_pred, up_pred, down_pred, left_pred, right_pred,confidence,tcl_thresh)
    # visualizing the result
    # visualization_(tr_pred,tcl_pred,up_pred,down_pred,left_pred,right_pred,rectangular_box_pred,image_id)

    # path1 = os.path.join(cfg.vis_dir, 'tcl_filter'+image_id)
    # path3 = os.path.join(cfg.vis_dir, 'rectangular_box_pred' + image_id)
    # path2 = os.path.join(cfg.vis_dir, 'tr_'+image_id)
    # tr_pred = (tr_pred * 255).astype(np.uint8)
    # cv2.imwrite(path2, tr_pred)
    # tcl_pred = (tcl_pred * 255).astype(np.uint8)
    # cv2.imwrite(path1, tcl_pred)
    # rectangular_box_pred = (rectangular_box_pred * 255).astype(np.uint8)
    # cv2.imwrite(path3, rectangular_box_pred)

    return all_Text_Instance


def get_candidate_area(tcl_textinstance,tr_pred):

    cnts = find_contours(tr_pred)
    area = []
    for i,point in enumerate(cnts):
        tr_textinstance = np.zeros(tr_pred.shape[:2], np.uint8)
        cv2.fillPoly(tr_textinstance, [point], color=(1,))
        cross_area = tr_textinstance * tcl_textinstance
        area.append(np.sum(cross_area))

    max_indx = area.index(max(area))
    max_area = cnts[max_indx]
    tr_textinstance = np.zeros(tr_pred.shape[:2], np.uint8)
    cv2.fillPoly(tr_textinstance, [max_area], color=(1,))
    return tr_textinstance
def get_right_direction(tcl_textinstance,direction):

    cnts = find_contours(direction)
    # area = np.zeros(tcl_textinstance.shape[:2], np.uint8)
    area = []
    for i,point in enumerate(cnts):
        direction_textinstance = np.zeros(tcl_textinstance.shape[:2], np.uint8)
        cv2.fillPoly(direction_textinstance, [point], color=(1,))
        cross_area = direction_textinstance * tcl_textinstance
        area.append(np.sum(cross_area))
        # if cross_area.sum() > 5:
        #     area = area | direction_textinstance
    direction_textinstance = np.zeros(tcl_textinstance.shape[:2], np.uint8)
    try:
        max_indx = area.index(max(area))
    except:
        return direction_textinstance
    max_area = cnts[max_indx]

    cv2.fillPoly(direction_textinstance, [max_area], color=(1,))
    if (direction_textinstance*tcl_textinstance).sum() > 5:
        return direction_textinstance
    else:
        return np.zeros(tcl_textinstance.shape[:2], np.uint8)

def decode_image_by_join(tr_pred,tcl_pred,up_pred,down_pred,left_pred,right_pred):
    h, w = np.shape(tr_pred)
    all_Text_Instance = []

    tcl_ = np.ones(tr_pred.shape[:2], np.uint8)
    tcl_ = tcl_ - tcl_pred
    tr_pred_ = tcl_ * tr_pred

    # up_pred ： 1
    # right_pred ： 2
    # down_pred ：4
    # left_pred ： 10
    up_pred = up_pred * tr_pred_
    down_pred = tr_pred_ * down_pred
    left_pred = tr_pred_ * left_pred
    right_pred = tr_pred_ * right_pred

    tcl_pred = tr_pred * tcl_pred
    cnts = find_contours(tcl_pred)

    for i , point in enumerate(cnts):

        tcl_Text_Instance = np.zeros(tcl_pred.shape[:2], np.uint8)
        cv2.fillPoly(tcl_Text_Instance, [point], color=(1,))
        path = os.path.join(cfg.vis_dir,str(i) + '_tcl.jpg')
        # cv2.imwrite(path, tcl_Text_Instance * 255)

        # Tr_Text_instance = get_candidate_area(tcl_Text_Instance,tr_pred)

        tcl_area = cv2.contourArea(point)
        tcl_wide = max(min_area_rect(point))
        if tcl_area<=2 or tcl_wide<=1:
            continue



        # up_right = Tr_Text_instance * up_pred
        # down_right = Tr_Text_instance * down_pred
        # right_right = Tr_Text_instance * right_pred
        # left_right = Tr_Text_instance * left_pred

        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.dilate(tcl_Text_Instance,kernel,iterations= 1)

        up_right = get_right_direction(erosion,up_pred)
        down_right = get_right_direction(erosion, down_pred)
        right_right = get_right_direction(erosion, right_pred)
        left_right = get_right_direction(erosion, left_pred)

        pixel_mask = tcl_Text_Instance == 1
        points = np.where(pixel_mask)

        final_Text_Insatnce_1 = link_up_down_direction(tcl_Text_Instance, up_right,down_right, points)
        final_Text_Insatnce_2 = link_left_right_direction(tcl_Text_Instance, left_right, right_right, points)
        final_Text_Insatnce = final_Text_Insatnce_1 | final_Text_Insatnce_2
        all_Text_Instance.append(final_Text_Insatnce)
        # path1 = os.path.join(cfg.vis_dir, str(i) + '_tcl.jpg')
        # path2 = os.path.join(cfg.vis_dir, str(i) + 'erosion_tcl.jpg')
        # path3 = os.path.join(cfg.vis_dir, str(i) + 'up_pred.jpg')
        # path4 = os.path.join(cfg.vis_dir, str(i) + 'final_textinsatnce.jpg')



        # cv2.imwrite(path2,erosion*255)
        # cv2.imwrite(path1, tcl_Text_Instance * 255)
        # cv2.imwrite(path3,  up_right* 255)
        # cv2.imwrite(path4, final_Text_Insatnce * 255)

        # raise NameError

    # add all text instance
    # Texts = np.zeros(tr_pred.shape[:2], np.uint8)
    # for text_instance in all_Text_Instance:
    #     Texts +=text_instance
    #
    # path = os.path.join(cfg.vis_dir,'final_Text_Insatnce.jpg')
    # cv2.imwrite(path, Texts * 255)

    return all_Text_Instance

def link_up_down_direction(tcl_Text_Instance,up_right,down_right,points):
    y_min, y_max, x_min, x_max = points[0].min(), points[0].max(), points[1].min(), points[1].max()
    pixel_mask = up_right == 1
    points_up = np.where(pixel_mask)

    pixel_mask = down_right == 1
    points_down = np.where(pixel_mask)

    up_right_ = up_right.copy()
    up_right_[:,0:x_min] = 0
    up_right_[:,x_max:512] = 0

    down_right_ = down_right.copy()
    down_right_[:,0:x_min] = 0
    down_right_[:,x_max:512] = 0

    final_Text_Insatnce = tcl_Text_Instance | up_right_ | down_right_
    for x in range(x_min,x_max,3):
        #up
        try:
            tcl_y_index =  np.where(points[1] == x)[0][0]
            up_y_index = np.where(points_up[1] == x)[0][0]
            tcl_y = points[0][tcl_y_index]
            up_y = points_up[0][up_y_index]
        except:
            continue
        x2 = x + 3
        try:
            tcl_y_index2 =  np.where(points[1] == x2)[0][0]
            up_y_index2 = np.where(points_up[1] == x2)[0][0]
            tcl_y2 = points[0][tcl_y_index2]
            up_y2 = points_up[0][up_y_index2]
        except:
            continue


        p1 ,p2,p3,p4 = [x,tcl_y] , [x2,tcl_y2] , [x2,up_y2] , [x,up_y]
        p1, p2, p3, p4 = restrict(p1, p2, p3, p4)
        polygon = np.stack([p1, p2, p3, p4])
        fill_polygon(final_Text_Insatnce, polygon, value=1)

        # down
        try:
            tcl_y_index =  np.where(points[1] == x)[0][0]
            up_y_index = np.where(points_down[1] == x)[0][0]
            tcl_y = points[0][tcl_y_index]
            up_y = points_down[0][up_y_index]
        except:
            continue
        x2 = x + 3
        try:
            tcl_y_index2 =  np.where(points[1] == x2)[0][0]
            up_y_index2 = np.where(points_down[1] == x2)[0][0]
            tcl_y2 = points[0][tcl_y_index2]
            up_y2 = points_down[0][up_y_index2]
        except:
            continue


        p1 ,p2,p3,p4 = [x,tcl_y] , [x2,tcl_y2] , [x2,up_y2] , [x,up_y]
        p1, p2, p3, p4 = restrict(p1, p2, p3, p4)
        polygon = np.stack([p1, p2, p3, p4])
        fill_polygon(final_Text_Insatnce, polygon, value=1)

    return final_Text_Insatnce


def link_left_right_direction(tcl_Text_Instance,left_right, right_right, points):
    y_min, y_max, x_min, x_max = points[0].min(), points[0].max(), points[1].min(), points[1].max()
    pixel_mask = left_right == 1
    points_left = np.where(pixel_mask)

    pixel_mask = right_right == 1
    points_right = np.where(pixel_mask)

    up_right_ = left_right.copy()
    up_right_[0:y_min, :] = 0
    up_right_[y_max:512, :] = 0

    down_right_ = right_right.copy()
    down_right_[0:y_min, :] = 0
    down_right_[y_max:512, :] = 0

    final_Text_Insatnce = tcl_Text_Instance | up_right_ | down_right_
    for y in range(y_min, y_max, 3):
        # left
        try:
            tcl_x_index = np.where(points[0] == y)[0][0]
            up_x_index = np.where(points_left[0] == y)[0][0]
            tcl_x = points[1][tcl_x_index]
            left_x = points_left[1][up_x_index]
        except:
            continue

        y2 = y + 3
        try:
            tcl_x_index2 = np.where(points[0] == y2)[0][0]
            up_x_index2 = np.where(points_left[0] == y2)[0][0]
            tcl_x2 = points[1][tcl_x_index2]
            left_x2 = points_left[1][up_x_index2]
        except:
            continue

        p1, p2, p3, p4 = [tcl_x, y], [tcl_x2, y2], [left_x2, y2], [left_x, y]
        p1, p2, p3, p4 = restrict(p1, p2, p3, p4)
        polygon = np.stack([p1, p2, p3, p4])
        fill_polygon(final_Text_Insatnce, polygon, value=1)

        # right
        try:

            tcl_x_index = np.where(points[0] == y)[0][0]
            right_x_index = np.where(points_right[0] == y)[0][0]
            tcl_x = points[1][tcl_x_index]
            right_x = points_right[1][right_x_index]
        except:
            continue
        y2 = y + 3
        try:
            tcl_x_index2 = np.where(points[0] == y2)[0][0]
            right_x_index2 = np.where(points_right[0] == y2)[0][0]
            tcl_x2 = points[1][tcl_x_index2]
            right_x2 = points_right[1][right_x_index2]
        except:
            continue

        p1, p2, p3, p4 = [tcl_x, y], [tcl_x2, y2], [right_x2, y2], [right_x, y]
        p1, p2, p3, p4 = restrict(p1, p2, p3, p4)
        polygon = np.stack([p1, p2, p3, p4])
        fill_polygon(final_Text_Insatnce, polygon, value=1)

    return final_Text_Insatnce

def decode_image_by_join_1(tr_pred,tcl_pred,up_pred,down_pred,left_pred,right_pred,confidence,tcl_thresh):
    all_Text_Instance = []

    tcl_ = np.ones(tr_pred.shape[:2], np.uint8)
    tcl_ = tcl_ - tcl_pred
    tr_pred_ = tcl_ * tr_pred

    # up_pred ： 1
    # right_pred ： 2
    # down_pred ：4
    # left_pred ： 10
    up_pred = up_pred * tr_pred_
    down_pred = tr_pred_ * down_pred
    left_pred = tr_pred_ * left_pred
    right_pred = tr_pred_ * right_pred

    tcl_pred = tr_pred * tcl_pred
    cnts = find_contours(tcl_pred)

    for i , point in enumerate(cnts):

        tcl_Text_Instance = np.zeros(tcl_pred.shape[:2], np.uint8)
        cv2.fillPoly(tcl_Text_Instance, [point], color=(1,))
        tcl_area = cv2.contourArea(point)
        tcl_wide = max(min_area_rect(point))
        if tcl_area<=2 or tcl_wide<=1:
            continue


        # confidence
        confidence_ = tcl_Text_Instance * confidence
        confidence_ = np.around(np.sum(confidence_) / np.sum(tcl_Text_Instance), decimals=2)
        if confidence_<tcl_thresh:
            continue

        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.dilate(tcl_Text_Instance,kernel,iterations= 1)

        up_right = get_right_direction(erosion,up_pred)
        down_right = get_right_direction(erosion, down_pred)
        right_right = get_right_direction(erosion, right_pred)
        left_right = get_right_direction(erosion, left_pred)

        pixel_mask = tcl_Text_Instance == 1
        points = np.where(pixel_mask)
        final_Text_Insatnce_1 = link_up_down_direction_1(tcl_Text_Instance, up_right,down_right, points)
        final_Text_Insatnce_2 = link_left_right_direction_1(tcl_Text_Instance, left_right, right_right, points)
        final_Text_Insatnce = final_Text_Insatnce_1 | final_Text_Insatnce_2
        all_Text_Instance.append(final_Text_Insatnce)
    return all_Text_Instance

def link_up_down_direction_1(tcl_Text_Instance,up_right,down_right,points):
    y_min, y_max, x_min, x_max = points[0].min(), points[0].max(), points[1].min(), points[1].max()
    pixel_mask = up_right == 1
    points_up = np.where(pixel_mask)

    pixel_mask = down_right == 1
    points_down = np.where(pixel_mask)

    final_Text_Insatnce = tcl_Text_Instance
    for x in range(x_min,x_max,3):
        #up
        try:
            tcl_y_index =  np.where(points[1] == x)[0]
            up_y_index = np.where(points_up[1] == x)[0]
            miny_y_list = []
            for min_y in tcl_y_index:
                tcl_y = points[0][min_y]
                miny_y_list.append(tcl_y)
            tcl_y = min(miny_y_list)

            miny_y_list = []
            for min_y in up_y_index:
                up_y = points_up[0][min_y]
                miny_y_list.append(up_y)
            up_y = min(miny_y_list)
        except:
            continue

        # if abs(tcl_y - up_y) > tcl_thresh:
        #     continue
        if tcl_y < up_y:
            continue
        x2 = x + 3
        try:
            tcl_y_index2 =  np.where(points[1] == x2)[0]
            up_y_index2 = np.where(points_up[1] == x2)[0]

            miny_y_list = []
            for min_y in tcl_y_index2:
                tcl_y2 = points[0][min_y]
                miny_y_list.append(tcl_y2)
            tcl_y2 = min(miny_y_list)

            miny_y_list = []
            for min_y in up_y_index2:
                up_y2 = points_up[0][min_y]
                miny_y_list.append(up_y2)
            up_y2 = min(miny_y_list)

        except:
            continue


        p1 ,p2,p3,p4 = [x,tcl_y] , [x2,tcl_y2] , [x2,up_y2] , [x,up_y]
        p1, p2, p3, p4 = restrict(p1, p2, p3, p4)
        polygon = np.stack([p1, p2, p3, p4])
        fill_polygon(final_Text_Insatnce, polygon, value=1)

        # down
        try:
            tcl_y_index =  np.where(points[1] == x)[0]
            down_y_index = np.where(points_down[1] == x)[0]

            miny_y_list = []
            for min_y in tcl_y_index:
                tcl_y = points[0][min_y]
                miny_y_list.append(tcl_y)
            tcl_y = max(miny_y_list)

            miny_y_list = []
            for min_y in down_y_index:
                up_y = points_down[0][min_y]
                miny_y_list.append(up_y)
            up_y = max(miny_y_list)

        except:
            continue
        # if abs(tcl_y - up_y) > tcl_thresh:
        #     continue
        if tcl_y > up_y:
            continue
        x2 = x + 3
        try:
            tcl_y_index2 =  np.where(points[1] == x2)[0]
            down_y_index2 = np.where(points_down[1] == x2)[0]

            miny_y_list = []
            for min_y in tcl_y_index2:
                tcl_y2 = points[0][min_y]
                miny_y_list.append(tcl_y2)
            tcl_y2 = max(miny_y_list)

            miny_y_list = []
            for min_y in down_y_index2:
                up_y2 = points_down[0][min_y]
                miny_y_list.append(up_y2)
            up_y2 = max(miny_y_list)

            # tcl_y2 = points[0][tcl_y_index2]
            # up_y2 = points_down[0][up_y_index2]
        except:
            continue


        p1 ,p2,p3,p4 = [x,tcl_y] , [x2,tcl_y2] , [x2,up_y2] , [x,up_y]
        p1, p2, p3, p4 = restrict(p1, p2, p3, p4)
        polygon = np.stack([p1, p2, p3, p4])
        fill_polygon(final_Text_Insatnce, polygon, value=1)

    return final_Text_Insatnce

def link_left_right_direction_1(tcl_Text_Instance,left_right, right_right, points):
    y_min, y_max, x_min, x_max = points[0].min(), points[0].max(), points[1].min(), points[1].max()
    pixel_mask = left_right == 1
    points_left = np.where(pixel_mask)

    pixel_mask = right_right == 1
    points_right = np.where(pixel_mask)

    final_Text_Insatnce = tcl_Text_Instance
    for y in range(y_min, y_max, 3):
        # left
        try:
            tcl_x_index = np.where(points[0] == y)[0]
            left_x_index = np.where(points_left[0] == y)[0]

            miny_x_list = []
            for min_x in tcl_x_index:
                tcl_x = points[1][min_x]
                miny_x_list.append(tcl_x)
            tcl_x = min(miny_x_list)

            miny_x_list = []
            for min_x in left_x_index:
                left_x = points_left[1][min_x]
                miny_x_list.append(left_x)
            left_x = min(miny_x_list)
        except:
            continue
        # if abs(tcl_x - left_x) > tcl_thresh:
        #     continue
        if tcl_x < left_x:
            continue
        y2 = y + 3
        try:
            tcl_x_index2 = np.where(points[0] == y2)[0]
            left_x_index2 = np.where(points_left[0] == y2)[0]

            miny_x_list = []
            for min_x in tcl_x_index2:
                tcl_x2 = points[1][min_x]
                miny_x_list.append(tcl_x2)
            tcl_x2 = min(miny_x_list)

            miny_x_list = []
            for min_x in left_x_index2:
                left_x2 = points_left[1][min_x]
                miny_x_list.append(left_x2)
            left_x2 = min(miny_x_list)

            # tcl_x2 = points[1][tcl_x_index2]
            # left_x2 = points_left[1][left_x_index2]
        except:
            continue

        p1, p2, p3, p4 = [tcl_x, y], [tcl_x2, y2], [left_x2, y2], [left_x, y]
        p1, p2, p3, p4 = restrict(p1, p2, p3, p4)
        polygon = np.stack([p1, p2, p3, p4])
        fill_polygon(final_Text_Insatnce, polygon, value=1)

        # right
        try:

            tcl_x_index = np.where(points[0] == y)[0]
            right_x_index = np.where(points_right[0] == y)[0]

            miny_x_list = []
            for max_x in tcl_x_index:
                tcl_x = points[1][max_x]
                miny_x_list.append(tcl_x)
            tcl_x = max(miny_x_list)

            miny_x_list = []
            for max_x in right_x_index:
                right_x = points_right[1][max_x]
                miny_x_list.append(right_x)
            right_x = max(miny_x_list)
        except:
            continue
        # if abs(tcl_x - right_x) > tcl_thresh:
        #     continue
        if tcl_x > right_x:
            continue
        y2 = y + 3
        try:
            tcl_x_index2 = np.where(points[0] == y2)[0]
            right_x_index2 = np.where(points_right[0] == y2)[0]

            miny_x_list = []
            for max_x in tcl_x_index2:
                tcl_x2 = points[1][max_x]
                miny_x_list.append(tcl_x2)
            tcl_x2 = max(miny_x_list)

            miny_x_list = []
            for max_x in right_x_index2:
                right_x2 = points_right[1][max_x]
                miny_x_list.append(right_x2)
            right_x2 = max(miny_x_list)

        except:
            continue

        p1, p2, p3, p4 = [tcl_x, y], [tcl_x2, y2], [right_x2, y2], [right_x, y]
        p1, p2, p3, p4 = restrict(p1, p2, p3, p4)
        polygon = np.stack([p1, p2, p3, p4])
        fill_polygon(final_Text_Insatnce, polygon, value=1)

    return final_Text_Insatnce

def fill_polygon(mask, polygon, value):
    """
    fill polygon in the mask with value
    :param mask: input mask
    :param polygon: polygon to draw
    :param value: fill value
    """
    rr, cc = drawpoly(polygon[:, 1], polygon[:, 0])
    mask[rr, cc] = value



def get_neighbours_8(x, y):
    """
    Get 8 neighbours of point(x, y)
    """
    return [(x - 1, y - 1), (x, y - 1), (x + 1, y - 1), \
        (x - 1, y),                 (x + 1, y),  \
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)]
def restrict(p1, p2, p3, p4):
    x, y = p1
    x = min(x, 512)
    # print(p1)
    y = min(y, 512)
    p1 = [x, y]

    x, y = p2
    x = min(x, 512)
    y = min(y, 512)
    p2 = [x, y]

    x, y = p3
    x = min(x, 512)
    y = min(y, 512)
    p3 = [x, y]

    x, y = p4
    x = min(x, 512)
    y = min(y, 512)
    p4 = [x, y]

    return p1, p2, p3, p4