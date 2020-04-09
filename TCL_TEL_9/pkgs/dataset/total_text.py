import copy
import cv2
import scipy.io as io
import numpy as np
import os
import sys

sys.path.insert(0, '.')
sys.path.append('../')
import torch
import warnings

warnings.filterwarnings("ignore")
from skimage.draw import polygon as drawpoly
from pkgs.util.misc import find_bottom, find_long_edges, split_edge_seqence, norm2
from pkgs.util.config import config as cfg
from pkgs.util.TextCohesion_decode import find_contours
import torch.utils.data as data
from pkgs.dataset.data_util import pil_load_img, restrict, read_lines, remove_all


class TextInstance(object):
    def __init__(self, points, orient, text, point_rectangle, pointss):
        self.orient = orient
        self.text = text
        self.point_rectangle = np.array(point_rectangle)
        self.pointss = np.array(pointss)

        # self.points = []

        # remove point if area is almost unchanged after removing
        # ori_area = cv2.contourArea(points)
        # for p in range(len(points)):
        #     index = list(range(len(points)))
        #     index.remove(p)
        #     area = cv2.contourArea(points[index])
        #     if np.abs(ori_area - area) / ori_area > 0.007:
        #         self.points.append(points[p])
        self.points = np.array(points)

    def find_bottom_and_sideline(self):
        self.bottoms = find_bottom(self.points)  # find two bottoms of this Text
        self.e1, self.e2 = find_long_edges(self.points, self.bottoms)  # find two long edge sequence

    def disk_cover(self, n_disk=40):
        """
        cover text region with several disks
        :param n_disk: number of disks
        :return:
        """
        inner_points1 = split_edge_seqence(self.points, self.e1, n_disk)
        inner_points2 = split_edge_seqence(self.points, self.e2, n_disk)
        inner_points2 = inner_points2[::-1]  # innverse one of long edge

        center_points = (inner_points1 + inner_points2) / 2  # disk center

        return inner_points1, inner_points2, center_points, self.point_rectangle

    def __repr__(self):
        return str(self.__dict__)

    def __getitem__(self, item):
        return getattr(self, item)


class TotalText(data.Dataset):

    def __init__(self, data_root, ignore_list=None, is_training=True, transform=None):
        super().__init__()
        self.data_root = data_root
        self.is_training = is_training
        self.transform = transform

        if ignore_list:
            with open(ignore_list) as f:
                ignore_list = f.readlines()
                ignore_list = [line.strip() for line in ignore_list]
        else:
            ignore_list = []

        self.image_root = os.path.join(data_root, 'Images', 'train_icdar2019' if is_training else 'test_icdar2019')
        self.annotation_root = os.path.join(data_root, 'gt', 'train_icdar2019' if is_training else 'test_icdar2019')
        self.image_list = os.listdir(self.image_root)
        self.image_list = list(filter(lambda img: img.replace('.jpg', '') not in ignore_list, self.image_list))
        self.polygons = [None] * len(self.image_list)  # polygon cache

    def parse_mat_1(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = read_lines(mat_path)
        polygon = []
        for line in lines:
            line = remove_all(line, '\xef\xbb\xbf')
            line = remove_all(line, '\n')
            gt = line.split(',')
            text = gt[-1]
            oriented_box = [int(gt[i]) for i in range(len(gt) - 1)]
            oriented_box = np.asarray(oriented_box)
            # print(len(oriented_box))
            xs = oriented_box.reshape(int(len(oriented_box) / 2), 2)[:, 0]
            ys = oriented_box.reshape(int(len(oriented_box) / 2), 2)[:, 1]

            if len(xs) < 4:  # too few points
                continue

            ori = 'c'
            pts = np.stack([xs, ys]).T.astype(np.int32)
            # get rectangle
            x_min = xs.min() - 8
            y_min = ys.min() - 8
            x_max = xs.max() + 8
            y_max = ys.max() + 8
            point_min = [x_min, x_max]
            point_max = [y_min, y_max]

            # point_rectangle = [x_min, y_min, x_max, y_max]
            point_rectangle = np.stack([point_min, point_max]).T.astype(np.int32)

            x_ = [x_min, x_max, x_max, x_min]
            y_ = [y_min, y_min, y_max, y_max]
            pointss = np.stack([x_, y_]).T.astype(np.int32)

            polygon.append(TextInstance(pts, ori, text, point_rectangle, pointss))
        return polygon

    def parse_mat(self, mat_path):
        """
        .mat file parser
        :param mat_path: (str), mat file path
        :return: (list), TextInstance
        """
        annot = io.loadmat(mat_path)
        polygon = []
        for cell in annot['polygt']:
            x = cell[1][0]
            y = cell[3][0]
            text = cell[4][0]
            if len(x) < 4:  # too few points
                continue
            try:
                ori = cell[5][0]
            except:
                ori = 'c'
            pts = np.stack([x, y]).T.astype(np.int32)
            # get rectangle
            x_min = x.min() - 8
            y_min = y.min() - 8
            x_max = x.max() + 8
            y_max = y.max() + 8
            point_min = [x_min, x_max]
            point_max = [y_min, y_max]

            # point_rectangle = [x_min, y_min, x_max, y_max]
            point_rectangle = np.stack([point_min, point_max]).T.astype(np.int32)

            x_ = [x_min, x_max, x_max, x_min]
            y_ = [y_min, y_min, y_max, y_max]
            pointss = np.stack([x_, y_]).T.astype(np.int32)

            polygon.append(TextInstance(pts, ori, text, point_rectangle, pointss))
        return polygon

    def make_text_region(self, image, polygons):

        tr_mask = np.zeros(image.shape[:2], np.uint8)
        train_mask = np.ones(image.shape[:2], np.uint8)

        rectangular_box = np.zeros(image.shape[:2], np.uint8)

        for polygon in polygons:
            cv2.fillPoly(tr_mask, [polygon.points.astype(np.int32)], color=(1,))
            # cv2.fillPoly(rectangular_box, [polygon.pointss.astype(np.int32)], color=(1,))
            a1, a2, a3, a4 = restrict(polygon.pointss[0], polygon.pointss[1], polygon.pointss[2], polygon.pointss[3])
            self.fill_polygon(rectangular_box, np.stack([a1, a2, a3, a4]), value=1)
            if polygon.text == '#':
                cv2.fillPoly(train_mask, [polygon.points.astype(np.int32)], color=(0,))
        return tr_mask, train_mask, rectangular_box

    def fill_polygon(self, mask, polygon, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param polygon: polygon to draw
        :param value: fill value
        """
        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0])
        mask[rr, cc] = value

    def make_text_center_line(self, sideline1, sideline2, center_line, \
                              tcl_mask, up_mask, down_mask, left_mask, right_mask, expand=0.2, shrink=2):

        # TODO: shrink 1/2 * radius at two line end
        for i in range(0, len(center_line) - 1):

            c1 = center_line[i]
            c2 = center_line[i + 1]
            top1 = sideline1[i]
            top2 = sideline1[i + 1]
            bottom1 = sideline2[i]
            bottom2 = sideline2[i + 1]

            top1, top2, bottom1, bottom2 = restrict(top1, top2, bottom1, bottom2)

            p1 = c1 + (top1 - c1) * expand
            p2 = c1 + (bottom1 - c1) * expand
            p3 = c2 + (bottom2 - c2) * expand
            p4 = c2 + (top2 - c2) * expand

            p1, p2, p3, p4 = restrict(p1, p2, p3, p4)

            new_coordinate_x = c2[0] - c1[0]
            new_coordinate_y = c2[1] - c1[1]

            if abs(new_coordinate_x) > 0.1:
                tanx = new_coordinate_y * 1.0 / new_coordinate_x
            else:
                tanx = 0

            if -0.588 < tanx < 0.588:
                if p1[1] + p4[1] > p2[1] + p3[1]:
                    polygon = np.stack([p1, p4, top2, top1])
                    self.fill_polygon(down_mask, polygon, value=1)

                    polygon = np.stack([p2, p3, bottom2, bottom1])
                    self.fill_polygon(up_mask, polygon, value=1)
                else:
                    polygon = np.stack([p1, p4, top2, top1])
                    self.fill_polygon(up_mask, polygon, value=1)

                    polygon = np.stack([p2, p3, bottom2, bottom1])
                    self.fill_polygon(down_mask, polygon, value=1)
            elif tanx < -1.732 or tanx > 1.732:
                if p1[0] + p4[0] > p2[0] + p3[0]:
                    polygon = np.stack([p1, p4, top2, top1])
                    self.fill_polygon(right_mask, polygon, value=1)

                    polygon = np.stack([p2, p3, bottom2, bottom1])
                    self.fill_polygon(left_mask, polygon, value=1)
                else:
                    polygon = np.stack([p1, p4, top2, top1])
                    self.fill_polygon(left_mask, polygon, value=1)

                    polygon = np.stack([p2, p3, bottom2, bottom1])
                    self.fill_polygon(right_mask, polygon, value=1)
            else:
                if p1[1] + p4[1] > p2[1] + p3[1]:
                    polygon = np.stack([p1, p4, top2, top1])
                    self.fill_polygon(down_mask, polygon, value=1)

                    polygon = np.stack([p2, p3, bottom2, bottom1])
                    self.fill_polygon(up_mask, polygon, value=1)
                else:
                    polygon = np.stack([p1, p4, top2, top1])
                    self.fill_polygon(up_mask, polygon, value=1)

                    polygon = np.stack([p2, p3, bottom2, bottom1])
                    self.fill_polygon(down_mask, polygon, value=1)
                if p1[0] + p4[0] > p2[0] + p3[0]:
                    polygon = np.stack([p1, p4, top2, top1])
                    self.fill_polygon(right_mask, polygon, value=1)

                    polygon = np.stack([p2, p3, bottom2, bottom1])
                    self.fill_polygon(left_mask, polygon, value=1)
                else:
                    polygon = np.stack([p1, p4, top2, top1])
                    self.fill_polygon(left_mask, polygon, value=1)

                    polygon = np.stack([p2, p3, bottom2, bottom1])
                    self.fill_polygon(right_mask, polygon, value=1)

            polygon = np.stack([p1, p2, p3, p4])

            self.fill_polygon(tcl_mask, polygon, value=1)

    def __getitem__(self, item):

        image_id = self.image_list[item]  # 'img725.jpg'
        # image_id = 'img1042.jpg'
        image_path = os.path.join(self.image_root, image_id)

        if image_id.split('_')[0] == 'gt':
            annotation_id = image_id.split('.')[0] + '.txt'
            annotation_path = os.path.join(self.annotation_root, annotation_id)
            polygons = self.parse_mat_1(annotation_path)

        else:
            annotation_id = 'poly_gt_{}.mat'.format(image_id.replace('.jpg', ''))
            annotation_path = os.path.join(self.annotation_root, annotation_id)
            polygons = self.parse_mat(annotation_path)

        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                polygon.find_bottom_and_sideline()

        # print(image_path, annotation_path)
        # Read image data
        image = pil_load_img(image_path)

        H, W, _ = image.shape

        try:
            if self.transform:
                image, polygons = self.transform(image, copy.copy(polygons))
        except:
            print(image_id)
            raise NameError
        tcl_mask = np.zeros(image.shape[:2], np.uint8)

        up_mask = np.zeros(image.shape[:2], np.uint8)
        down_mask = np.zeros(image.shape[:2], np.uint8)
        left_mask = np.zeros(image.shape[:2], np.uint8)
        right_mask = np.zeros(image.shape[:2], np.uint8)

        dege_mask_weight = np.zeros(image.shape[:2], np.uint8)
        dege_mask = np.zeros(image.shape[:2], np.uint8)

        # get radius, tcl_mask, radius_map, sin_map, cos_map

        tr_mask, train_mask, rectangular_box = self.make_text_region(image, polygons)
        for i, polygon in enumerate(polygons):
            if polygon.text != '#':
                sideline1, sideline2, center_points, point_rectangle = polygon.disk_cover()
                self.make_text_center_line(sideline1, sideline2, center_points, tcl_mask \
                                           , up_mask, down_mask, left_mask, right_mask)

        # tcl weight
        tcl_number = find_contours(tcl_mask)
        all_tcl_area = tcl_mask.sum()
        text_instance_weight = all_tcl_area / len(tcl_number)
        tcl_weight = tr_mask

        for number, tcl_coordinate in enumerate(tcl_number):
            one_tcl_weight = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(one_tcl_weight, [tcl_coordinate.astype(np.int32)], color=(1,))
            single_text_instance_weight = text_instance_weight * 1.0 / one_tcl_weight.sum()
            one_tcl_weight_final = one_tcl_weight * single_text_instance_weight

            tcl_weight = tcl_weight - one_tcl_weight + one_tcl_weight_final

        # to pytorch channel sequence
        image = image.transpose(2, 0, 1)
        image = torch.from_numpy(image).float()
        tcl_weight = torch.from_numpy(tcl_weight).float()
        dege_mask_weight = torch.from_numpy(dege_mask_weight).float()
        meta = {
            'image_id': image_id,
            'image_path': image_path,
            'Height': H,
            'Width': W
        }
        return image, train_mask, rectangular_box, tr_mask, tcl_mask, tcl_weight, dege_mask, dege_mask_weight, up_mask, down_mask, left_mask, right_mask, meta
    def __len__(self):
        return len(self.image_list)

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from pkgs.util.option import BaseOptions
    from pkgs.util.augmentation import BaseTransform, Augmentation
    from pkgs.util.config import config as cfg, update_config, print_config
    # parse arguments
    option = BaseOptions()
    args = option.initialize()
    update_config(cfg, args)


    vis_dir = '/home/weijia.wu/workspace/Sence_Text_detection/Paper-ICDAR/TCL_TEL_9/pkgs/Test_result/train_show/'

    trainset = TotalText(
        data_root='/data/data_weijia/CTW1500_Total_ICDAR2019/',
        ignore_list='/data/data_weijia/Total_Text/ignore_list.txt',
        is_training=True,
        transform=BaseTransform
        (size=cfg.input_size, mean=cfg.means, std=cfg.stds)
    )
    train_loader = data.DataLoader(trainset,batch_size=2,shuffle=True,num_workers=cfg.num_workers)

    # for i, (image, train_mask, rectangular_box, tr_mask, tcl_mask, tcl_weight, dege_mask, dege_mask_weight, up_mask, down_mask, left_mask, right_mask, \
    # meta) in enumerate(train_loader):
    #     print('1111')

    image, train_mask,rectangular_box, tr_mask, tcl_mask,tcl_weight, dege_mask, dege_mask_weight,up_mask, down_mask, left_mask, right_mask,\
    meta = next(iter(train_loader))

    print('image:',image.shape)
    print('train_mask:',train_mask.shape)
    print('tr_mask:',tr_mask.shape)
    print('tcl_mask:',tcl_mask.shape)


    image = image[0].numpy() * 255
    tr_mask =tr_mask[0].numpy() * 255
    tcl_mask = tcl_mask[0].numpy() * 255
    dege_mask = dege_mask[0].numpy() * 255
    rectangular_box = rectangular_box[0].numpy() * 255

    up_mask = up_mask[0].numpy() * 255
    down_mask = down_mask[0].numpy() * 255
    left_mask = left_mask[0].numpy() * 255
    right_mask = right_mask[0].numpy() * 255
    tcl_weight = tcl_weight[0].numpy()


    dege_mask_weight = dege_mask_weight[0].numpy()
    np.savetxt(vis_dir + 'tcl_weight.txt',
              tcl_weight, fmt='%0.2f')


    path = os.path.join(vis_dir, 'tcl_mask.jpg')
    path1 = os.path.join(vis_dir, 'tr_mask.jpg')
    path2 = os.path.join(vis_dir, 'image.jpg')
    path3 = os.path.join(vis_dir, 'rectangular_box.jpg')
    cv2.imwrite(path3, rectangular_box)
    cv2.imwrite(path, tcl_mask)
    cv2.imwrite(path1, tr_mask)
    image = image.transpose(1,2,0)


    cv2.imwrite(path2, image)
    print(meta)

    from pkgs.util.TextCohesion_decode import visualization_
    from pkgs.util.config import config as cfg, update_config, print_config

    update_config(cfg, args)
    visualization_(tr_mask/255,tcl_mask/255,up_mask/255,down_mask/255,left_mask/255,right_mask/255,rectangular_box/255,meta['image_id'][0])

    # rectangle = np.zeros([512,512],np.uint8)
    # all_point_rectangele = all_point_rectangele[0]
    # cv2.fillPoly(rectangle, [all_point_rectangele.numpy().astype(np.int32)], color=(255,))
    # path2 = os.path.join(vis_dir, 'rectangle.jpg')
    # cv2.imwrite(path2, rectangle)





