import numpy as np
import math
import cv2
import numpy.random as random
from matplotlib import pyplot as plt
from skimage.draw import polygon as drawpoly
from skimage.draw import draw
import torchvision
import PIL.Image as Image
import copy

class Color(torchvision.transforms.ColorJitter):
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25):
        super(Color, self).__init__(brightness, contrast, saturation, hue)

    def __call__(self, img, polygons=None):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return np.array(transform(Image.fromarray(img))), polygons


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, pts=None):
        for t in self.transforms:
            img, pts = t(img, pts)
        return img, pts


def rm_pts(points):
    points = np.vstack((points[0], points[1])).T
    ori_area = cv2.contourArea(points)
    #print('done')


class perspective(object):
    def __init__(self, anglex=0, angley=30, anglez=0, fov=42, r=0):
        self.anglex = anglex
        self.angley = angley
        self.anglez = anglez
        self.fov = fov
        self.r = r

    def rad(self, x):
        return x * np.pi / 180

    def getOnePolygonMask(self, polygon, value):
        """
        fill polygon in the mask with value
        :param mask: input mask
        :param polygon: polygon to draw
        :param value: fill value
        """
        mask = np.zeros((self.w, self.h))
        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0], shape=mask.shape)
        mask[rr, cc] = value

        return mask

    def getWarpR(self):
        self.anglex = np.random.randint(-50, 50)
        self.angley = np.random.randint(-50, 50)
        self.anglez = np.random.randint(-50, 50)

        z = np.sqrt(self.w ** 2 + self.h ** 2) / 2 / np.tan(self.rad(self.fov / 2))
        rx = np.array([[1, 0, 0, 0],
                       [0, np.cos(self.rad(self.anglex)), -np.sin(self.rad(self.anglex)), 0],
                       [0, -np.sin(self.rad(self.anglex)), np.cos(self.rad(self.anglex)), 0, ],

                       [0, 0, 0, 1]], np.float32)
        ry = np.array([[np.cos(self.rad(self.angley)), 0, np.sin(self.rad(self.angley)), 0],
                       [0, 1, 0, 0],
                       [-np.sin(self.rad(self.angley)), 0, np.cos(self.rad(self.angley)), 0, ],
                       [0, 0, 0, 1]], np.float32)

        rz = np.array([[np.cos(self.rad(self.anglez)), np.sin(self.rad(self.anglez)), 0, 0],
                       [-np.sin(self.rad(self.anglez)), np.cos(self.rad(self.anglez)), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]], np.float32)

        r = rx.dot(ry).dot(rz)

        # 四对点的生成
        pcenter = np.array([self.h / 2, self.w / 2, 0, 0], np.float32)

        p1 = np.array([0, 0, 0, 0], np.float32) - pcenter
        p2 = np.array([self.w, 0, 0, 0], np.float32) - pcenter
        p3 = np.array([0, self.h, 0, 0], np.float32) - pcenter
        p4 = np.array([self.w, self.h, 0, 0], np.float32) - pcenter

        dst1 = r.dot(p1)
        dst2 = r.dot(p2)
        dst3 = r.dot(p3)
        dst4 = r.dot(p4)

        list_dst = [dst1, dst2, dst3, dst4]

        org = np.array([[0, 0],
                        [self.w, 0],
                        [0, self.h],
                        [self.w, self.h]], np.float32)

        dst = np.zeros((4, 2), np.float32)

        # 投影至成像平面
        for i in range(4):
            dst[i, 0] = list_dst[i][0] * z / (z - list_dst[i][2]) + pcenter[0]
            dst[i, 1] = list_dst[i][1] * z / (z - list_dst[i][2]) + pcenter[1]

        warpR = cv2.getPerspectiveTransform(org, dst)
        return warpR

    def __call__(self, img, anno=None):
        # showpoly(img,anno,1)
        # print(anno[0].points)
        # print('nnn')
        # show(img)
        # showpoly(img, anno, value=1)
        # if np.random.randint(2):
        #     return img, polygons
        self.w, self.h = img.shape[0:2]
        # show(img)
        # showpoly(np.zeros((512,512,3)),anno,1)
        warpR = self.getWarpR()
        img = cv2.warpPerspective(img, warpR, (self.h, self.w))
        # show(img)
        poly_mask = []
        # pm_mask = np.zeros((512, 512))
        new_anno = copy.copy(anno)
        for i,ann in enumerate(anno):
            mask = cv2.warpPerspective(self.getOnePolygonMask(ann.points, value=1), warpR, (self.h, self.w))
            # mask = np.array(mask, np.uint8)
            cts = find_contours(mask)
            if len(cts) != 1:
                print(len(cts))
                show_numpypoly(ann.points)
                show(mask)
                print('')

            if len(cts) == 1:
                cts = np.squeeze(cts[0])
                result = np.squeeze(cv2.approxPolyDP(cts, 2, True, ))
                ann.points = result
                new_anno[i].points = result



            # show_numpypoly(result)
        showpoly(img,anno,1)
        # print('done')
        # print(new_anno[0].points)

        return img, new_anno

        # cnts,relation = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # cnts = cnts[0].reshape(cnts.shape[0],-1)
        # cts = np.where(mask)
        # show_numpypoly(cts)
        #     show_numpypoly(cts)
        #     print(np.size(cnts))
        #     point = np.array(cv_point[0], np.uint32)
        #     cv2.drawContours(img, mas, -1, (0, 0, 255), 3)
        #     show(img)
        #     show(edge)
        #     rm_pts(point)
        #     poly_mask.append(mask)
        # for pm in poly_mask:
        #     pm_mask += pm
        #     show(pm_mask)
        #     print('done')

        # showarray(img,poly_mask,value=1)

        for pm in poly_mask:
            point = np.where(pm)
            rm_pts(point)
        point = np.where(mask)
        test_point = np.vstack((point[0], point[1])).T
        rm_pts(test_point)
        showarray(img, point[0], point[1], 1)

        # self.get_polygon(np.zeros((self.w,self.h)),anno.)
        showpoly(img, anno, value=1)
        return img, anno


def find_contours(mask, method=None):
    if method is None:
        method = cv2.CHAIN_APPROX_SIMPLE
    mask = np.asarray(mask, dtype=np.uint8)
    mask = mask.copy()
    try:
        contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                       method=method)
    except:
        _, contours, _ = cv2.findContours(mask, mode=cv2.RETR_CCOMP,
                                          method=method)
    return contours


class RandomMirror(object):
    def __init__(self):
        pass

    def __call__(self, image, TextInstances=None):
        if np.random.randint(2):
            image = np.ascontiguousarray(image[:, ::-1])
            _, width, _ = image.shape
            for oneInstance in TextInstances:
                oneInstance.points[:, 0] = width - oneInstance.points[:, 0]

                oneInstance.pointss[:, 0] = width - oneInstance.pointss[:, 0]

        return image, TextInstances


class AugmentColor(object):
    def __init__(self):
        self.U = np.array([[-0.56543481, 0.71983482, 0.40240142],
                           [-0.5989477, -0.02304967, -0.80036049],
                           [-0.56694071, -0.6935729, 0.44423429]], dtype=np.float32)
        self.EV = np.array([1.65513492, 0.48450358, 0.1565086], dtype=np.float32)
        self.sigma = 0.1
        self.color_vec = None

    def __call__(self, img, polygons=None):
        color_vec = self.color_vec
        if self.color_vec is None:
            if not self.sigma > 0.0:
                color_vec = np.zeros(3, dtype=np.float32)
            else:
                color_vec = np.random.normal(0.0, self.sigma, 3)

        alpha = color_vec.astype(np.float32) * self.EV
        noise = np.dot(self.U, alpha.T) * 255
        return np.clip(img + noise[np.newaxis, np.newaxis, :], 0, 255), polygons


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)

        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            image *= alpha
        #show(image)
        return np.clip(image, 0, 255), polygons


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
        return np.clip(image, 0, 255), polygons


class Rotate(object):
    def __init__(self, up=30):
        self.up = up

    def rotate(self, center, pt, theta):  # 二维图形学的旋转
        xr, yr = center
        yr = -yr
        x, y = pt[:, 0], pt[:, 1]
        y = -y

        theta = theta / 360 * 2 * math.pi
        cos = math.cos(theta)
        sin = math.sin(theta)

        _x = xr + (x - xr) * cos - (y - yr) * sin
        _y = yr + (x - xr) * sin + (y - yr) * cos

        return _x, -_y

    def __call__(self, img, polygons=None):
        #show(img)
        if np.random.randint(2):
            return img, polygons
        angle = np.random.normal(loc=0.0, scale=0.5) * self.up  # angle 按照高斯分布
        rows, cols = img.shape[0:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (cols, rows), borderValue=[0, 0, 0])

        center = cols / 2.0, rows / 2.0
        if polygons is not None:
            for polygon in polygons:
                x, y = self.rotate(center, polygon.points, angle)
                pts = np.vstack([x, y]).T
                polygon.points = pts

                x_, y_ = self.rotate(center, polygon.pointss, angle)
                pts_ = np.vstack([x_, y_]).T
                polygon.pointss = pts_

        return img, polygons


class SquarePadding(object):

    def __call__(self, image, pts=None):

        H, W, _ = image.shape

        if H == W:
            return image, pts

        padding_size = max(H, W)
        expand_image = np.zeros((padding_size, padding_size, 3), dtype=image.dtype)

        if H > W:
            y0, x0 = 0, (H - W) // 2
        else:
            y0, x0 = (W - H) // 2, 0
        if pts is not None:
            pts[:, 0] += x0
            pts[:, 1] += y0

        expand_image[y0:y0 + H, x0:x0 + W] = image
        image = expand_image

        return image, pts


class Padding(object):

    def __init__(self, fill=0):
        self.fill = fill

    def __call__(self, image, pts):
        if np.random.randint(2):
            return image, pts

        height, width, depth = image.shape
        ratio = np.random.uniform(1, 2)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)

        expand_image = np.zeros(
            (int(height * ratio), int(width * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = self.fill
        expand_image[int(top):int(top + height),
        int(left):int(left + width)] = image
        image = expand_image

        pts[:, 0] += left
        pts[:, 1] += top
        return image, pts


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w

            if h < img.shape[0] and w < img.shape[1]:
                j = np.random.randint(0, img.shape[1] - w)
                i = np.random.randint(0, img.shape[0] - h)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, image, pts=None):
        i, j, h, w = self.get_params(image, self.scale, self.ratio)
        cropped = image[i:i + h, j:j + w, :]
        pts = pts.copy()
        mask = (pts[:, 1] >= i) * (pts[:, 0] >= j) * (pts[:, 1] < (i + h)) * (pts[:, 0] < (j + w))
        pts[~mask, 2] = -1
        scales = np.array([self.size[0] / w, self.size[1] / h])
        pts[:, :2] -= np.array([j, i])
        pts[:, :2] = (pts[:, :2] * scales)
        img = cv2.resize(cropped, self.size)
        return img, pts


class RandomResizedLimitCrop(object):
    def __init__(self, size, scale=(0.3, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img, scale, ratio):
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = np.random.uniform(*scale) * area
            aspect_ratio = np.random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if np.random.random() < 0.5:
                w, h = h, w

            if h < img.shape[0] and w < img.shape[1]:
                j = np.random.randint(0, img.shape[1] - w)
                i = np.random.randint(0, img.shape[0] - h)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, image, pts):
        num_joints = np.sum(pts[:, -1] != -1)
        attempt = 0
        scale_vis = 0.75
        while attempt < 10:
            i, j, h, w = self.get_params(image, self.scale, self.ratio)
            mask = (pts[:, 1] >= i) * (pts[:, 0] >= j) * (pts[:, 1] < (i + h)) * (pts[:, 0] < (j + w))
            if np.sum(mask) >= (round(num_joints * scale_vis)):
                break
            attempt += 1
        if attempt == 10:
            w = min(image.shape[0], image.shape[1])
            h = w
            i = (image.shape[0] - w) // 2
            j = (image.shape[1] - w) // 2

        cropped = image[i:i + h, j:j + w, :]
        pts = pts.copy()
        mask = (pts[:, 1] >= i) * (pts[:, 0] >= j) * (pts[:, 1] < (i + h)) * (pts[:, 0] < (j + w))
        pts[~mask, 2] = -1
        scales = np.array([self.size[0] / w, self.size[1] / h])
        pts[:, :2] -= np.array([j, i])
        pts[:, :2] = (pts[:, :2] * scales).astype(np.int)
        img = cv2.resize(cropped, self.size)
        return img, pts


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, image, polygons=None):
        image = image.astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std
        return image, polygons


class Resize(object):
    def __init__(self, size=256):
        self.size = size

    def __call__(self, image, polygons=None):
        h, w, _ = image.shape
        image = cv2.resize(image, (self.size,
                                   self.size))
        scales = np.array([self.size / w, self.size / h])

        if polygons is not None:
            for polygon in polygons:
                polygon.points = polygon.points * scales
                polygon.pointss = polygon.pointss *scales

        return image, polygons


class Augmentation(object):

    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([

            Color(),
            RandomMirror(),
            #perspective(),
            # RandomBrightness(),
            # RandomContrast(),
            Rotate(),
            Resize(size),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)


class BaseTransform(object):
    def __init__(self, size, mean, std):
        self.size = size
        self.mean = mean
        self.std = std
        self.augmentation = Compose([
            Resize(size),
            Normalize(mean, std)
        ])

    def __call__(self, image, polygons=None):
        return self.augmentation(image, polygons)


def showpoly(img, polygons, value=1):
    for polygon in polygons:
        test1,test2 = polygon.points[:, 1], polygon.points[:, 0]
        rr, cc = drawpoly(polygon.points[:, 1], polygon.points[:, 0])
        draw.set_color(img, [rr, cc], [255, 0, 0], alpha=0.5)
    show(img)

def showrec(img, recs, value=1):
    for rec in recs:
        x1,y1 = rec.point_rectangle[0]
        x2,y2 = rec.point_rectangle[1]
        rr, cc = drawpoly([y2,y2,y1,y1], [x1,x2,x2,x1])
        draw.set_color(img, [rr, cc], [255, 0, 0], alpha=0.5)
    # show(img)


def show_numpypoly(inx,img=None):
    if img is not None:
        mask = img
    else:
        mask = np.zeros((512, 512, 3))
    rr, cc = drawpoly(inx[:, 1], inx[:, 0])
    draw.set_color(mask, [rr, cc], [255, 0, 0], alpha=0.5)
    # show(mask)


def showarray(img, polygons, value=1):
    for polygon in polygons:
        rr, cc = drawpoly(polygon[:, 1], polygon[:, 0])
        draw.set_color(img, [rr, cc], [255, 0, 0], alpha=0.5)
    show(img)


def mask_add(img, mask):
    cv2.addWeighted(img, 1, mask, 0.5, 0.001)


def show(img):
    plt.imshow(img)
    plt.show()
