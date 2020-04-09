from PIL import Image
import numpy as np
import cv2
import os

def read_lines(p):
    """return the text in a file in lines as a list """
    p = get_absolute_path(p)
    f = open(p, 'rU', encoding='utf-8-sig')
    return f.readlines()

def get_absolute_path(p):
    if p.startswith('~'):
        p = os.path.expanduser(p)
    return os.path.abspath(p)

def remove_all(s, sub):
    return replace_all(s, sub, '')

def replace_all(s, old, new, reg=False):
    if reg:
        import re
        targets = re.findall(old, s)
        for t in targets:
            s = s.replace(t, new)
    else:
        s = s.replace(old, new)
    return s

def pil_load_img(path):
    image = Image.open(path)
    image = np.array(image)
    return image

def norm2(x,y):
    return np.sqrt(x ** 2 + y ** 2)

def get_distance_center_edge(x,y,center_points):
    distances = []
    for i , point in enumerate(center_points):
        distance = norm2(x - point[0], y - point[1])
        distances.append(distance)
    return distances

def make_centripetal_force(tr_mask, tcl_mask, point_rectangle,center_points, \
            up_mask, down_mask, left_mask, right_mask, left_up_mask, left_down_mask, right_up_mask, right_down_mask, \
                           dege_mask, dege_mask_weight,all_point):
    # TODO: get
    for x in range(int(point_rectangle[0][0]),int(point_rectangle[1][0])):
        for y in range(int(point_rectangle[0][1]),int(point_rectangle[1][1])):
            # if (tr_mask[y][x] == 1) and (tcl_mask[y][x] == 0):
            if x >= 512 or y >= 512:
                continue
            if (cv2.pointPolygonTest(all_point,(x,y), False) >= 0) and (tcl_mask[y][x] == 0):

                # get the distance to tcl
                distances = get_distance_center_edge(x,y,center_points)

                min_distance = distances.index(min(distances))

                # get the shortest distance point with me
                great_center_point = center_points[min_distance]

                dege_mask_weight[y][x] = min(distances)
                dege_mask[y][x] = 1

                new_coordinate_x = x - great_center_point[0]
                new_coordinate_y = y - great_center_point[1]

                if abs(new_coordinate_x) > 0.1:
                    tanx = new_coordinate_y * 1.0 /new_coordinate_x
                else:
                    tanx = 0

                if norm2(x - great_center_point[0],y - great_center_point[1]) < 0.01:
                    # print("common level points")
                    pass
                elif (abs(new_coordinate_x) >= abs(new_coordinate_y))and(new_coordinate_x >= 0):
                    right_mask[y][x] = 1
                    if tanx > 0.466:
                        down_mask[y][x] = 1
                    elif tanx < -0.466:
                        up_mask[y][x] = 1


                elif (abs(new_coordinate_x) >= abs(new_coordinate_y))and(new_coordinate_x <= 0):
                    left_mask[y][x] = 1
                    if tanx < -0.466:
                        down_mask[y][x] = 1
                    elif tanx > 0.466:
                        up_mask[y][x] = 1


                elif (abs(new_coordinate_x) <= abs(new_coordinate_y))and(new_coordinate_y <= 0):
                    up_mask[y][x] = 1
                    if 1< tanx < 2.144:
                        left_mask[y][x] = 1
                    elif -1> tanx > -2.144:
                        right_mask[y][x] = 1

                elif (abs(new_coordinate_x) <= abs(new_coordinate_y))and(new_coordinate_y >= 0):

                    down_mask[y][x] = 1
                    if -1>tanx > -2.144:
                        left_mask[y][x] = 1

                    elif 1< tanx < 2.144:
                        right_mask[y][x] = 1

                else:
                    print("abormal point")



    return up_mask, down_mask, left_mask, right_mask,dege_mask, dege_mask_weight

def make_dege_mask_force(tr_mask, tcl_mask, point_rectangle,center_points, \
            dege_mask, dege_mask_weight,all_point):


    # TODO: get
    for x in range(int(point_rectangle[0][0]), int(point_rectangle[1][0])):
        for y in range(int(point_rectangle[0][1]), int(point_rectangle[1][1])):
            # if (tr_mask[y][x] == 1) and (tcl_mask[y][x] == 0):
            if (cv2.pointPolygonTest(all_point, (x, y), False) >= 0) and (tcl_mask[y][x] == 0):

                distances = get_distance_center_edge(x,y,center_points)
                min_distance = min(distances)
                dege_mask_weight[y][x] = min_distance
                dege_mask[y][x] = 1

    return dege_mask, dege_mask_weight


def restrict(p1, p2, p3, p4):
    x, y = p1
    x = min(x, 512)
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