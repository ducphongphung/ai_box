# -*- coding: utf-8 -*-
import cv2
import PIL
from math import sin, cos, sqrt, atan2, radians
import os
import sys
import time
import numpy as np
import tornado.httpserver
import tornado.ioloop
import tornado.wsgi
import logging
import shutil
import base64
import datetime
import uuid
import codecs
import threading
import smtplib
import subprocess
import multiprocessing
import re
import six
import requests
import netifaces
from shapely.geometry import LineString
from collections import Counter
from io import StringIO


# region Computer Vision
# ==============================================================================
def center(bb):
    return (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2


def get_l2(v0, v1):
    v0 = list_to_npArray(v0)
    v1 = list_to_npArray(v1)
    return np.linalg.norm(v0 - v1)


def get_bb(coords):
    min_x = float('inf')  # start with something much higher than expected min
    min_y = float('inf')
    max_x = -float('inf')  # start with something much lower than expected max
    max_y = -float('inf')

    for item in coords:
        if item[0] < min_x:
            min_x = item[0]

        if item[0] > max_x:
            max_x = item[0]

        if item[1] < min_y:
            min_y = item[1]

        if item[1] > max_y:
            max_y = item[1]

    return [min_x, min_y, max_x, max_y]


def to_bbox(cv_rect):
    """converts (left, top, w, h) to (left, top, right, bottom) """
    return cv_rect[0], cv_rect[1], cv_rect[2] + cv_rect[0], cv_rect[3] + cv_rect[1]


def to_cv_rect(bbox):
    return bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]


def to_poly(bbox):
    return [(bbox[0], bbox[1]),
            (bbox[2], bbox[1]),
            (bbox[2], bbox[3]),
            (bbox[0], bbox[3]), ]


def to_square(cv_rect=None, bb=None):
    if bb is not None:
        cv_rect = to_cv_rect(bb)
    if cv_rect is not None:
        m = int(max(cv_rect[2], cv_rect[3]) / 2)
        c = (cv_rect[0] + int(cv_rect[2] / 2), cv_rect[1] + int(cv_rect[3] / 2))
        if bb is None:
            return c[0] - m, c[1] - m, m * 2, m * 2
        else:
            return c[0] - m, c[1] - m, c[0] + m, c[1] + m
    return None


def clip_bb(bb, max_w, max_h):
    return bb_intersection(bb, [0, 0, max_w, max_h])


def scale_bb(bb, scale):
    return [int(round(e * scale)) for e in bb[:4]]


def distance_to_image_border(box, img_w, img_h):
    return min((box[0], box[1], img_w - box[2], img_h - box[3]))


def bb_area(bb):
    return (bb[2] - bb[0]) * (bb[3] - bb[1])


def bb_min_distance_x(bb1, bb2):
    if (bb1[0] < bb2[0] < bb1[2]) or (bb2[0] < bb1[0] < bb2[2]):
        return 0
    return min(abs(bb1[0] - bb2[0]), abs(bb1[0] - bb2[2]), abs(bb1[2] - bb2[0]), abs(bb1[2] - bb2[2]))


def bb_min_distance_y(bb1, bb2):
    if (bb1[1] < bb2[1] < bb1[3]) or (bb2[1] < bb1[1] < bb2[3]):
        return 0
    return min(abs(bb1[1] - bb2[1]), abs(bb1[1] - bb2[3]), abs(bb1[3] - bb2[1]), abs(bb1[3] - bb2[3]))


def bb_max_distance(bb1, bb2):
    p10 = bb1[0], bb1[1]
    p11 = bb1[2], bb1[1]
    p12 = bb1[2], bb1[3]
    p13 = bb1[0], bb1[3]

    p20 = bb2[0], bb2[1]
    p21 = bb2[2], bb2[1]
    p22 = bb2[2], bb2[3]
    p23 = bb2[0], bb2[3]

    def d(p1, p2):
        return sqrt((p1[0] - p2[0])*(p1[0] - p2[0]) + (p1[1] - p2[1])*(p1[1] - p2[1]))

    return max(d(p10, p20), d(p11, p21), d(p12, p22), d(p13, p23))


def bb_intersection_over_union(bb1, bb2):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(bb1[0], bb2[0])
    yA = max(bb1[1], bb2[1])
    xB = min(bb1[2], bb2[2])
    yB = min(bb1[3], bb2[3])

    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)

    if (xB < xA) or (yB < yA):
        interArea = 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    boxBArea = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def bb_union(bb1, bb2):
    l = min(bb1[0], bb2[0])
    t = min(bb1[1], bb2[1])
    r = max(bb1[2], bb2[2])
    b = max(bb1[3], bb2[3])
    return l, t, r, b


def bb_intersection(bb1, bb2):
    l = max(bb1[0], bb2[0])
    t = max(bb1[1], bb2[1])
    r = min(bb1[2], bb2[2])
    b = min(bb1[3], bb2[3])
    if (r - l) <= 0 or (b - t) <= 0:
        return None
    return l, t, r, b


def bb_biggest(bboxes, topk=1):
    """ returns id of biggest bb in bboxes """
    if len(bboxes) < 1:
        return None
    else:
        ibbs = list(enumerate(bboxes))
        ibbs.sort(key=lambda x: (x[1][2] - x[1][0]) * (x[1][3] - x[1][1]), reverse=True)
        if topk == 1:
            return ibbs[0][0]
        else:
            return [ib[0] for ib in ibbs[:topk]]


def bb_rot90r(bb, img_h):
    return [img_h - bb[3], bb[0], img_h - bb[1], bb[2]]


def bb_rot90l(bb, img_w):
    return [bb[1], img_w - bb[2], bb[3], img_w - bb[0]]


def rect_intersection_over_union(cv_rect0, cv_rect1):
    bb1 = to_bbox(cv_rect0)
    bb2 = to_bbox(cv_rect1)

    return bb_intersection_over_union(bb1, bb2)


def to_rgb(bgr):
    return bgr[..., ::-1]


def to_bgr(rgb):
    return rgb[..., ::-1]


def to_pil(bgr):
    rgb = to_rgb(bgr)
    return PIL.Image.fromarray(rgb)


def to_wh(im):
    return im.shape[1], im.shape[0]


def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def _prewhiten_w_offset(x, roi_offset=0, nstd=1.0):
    w, h, c = x.shape
    x_center = x[roi_offset:h - roi_offset, roi_offset:w - roi_offset, :]
    mean = np.mean(x_center)
    std = np.std(x_center)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), nstd / std_adj)
    return y


_clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(4, 4))


def prewhiten(x, method=1, gamma_correction=True):
    if gamma_correction is True:
        x = adjust_gamma(x, 2)

    if method == 1:
        lab = cv2.cvtColor(x, cv2.COLOR_RGB2Lab)
        l = _clahe.apply(lab[:, :, 0])
        y = cv2.cvtColor(l, cv2.COLOR_GRAY2RGB)
        # y_bgr = to_bgr(y)
        # cv2.imwrite(os.path.expanduser('~/tmp/x1.jpg'), y_bgr)
        # y = _prewhiten_w_offset(y, 20, 0.5)
        y = _prewhiten(y)
    else:
        y = _prewhiten(x)

    # y_bgr = to_bgr(y+0.5)
    # cv2.imwrite(os.path.expanduser('~/tmp/1.jpg'), y_bgr * 255)

    return y


def _prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1.0 / std_adj)
    return y


def padAspectRatio(cv_img, woh):
    if woh < 1920.0 / 1080:
        return padFullScreen(cv_img, woh * 1080, 1080)
    else:
        return padFullScreen(cv_img, 1920, 1920 / woh)


def padFullHD(cv_img):
    return padFullScreen(cv_img, 1920, 1080)


def padFullScreen(cv_img, screen_w, screen_h, pad_color=(0, 0, 0)):
    s = max(float(cv_img.shape[1]) / screen_w, float(cv_img.shape[0]) / screen_h)
    h = int(cv_img.shape[0] / s)
    w = int(cv_img.shape[1] / s)
    hm = int((screen_h - h + 1) / 2)  # top, bottom margin
    wm = int((screen_w - w + 1) / 2)  # left, right margin
    resized = cv2.resize(cv_img, (w, h))
    return cv2.copyMakeBorder(resized, hm, hm, wm, wm, cv2.BORDER_CONSTANT, value=pad_color)


def padToSize(cv_img, w, h, pad_color=(0, 0, 0)):
    return padFullScreen(cv_img, w, h, pad_color)


def is_in_roi(bb, roi_tl, roi_br):
    """ True if bb center in roi_bb"""
    x = (bb[0] + bb[2]) / 2
    y = (bb[1] + bb[3]) / 2

    if (roi_br[0] > x > roi_tl[0]) and (roi_br[1] > y > roi_tl[1]):
        return True
    else:
        return False


def is_all_in_roi(bb, roi_bb):
    """ True if all bb point in roi_bb"""
    roi_tl = (roi_bb[0], roi_bb[1])
    roi_br = (roi_bb[2], roi_bb[3])
    tl_in = roi_tl[0] < bb[0] < roi_br[0] and roi_tl[1] < bb[1] < roi_br[1]
    br_in = roi_tl[0] < bb[2] < roi_br[0] and roi_tl[1] < bb[3] < roi_br[1]
    if tl_in and br_in:
        return True
    return False


def is_out_roi(bb, roi_bb):
    """ True if bb partly out of roi_bb"""
    return bb[0] <= roi_bb[0] or bb[1] <= roi_bb[1] or bb[2] >= roi_bb[2] or bb[3] >= roi_bb[3]


def crop(img, bbox=None, cv_rect=None, tl=None, br=None):
    if tl is None or br is None:
        if cv_rect is not None:
            bbox = to_bbox(cv_rect)
        if bbox is not None:
            tl = (bbox[0], bbox[1])
            br = (bbox[2], bbox[3])
    if tl is None or br is None:
        raise ValueError('crop error: roi is not specified')

    try:
        tl = (int(max(0, tl[0])), int(max(0, tl[1])))
        br = (int(min(img.shape[1], br[0])), int(min(img.shape[0], br[1])))

        if tl[1] >= br[1] or tl[0] >= br[0]:
            return None

        if len(img.shape) == 2:  # gray image?
            return img[tl[1]:br[1], tl[0]:br[0]]
        else:  # color image
            return img[tl[1]:br[1], tl[0]:br[0], :]
    except:
        return None


def crop_warp(im, projected_points, w, h):
    """warps perspective to crop & deskew [im]"""
    deskewed_points = np.float32([(0, 0), (w, 0), (w, h), (0, h)])
    T = cv2.getPerspectiveTransform(projected_points, deskewed_points)
    return cv2.warpPerspective(im, T, (w, h))


def resize(img, scale=None, dst_size=None, dst_w=None, dst_h=None, fast=False):
    if scale is not None:
        dst_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    elif dst_size is None:
        if dst_w is not None and dst_h is not None:
            dst_size = (int(dst_w), int(dst_h))
        elif dst_w is not None:
            dst_size = (int(dst_w), int(dst_w * img.shape[0] / img.shape[1]))
        elif dst_h is not None:
            dst_size = (int(dst_h * img.shape[1] / img.shape[0]), int(dst_h))

    if dst_size is not None:
        scale = max([float(dst_size[0]) / img.shape[1], float(dst_size[1]) / img.shape[0]])
    else:
        raise ValueError('resize error: cannot infer dst_size from input params')

    if scale < 1.0:
        interpolation = cv2.INTER_AREA
    else:
        if dst_size[0] == img.shape[1] and dst_size[1] == img.shape[0]:
            return img
        interpolation = cv2.INTER_LANCZOS4
    if fast:
        interpolation = cv2.INTER_LINEAR
    return cv2.resize(img, dst_size, interpolation=interpolation)


def line(p1, p2):
    A = (p2[1] - p1[1])
    B = (p1[0] - p2[0])
    C = A * p1[0] + B * p1[1]
    return A, B, -C


def distance_to_line(p, l):
    """ return distance from p to the line (x1, y1, x2, y2) """
    a, b, c = line(l[0:2], l[2:4])
    return (a*p[0] + b*p[1] + c) / sqrt(a*a + b*b)


def distance_2p(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


def overlap_len(x0, x1, x2, x3):
    """ return overlapping length between (x0, x1) and (x2, x3) """
    if x1 < x0 < x3:
        return min(abs(x0 - x3), abs(x0 - x1))
    if x0 < x2 < x1:
        return min(abs(x2 - x1), abs(x2 - x3))
    return 0


def midpoint(ptA, ptB):
    return (ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5


def check_validity(bbox, img):
    box = bbox.astype(int)
    return (box[0] >= 0) & (box[2] < img.shape[1]) & (box[1] >= 0) & (box[3] < img.shape[0])


def translate(bbox, dx, dy):
    return [bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy]


def nonezero_ratio(mask):
    if mask is None:
        return 0
    else:
        return float(cv2.countNonZero(mask)) / (mask.shape[0] * mask.shape[1])


def get_stability(trace, tt):
    if len(trace) < 3 or tt <= 0:
        return 0

    def track_center():
        i = -2
        dt = 0
        sumx = 0
        sumy = 0
        sumstep = 0
        while i > -len(trace) and dt < tt:
            ci = center(trace[i][0])
            sumx += ci[0]
            sumy += ci[1]
            sumstep += 1
            dt += (trace[i + 1][1] - trace[i][1])
            i -= 1
        return sumx / sumstep, sumy / sumstep

    c = track_center()
    i = -2
    dt = 0
    sum_d2 = 0
    sumstep = 0
    while i > -len(trace) and dt < tt:
        sum_d2 += distance_2p(center(trace[i][0]), c) ** 2
        sumstep += 1
        dt += (trace[i + 1][1] - trace[i][1])
        i -= 1

    std = (sum_d2 / sumstep) ** 0.5
    return std


def four_point_transform(image, pts, dst_size=None, padding_rate=0):
    # obtain a consistent order of the points and unpack them
    # individually
    (tl, tr, br, bl) = pts

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    pw, ph = (0, 0)
    if dst_size is not None:
        pw = int(padding_rate * dst_size[0])
        ph = int(padding_rate * dst_size[1])
        maxWidth = dst_size[0] + pw
        maxHeight = dst_size[1] + ph

    dst = np.array([
        [pw, ph],
        [maxWidth - pw, ph],
        [maxWidth - pw, maxHeight - ph],
        [pw, maxHeight - ph]], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(pts[-4:], dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def undistort_barrel(src, f=2.8, curvature=-1.0e-6):
    width = src.shape[1]
    height = src.shape[0]

    distCoeff = np.zeros((4, 1), np.float64)

    k1 = curvature  # negative to remove barrel distortion
    k2 = 0.0
    p1 = 0.0
    p2 = 0.0

    distCoeff[0, 0] = k1
    distCoeff[1, 0] = k2
    distCoeff[2, 0] = p1
    distCoeff[3, 0] = p2

    # assume unit matrix for camera
    cam = np.eye(3, dtype=np.float32)

    cam[0, 2] = width / 2.0  # define center x
    cam[1, 2] = height / 2.0  # define center y
    cam[0, 0] = f  # define focal length x
    cam[1, 1] = f  # define focal length y

    return cv2.undistort(src, cam, distCoeff)


def rotate_bound(image, angle_degree):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle_degree, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC)


def drect_to_cvrect(d):
    x = int(d.left())
    y = int(d.top())
    w = int((d.right() - d.left()))
    h = int((d.bottom() - d.top()))
    return x, y, w, h


def non_max_suppression(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    return pick


def bb_merge_list(high, low, overlap=0.5):
    """ merge, remove bb in [low] which overlaps any in [high] more than [overlap] """
    bboxes = list(low)  # copy
    for i in range(len(high)):
        bb = high[i]
        for j in range(len(bboxes)):
            if bboxes[j] is not None:
                inter_bb = bb_intersection(bboxes[j], bb)
                if inter_bb is not None:
                    overlap_rate = float(bb_area(inter_bb)) / bb_area(bb)
                    if overlap_rate > overlap:
                        bboxes[j] = None
        bboxes.append(bb)
    return [e for e in bboxes if e is not None]


def correlate(small, big, scale_range=(1,)):
    """Returns correlation between image crops"""
    min = 1
    for scale in scale_range:
        ss = resize(small, scale=scale)
        if ss.shape[0] < big.shape[0] and ss.shape[1] < big.shape[1]:
            match = cv2.matchTemplate(ss, big, cv2.TM_SQDIFF_NORMED)
            smin = cv2.minMaxLoc(match)[0]
            if smin < min:
                min = smin
    return 1-min


def intersection(pts1, pts2):
    s1 = LineString(pts1)
    s2 = LineString(pts2)
    return s1.intersection(s2)


def shape_intersects(pts1, pts2):
    """ return True if shapes intersect """
    s1 = LineString(pts1)
    s2 = LineString(pts2)
    return s1.intersects(s2)


def imsize(cvimage):
    h, w, c = cvimage.shape
    return w, h
# ==============================================================================
# endregion

# region File system
# ==============================================================================


def mkdir_if_not_existed(path):
    try:
        if not os.path.isdir(path):
            os.makedirs(path)
    except:
        pass


def get_immediate_subdirs(dir, full_path=False):
    if full_path:
        return [os.path.join(dir, name) for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]
    else:
        return [name for name in os.listdir(dir) if os.path.isdir(os.path.join(dir, name))]


def get_all_images(dir):
    return [os.path.join(dir, name) for name in os.listdir(dir) if is_valid_image(os.path.join(dir, name))]


def get_all_files(dir, full_path=True, ext=None):
    rs = [name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))]
    if ext:
        rs = [name for name in rs if name.endswith(ext)]
    if full_path:
        rs = [os.path.join(dir, f) for f in rs]
    return rs


def get_all_files_recursively(dir, ext=None):
    files = []
    for root, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            valid = False
            if ext:
                if filename.endswith(ext):
                    valid = True
            else:
                valid = True
            if valid:
                files.append(os.path.join(root, filename))
    return files


def get_file_extension(path):
    return os.path.splitext(path)[1]


def move_overwrite_file(src_file, dst_path, log_exception=True):
    src_file = os.path.abspath(src_file)
    dst_path = os.path.abspath(dst_path)
    if os.path.isfile(dst_path):
        if src_file == dst_path:
            return
    elif os.path.isdir(dst_path):
        if os.path.dirname(src_file) == dst_path:
            return

    dst_dir = os.path.dirname(dst_path)
    mkdir_if_not_existed(dst_dir)

    try:
        shutil.copy2(src_file, dst_path)
        os.remove(src_file)
    except Exception as ex:
        if log_exception:
            logging.error('move_overwrite error: %s', ex)


def move_overwrite(src_path, dst_path, exclude_exts=None):
    if os.path.isdir(src_path):  # move folder
        for src_dir, dirs, files in os.walk(src_path):
            dst_dir = src_dir.replace(src_path, dst_path, 1)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)
            for file_ in files:
                src_file = os.path.join(src_dir, file_)
                dst_file = os.path.join(dst_dir, file_)
                skip = False
                if exclude_exts is not None:
                    for ext in exclude_exts:
                        if src_file.endswith(ext):
                            skip = True
                if skip is False:
                    move_overwrite_file(src_file, dst_file)
        try:
            shutil.rmtree(src_path)
        except Exception as ex:
            logging.error('move_overwrite, remove error: %s', ex)
    elif os.path.isfile(src_path):
        move_overwrite_file(src_path, dst_path)


def cp_img(src_file, dst_dir=None, dst_file=None):
    """ copy image <src_file> to <dst_dir>, and change the image format (jpg, png...) """
    dst = None
    if dst_dir:
        mkdir_if_not_existed(dst_dir)
        dst = dst_dir

    if dst_file:
        dir_path = os.path.dirname(dst_file)
        mkdir_if_not_existed(dir_path)
        # if not same format, convert
        if not src_file.endswith(get_file_extension(dst_file)):
            img = cv2.imread(src_file)
            cv2.imwrite(dst_file, img)
            return
        dst = dst_file

    if dst:
        shutil.copy(src_file, dst)


def dir_size(path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += (os.path.getsize(fp) if os.path.isfile(fp) else 0)
    return total_size


def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

# ==============================================================================
# endregion

# region Operating System
# ==============================================================================


def get_free_disk_space(path):
    st = os.statvfs(path)
    free = (st.f_bavail * st.f_frsize)
    return free


def get_iface_ip(ifname):
    """
    :param ifname: interface name, e.g. eth0 or wlan0 (wifi-card)
    :return: ip of the input interface
    """
    import socket
    import fcntl
    import struct
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    p = struct.pack('256s', bytes(ifname[:15], 'utf8'))
    try:
        return socket.inet_ntoa(fcntl.ioctl(
            s.fileno(),
            0x8915,  # SIOCGIFADDR
            p,
        )[20:24])
    except:
        return ''


def disable_gpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def get_uuid():
    return uuid.uuid4().hex


def ping(hostname, timeout=-1):
    if timeout > 0:
        cmd = "ping -c 1 -w {} ".format(timeout)
    else:
        cmd = "ping -c 1 "
    response = os.system(cmd + hostname)
    return response == 0  # 0: is up


def ping_latency(hostname):
    try:
        ping = subprocess.Popen("ping {} -c 1".format(hostname), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                shell=True)
        output = ping.communicate()
        pattern = r"min/avg/max/mdev = (\d+)"
        matches = re.findall(pattern, output[0].decode())
        if len(matches) > 0:
            return int(matches[0])
    except Exception as ex:
        logging.exception(ex)

    return 100000


def parallel_do(func, list_args):
    pool = multiprocessing.Pool()
    try:
        results = pool.map(func, list_args)
    finally:
        pool.close()
    return results


def exit_if_maxi_reached(process_name, maxi):
    cnt = 0
    for line in os.popen("ps -A | grep " + process_name[:10] + " | grep -v grep"):
        cnt += 1
    if cnt > maxi:
        logging.error('Max number of licensed service ports: {}. Limit reached.'.format(maxi))
        exit(1)


_is_virtual = None
def is_virtual():
    global _is_virtual
    if _is_virtual is None:
        try:
            # lscpu | grep -i hypervisor
            p1 = subprocess.Popen(["lscpu"], stdout=subprocess.PIPE)
            p2 = subprocess.Popen(["grep", "-i", "hypervisor"], stdin=p1.stdout, stdout=subprocess.PIPE)
            p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
            output = p2.communicate()
            _is_virtual = output[0]
        except Exception as ex:  # hide the exception
            logging.exception(ex)
            _is_virtual = True
    return _is_virtual


def get_public_ip(timeout=(5,5)):
    try:
        return requests.get('https://checkip.amazonaws.com', timeout=timeout).text.strip()
    except:  # hide the exception
        return None


def get_lan_ip():
    rs = []
    for iface in netifaces.interfaces():
        if not iface.startswith('lo'):
            try:
                rs.append(netifaces.ifaddresses(iface)[netifaces.AF_INET][0]['addr'])
            except:
                pass
    return rs


def process_running(pid):
    """ Check For the existence of a unix pid. """
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


def can_lock_file(file_path):
    lock_marker_path = file_path + '.lock'
    can_lock = False
    if not os.path.isfile(lock_marker_path):
        can_lock = True
    else:
        with open(lock_marker_path, 'r') as f:
            pid = f.read()
            if not process_running(pid):
                can_lock = True
    return can_lock


def lock_file(file_path):
    lock_marker_path = file_path + '.lock'
    with open(lock_marker_path, 'w') as f:
        f.write(str(os.getpid()))
        f.flush()


def unlock_file(file_path):
    lock_marker_path = file_path + '.lock'
    os.remove(lock_marker_path)


# ==============================================================================
# endregion

# region Web
# ==============================================================================


def start_tornado(app, port, force_https=False, domain='localhost'):
    max_buffer_size = 1024 * 1024 * 1024  # 1Gb
    if force_https:
        # ==================== letsencrypt certificates
        cerbot_dir = '/etc/letsencrypt/live/{}'.format(domain)
        if os.path.isdir(cerbot_dir):
            certfile = "{}/fullchain.pem".format(cerbot_dir)
            keyfile = "{}/privkey.pem".format(cerbot_dir)
        else:
            certfile = "../../data/.certs/{}.crt".format(domain)
            keyfile = "../../data/.certs/{}.key".format(domain)
        settings = dict(
            ssl_options={
                "certfile": certfile,
                "keyfile": keyfile,
            },
            max_buffer_size=max_buffer_size,
        )

        http_server = tornado.httpserver.HTTPServer(tornado.wsgi.WSGIContainer(app), **settings)
    else:
        http_server = tornado.httpserver.HTTPServer(
            tornado.wsgi.WSGIContainer(app),
            max_buffer_size=max_buffer_size,
        )

    http_server.listen(port)
    logging.info("Starting Tornado server on {}:{}".format(domain, port))
    tornado.ioloop.IOLoop.instance().start()


def stop_tornado():
    tornado.ioloop.IOLoop.instance().stop()


def html_img_src(bgr, ext='jpg', w=-1, h=-1):
    """Creates an image embedded in HTML base64 format."""
    if bgr is not None:
        try:
            if w != -1 and h != -1:
                bgr = cv2.resize(bgr, (w, h))
            retval, buf = cv2.imencode('.' + ext, bgr)

            data = np.array(buf).tostring()
            data = base64.b64encode(data)
            data = data.decode('utf8').replace('\n', '')

            return 'data:image/{};base64,{}'.format(ext, data)
        except Exception as ex:
            logging.error('Fail to open embed image in html: '.format(ex))
    return ''


def html_img_src_to_bgr(img_src):
    if img_src.startswith('data:image/'):
        m = re.search(r'data:image/.+;base64,', img_src)
        if m:
            prefix = m.group()
            img_src = img_src[len(prefix):]
    return deserialize_bgr(img_src)


def html_img_src_from_path(path, w=-1, h=-1):
    if not path:
        return None

    if not os.path.isfile(path):
        return None

    try:
        bgr = cv2.imread(path)
        return html_img_src(bgr, w=w, h=h)
    except Exception as ex:
        logging.error('Fail to open image: {}. Error: {}'.format(path, ex))
    return None


def serialize_np(nparray):
    return base64.standard_b64encode(nparray)


def deserialize_np(str, dtype=np.float32):
    return np.frombuffer(base64.standard_b64decode(str), dtype=dtype)


def serialize_bgr(bgr, ext='jpg'):
    try:
        str = cv2.imencode('.' + ext, bgr)[1]
        return base64.b64encode(str)
    except Exception as ex:
        logging.error('Fail to serialize image: %s (%d x %d)', ex, bgr.shape[1], bgr.shape[0])
    return None


def deserialize_bgr(str):
    try:
        return cv2.imdecode(np.frombuffer(base64.b64decode(str.encode('utf8')), np.int8), cv2.IMREAD_COLOR)
    except Exception:
        logging.error('Fail to derialize image')
    return None


def fireforget(ws_url, data=None, json=None, headers=None):
    def _invoke_ws():
        try:
            if data is None and json is None:
                requests.get(ws_url, headers=headers, timeout=(0.1, 0.1))
            else:
                requests.post(ws_url, headers=headers, data=data, json=json, verify=False, timeout=(0.1, 0.1))
        except:
            pass
    threading.Thread(target=_invoke_ws).start()


def auth_header(user, pw):
    auth = '{}:{}'.format(user, pw)

    b64data = auth.encode('utf8')
    b64data = base64.b64encode(b64data)
    b64data = b64data.decode('utf8')
    return {'Authorization': 'Basic ' + b64data}

# ==============================================================================
# endregion

# region Camera
# ==============================================================================


def is_webcam_index(ip):
    try:
        int(ip)
        return True
    except:
        return False


def set_webcam_resolution(w, h):
    os.environ["WEBCAM_FARME_W"] = str(w)
    os.environ["WEBCAM_FARME_H"] = str(h)


def get_capture(video_url, gpu_id=0):
    if not os.environ.get("HW_DECODE"):
        #if isinstance(video_url, str) and video_url.startswith('rtsp'):
        #    if cv2.getBuildInformation().find('GStreamer') > 0:
        #        return cv2.VideoCapture(
        #            f'rtspsrc location={video_url} ! rtph264depay ! decodebin ! '
        #            f'videoconvert ! appsink max-buffers=1 drop=True sync=false',
        #            cv2.CAP_GSTREAMER)
        return cv2.VideoCapture(video_url)  # CPU decoding
    else:
        try:
            return cv2.vs_nvcuvid_reader(video_url, gpu_id)  # GPU decoding
        except:
            return cv2.VideoCapture(video_url)  # CPU decoding


def handle_cam_disconnected(cam_service_url, cap):
    """
    video capture blocks video writer so this function cannot be called frequently,
    must use big sleep_sec to limit the frequency when there are many disconnected cameras
    """
    if cap is not None:
        cap.release()

    if is_webcam_index(cam_service_url):
        if 'linux' in sys.platform.lower():
            cap = get_capture(cam_service_url)
            time.sleep(1)
            if not cap.isOpened():
                for i in range(6):
                    if 'video{}'.format(i) in os.listdir('/dev/'):
                        logging.info('Try to reconnect with webcam {}'.format(i))
                        cap = get_capture(i)
                        time.sleep(1)
                        if not cap.isOpened():
                            continue
                        else:
                            if os.environ.get("WEBCAM_FARME_W"):
                                cap.set(3, int(os.environ["WEBCAM_FARME_W"]))
                            if os.environ.get("WEBCAM_FARME_H"):
                                cap.set(4, int(os.environ["WEBCAM_FARME_H"]))
                            return cap
        else:
            cap = get_capture(0)
            return cap
    else:
        logging.info('Try to reconnect with {}'.format(cam_service_url))
        cap = get_capture(cam_service_url)

    return cap


# ==============================================================================
# endregion

# region I/O
# ==============================================================================
def load_rgb(path):
    try:
        cv_img = cv2.imread(os.path.expanduser(path))
        return to_rgb(cv_img)
    except:
        return None


def get_rgb_bgr(rgb=None, bgr=None, path=None):
    if rgb is None:
        if bgr is not None:
            rgb = to_rgb(bgr)
        elif path:
            rgb = load_rgb(path)

    if bgr is None and rgb is not None:
        bgr = to_bgr(rgb)

    return rgb, bgr


def read_lines(filename, encoding='utf8'):
    lines = []
    if os.path.isfile(filename):
        with codecs.open(filename, encoding=encoding) as f_in:
            for line in f_in:
                lines.append(line.replace('\n', '').replace('\r', ''))
    return lines


def write_lines(filepath, list_str, encoding='utf8'):
    with open(filepath, 'w', encoding=encoding) as f:
        f.writelines([s + '\n' for s in list_str])


def is_valid_image(path, fast=True):
    """fast==True: skip decoding image"""
    if not path:
        return False
    if not path.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        return False
    else:
        if fast:
            return True
        try:
            im = cv2.imread(path)
            if im is None:
                return False
            else:
                return True
        except:
            return False


# ==============================================================================
# endregion

# region Datetime
# ==============================================================================

def get_timestamp(time_obj=None):
    if time_obj is None:
        return int(time.time())
    elif isinstance(time_obj, datetime.datetime):
        return int(time.mktime(time_obj.timetuple()))
    elif isinstance(time_obj, int):
        return time_obj
    else:
        return int(time.mktime(time_obj))


def datetime_from_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(int(timestamp))


def strftimestamp(timestamp, format):
    return datetime_from_timestamp(timestamp).strftime(format)


def timestampfstr(datestr, datetime_format):
    return get_timestamp(time.strptime(datestr, datetime_format))


def limit_fps_by_sleep(max_fps, frame_start_time):
    if max_fps <= 0:
        time.sleep(1)
        return 1
    else:
        sleep_time = 1.0 / max_fps - max((time.time() - frame_start_time), 0)
        if sleep_time > 0:
            time.sleep(sleep_time)
        return sleep_time


# ==============================================================================
# endregion

# region Type conversion
# ==============================================================================
def to_int(str_val, fallback_val=None):
    """ fallback_val prevents exception to be raised, and is returned in case of exception """
    if str_val in ['True', 'true']:
        return 1
    if str_val in ['False', 'false', 'undefined', 'None', '']:
        return 0

    if fallback_val is not None:
        try:
            int_val = int(str_val)
        except:
            int_val = fallback_val
    else:
        int_val = int(str_val)
    return int_val


def to_utf8(the_str):
    return the_str


# ==============================================================================
# endregion

# region Collection math
# ==============================================================================
def flatten(nested_list):
    """Given a list, possibly nested to any level, return it flattened."""
    flat_list = []
    for item in nested_list:
        if type(item) == type([]):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def list_to_npArray(l):
    """convert the list to numpy array """
    if (type(l) == list) | (type(l) == tuple):
        l = np.array(l)
    return l


def list_npArray_to_npArray(ll):
    """
    fix the error: could not broadcast input array from shape ... into shape ...
    when casting list of np array to np array
    """
    rs = np.empty(len(ll), dtype=object)
    rs[:] = ll
    return rs


def filter(filter_function, array_of_np_array):
    keeps = filter_function(array_of_np_array)
    out_array_of_np_array = []
    for np_array in array_of_np_array:
        out_array = np_array[keeps]
        out_array_of_np_array.append(out_array)
    return out_array_of_np_array


def merge_npys(paths, merged_file):
    """concat npy files (list of emb) into merged_file"""
    rs = []
    for path in paths:
        try:
            embs = np.load(path)
            rs.extend(embs)
        except:
            pass
    np.save(merged_file, rs)


def most_common(lst, split_char=None):
    if not lst:
        return None
    if split_char:
        # treat each element string in list as list of lines.
        # e.g: ['30A\n1234'] -> [[30A, 1234]]
        # [::-1] revert list so that
        # last line is compared to last line not 1st line
        # in case list contains plates where 1st line is missing (openalpr fails), e.g
        # ['30A\n2345\nc', '2344\nb'] -> [[c, 2345, 30A], [b, 2344]]
        lst = np.array([e.split(split_char)[::-1] for e in lst])
        rs = []
        cols = max([len(lol) for lol in lst])
        for i in range(cols):
            lol = [l[i] for l in lst if i < len(l)]
            lol_most_common = Counter(lol).most_common(1)[0][0]
            # print('{} -> {}'.format(lol, lol_most_common))
            rs.append(lol_most_common)
        return split_char.join()

    return Counter(lst).most_common(1)[0][0]


def merge_dict_lists(list_list_dict):
    """ merge([[{'a': 1, 'b': 2}, {'a': 3, 'b': 4}], [{'c': 5}, {'c': 6}]])
    return: [{'a': 1, 'b': 2, 'c': 5}, {'a': 3, 'b': 4, 'c': 6}]
    """
    rs = list_list_dict[0]
    for list_dict in list_list_dict[1:]:
        for dict1, dict2 in zip(rs, list_dict):
            dict1.update(dict2)
    return rs
# ==============================================================================
# endregion

# region String
# ==============================================================================
import string
import unicodedata


def is_str(s):
    return isinstance(s, six.string_types)


def remove_punctuations(s):
    return s.translate(str.maketrans('', '', string.punctuation))


def vietnamese_to_ascii(s):
    s = s.replace(u'Đ', 'D')
    s = s.replace(u'đ', 'd')
    return unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode()

# ==============================================================================
# endregion

def alert(msg):
    mat = np.zeros((4, 500))
    cv2.imshow(msg, mat)
    cv2.waitKey(0)


def send_email(data, forget=False):
    def send_email_thread(data):
        try:
            toaddrs = data['to_addresses'].split(',')
            if not toaddrs or len(toaddrs) == 0:
                logging.error("send_email error: no toaddress")
                return False

            msg = "\r\n".join([
                "From: {}".format(data['username']),
                "To: {}".format(",".join(toaddrs)),
                "Subject: Automatic Notification (no reply)",
                "",
                to_utf8(data['message']),
            ])

            username = data['username']
            password = data['password']
            smtp_server_url = data['smtp_server_url']
            port = int(data['smtp_server_port'])

            server = None
            if port == 587:
                server = smtplib.SMTP(smtp_server_url, port, timeout=15)
                server.ehlo()
                server.starttls()
            elif port == 465:
                server = smtplib.SMTP_SSL(smtp_server_url, port, timeout=15)
                server.ehlo()

            if server is None:
                logging.error("send_email error: unsupported port: {}".format(port))
                return False
            else:
                server.login(username, password)
                server.sendmail(username, toaddrs, msg)
                server.quit()
            return True
        except Exception as ex:
            logging.exception(ex)
            return False

    if forget:
        threading.Thread(target=send_email_thread, args=(data,)).start()
        return None
    else:
        return send_email_thread(data)


_ip_re_pattern = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
_ip_port_re_pattern = re.compile(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\:(\d+)\/")


def is_ip(input):
    return _ip_re_pattern.match(str(input)) is not None


def to_ip(url):
    rs = re.findall(_ip_re_pattern, url)
    if len(rs) > 0:
        return rs[0]
    else:
        return None


def to_ip_port(url):
    """ rtsp://username:password@192.168.1.3:554/Streaming -> ('192.168.1.3', '554') """
    rs = re.findall(_ip_port_re_pattern, url)
    if len(rs) > 0:
        return rs[0]
    else:
        return None, None


def gps_dist_km(lat_deg1, lon_deg1, lat_deg2, lon_deg2):
    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat_deg1)
    lon1 = radians(lon_deg1)
    lat2 = radians(lat_deg2)
    lon2 = radians(lon_deg2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c  # in km

    return distance

def tpl_match(img_gray, tpl_gray, threshold=0.8):
    if tpl_gray is None or img_gray is None:
        return None, -1
    if tpl_gray.shape[0] > img_gray.shape[0]:
        return None, -1
    if tpl_gray.shape[1] > img_gray.shape[1]:
        return None, -1
    res = cv2.matchTemplate(img_gray, tpl_gray, cv2.TM_CCORR_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    if max_val > threshold:
        return max_loc, max_val
    else:
        return None, max_val


def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)