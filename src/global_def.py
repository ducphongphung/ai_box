# -*- coding: utf-8 -*-

# ===============================================================================
# This file should contain configurations those are not likely to change during deployment
# but are likely to change in new software release,
# also the enlisted configurations should apply for many different applications
# ===============================================================================

import sys

MAX_INT = sys.maxsize

import logging
import urllib3 as urllib
import os
import optparse
import numpy as np
import time
import operator
from collections import deque
import cv2
import json

from src.utils.types_ex import *
from src.utils import common as ut
from src.utils import debug as dbg
from src.utils import draw as dr

from src.app_cfg import *

try:
    urllib.disable_warnings(urllib.exceptions.InsecureRequestWarning)
except:
    pass


# ===============================================================================
# App settings
# ===============================================================================
ENABLE_MOTION_DETECTION = True
MIN_MOTION_AREA_RATIO = 0.01  # of the image area

# timeout for cache of slow & frequent functions like:
# get_profile_image, functions that require database connection...
STD_CACHE_TIMEOUT_SEC = 5
# timeout cua request den LAN services: door service, display service...
STD_SERVICE_TIMEOUT_SEC = (0.1, 0.1)  # (connect, read)
STD_HTTPS_SERVICE_TIMEOUT_SEC = (1, 1)  # (connect, read)
INTERNET_SERVICE_TIMEOUT_SEC = (10, 10)

LOG_RETENTION_DAY = 31
LOG_RETENTION_DICT = {
    'events': 93,
    'tasks': 3,
}
LOG_VIDEO_RESOLUTION = (1280, 720)
LOG_FORMAT = '%(asctime)s %(levelname)s %(message)s'

MAX_UPLOAD_RES = 1600

logging.basicConfig(level=logging.INFO, format=str(LOG_FORMAT))

DATETIME_FORMAT = '%Y-%m-%d %H:%M'
DATE_FORMAT = '%Y-%m-%d'
TIME_HM_FORMAT = '%H:%M'
TIME_HMS_FORMAT = '%H:%M:%S'
TIME_FULL_FORMAT = '%Y-%m-%d %H:%M:%S'

MAINTENANCE_TIME = (2, 3)  # (from, to), avoid operational time

# ===============================================================================
# Paths, IPs
# ===============================================================================
PATH_LOG_DIR = '../../src/log/'
PATH_TASK_DIR = PATH_LOG_DIR + 'tasks/'  # queued tasks: "f_": forward task
PATH_SCHEDULED_TASK_DIR = PATH_TASK_DIR + 'scheduled/'

PATH_DATA_DIR = '../../data/'
PATH_MODEL_DIR = PATH_DATA_DIR + 'models/'
PATH_DB_DIR = '{}/facedb/'.format(PATH_DATA_DIR)
PATH_UPLOAD_DIR = PATH_DATA_DIR + 'upload/'
PATH_SNAPSHOT = PATH_DATA_DIR + 'snapshots/'

# ===============================================================================
# Interface services
# ===============================================================================
SV_DOOR = 0
SV_PI_DISPLAY = 1
SV_PC_DISPLAY_CAM = 2
SV_PC_DISPLAY_VIDEO = 3
SV_CAM_HK = 4  # hkivision camera
SV_CAM_KB = 5  # kbvision camera
SV_WEBCAM = 6
SV_WEBCAM_HD = 7
SV_FILE = 8
SV_TICKET = 9
SV_AUTO_INDEX = 10
SV_CONTROLLER = 11
SV_CAM_VC = 12
SV_ALARM = 13
SV_DOOR_ERROR = 14
SV_DOOR_HEALTH = 15
SV_FACEDB = 16
SV_APPDB = 17
SV_FA = 18
SV_CAM_DA = 19
SV_CAM_SN = 20
SV_DATA_HUB = 21
SV_PORT_MANAGER = 22
SV_BW_SUBSCRIBE_HTTP = 23

SERVICES = {
    SV_ALARM: {'url': 'alarm_service', 'port': 8081},
    SV_DOOR: {'url': 'door_service', 'port': 8081},
    SV_DOOR_ERROR: {'url': 'door_service_error', 'port': 8081},
    SV_DOOR_HEALTH: {'url': 'door_service_health', 'port': 8081},
    SV_PI_DISPLAY: {'url': 'display', 'port': 5000},
    SV_PC_DISPLAY_CAM: {'url': 'display_cam', 'port': 8082},
    SV_PC_DISPLAY_VIDEO: {'url': 'display_video', 'port': 8082},
    SV_CAM_HK: {'url': 'rtsp://admin:abcd1234@{}:554/Streaming/Channels/1'},
    SV_CAM_KB: {'url': 'rtsp://admin:abcd1234@{}:554/cam/realmonitor?channel=1&subtype=0'},
    SV_CAM_DA: {'url': 'rtsp://admin:admin@{}:554/cam/realmonitor?channel=1&subtype=0'},
    SV_CAM_VC: {'url': 'rtsp://admin:1111@{}:554/live/ch0'},
    SV_CAM_SN: {'url': 'rtsp://admin:abcde123456@{}:554/profile2/media.smp'},
    SV_FILE: {'url': 'put_file', 'port': 8083},
    SV_TICKET: {'url': 'ticket', 'port': 8084},
    SV_AUTO_INDEX: {'url': '', 'port': 8085},
    SV_CONTROLLER: {'url': '', 'port': 8080},
    SV_FACEDB: {'url': '', 'port': 8002},
    SV_APPDB: {'url': '', 'port': 8086},
    SV_FA: {'url': 'fa', 'port': 8087},
    SV_DATA_HUB: {'url': '', 'port': 8864},
    SV_PORT_MANAGER: {'url': '', 'port': 9998},
    SV_BW_SUBSCRIBE_HTTP: {'port': 8865},
}


def get_ip_from_suffix(ip_suffix):
    if ip_suffix == 'localhost':
        return ip_suffix
    if ip_suffix is None:
        return 'localhost'
    #  if ip suffix is actually a full ip (with 4 slots)
    if ut.is_ip(ip_suffix):
        return ip_suffix
    if ip_suffix == 0 or ip_suffix == '0':
        return 0

    return None


def get_cam_ip(ip_suffix=None):
    ip = get_ip_from_suffix(ip_suffix)
    if ip is None:
        return 0  # opencv webcam ip
    return ip


def get_server_ip(ip_suffix=None):
    ip = get_ip_from_suffix(ip_suffix)
    if ip is None:
        return 'localhost'
    return ip


def get_cam_service_url(cam_service_id=None, ip=None, ip_suffix=None):
    if cam_service_id:  # if cam_service_id is not (None or '')
        if cam_service_id in [SV_WEBCAM, SV_WEBCAM_HD]:
            return 0  # handle invalid webcam ip
        else:
            if not ip:
                if ip_suffix:
                    ip = get_ip_from_suffix(ip_suffix)
            if not ip:
                raise ValueError("Camera IP not specified.")

            return SERVICES[cam_service_id]['url'].format(ip)
    else:  # if cam_service_id is None or '' -> try to guess service type by ip
        if ut.is_webcam_index(ip):
            return int(ip)

        if ip:
            if ip.startswith('webcam'):
                return 0
            else:
                return SERVICES[SV_CAM_HK]['url'].format(ip)

        raise ValueError("Missing input param: cam_service_id")


def get_http_service_url(service_id, ip=None, ip_suffix=None):
    sv = SERVICES[service_id]
    if not ip:
        ip = get_server_ip(ip_suffix)
    return 'http://{}:{}/{}'.format(ip, sv['port'], sv['url'])


# ===============================================================================
# Camera ROIs
# ===============================================================================
def get_default_roi(img_w=1920, img_h=1080, margin_rates=(1. / 14, 1. / 12), offset_rates=(0., 0.)):
    margins = (int(img_w * margin_rates[0]), int(img_h * margin_rates[1]))
    offsets = (int(margins[0] * offset_rates[0]), int(margins[1] * offset_rates[1]))

    return (margins[0] + offsets[0], margins[1] + offsets[1]), \
           (img_w - margins[0] + offsets[0], img_h - margins[1] + offsets[1])


# ===============================================================================
# Special Strings
# ===============================================================================
STR_UNKNOWN = "Unknown"
STR_PREFIX_DISTRACT = "distract_"
STR_PREFIX_DELETED = "deleted_"
STR_SUFFIX_PROFILE = "_profile.jpg"
STR_FOURCC = str("XVID")
STR_ID_RENDERED = "_id_rendered.avi"
STR_SEPARATOR = '_0S0_'

STR_OUT_JPG = "_out.jpg"


def must_ignore_identity(identity):
    if not identity:
        return True
    return identity.startswith(STR_PREFIX_DISTRACT) or \
           identity.startswith(STR_PREFIX_DELETED) or \
           identity.startswith('.')


def is_valid_identity(identity):
    return identity != STR_UNKNOWN and (not must_ignore_identity(identity))


def normalize_id_field(text):
    if not NORMALIZE_ID:
        return text

    if text:
        id = text.strip().replace(' ', '_')
        return ''.join([i if ord(i) < 128 else '-' for i in id])
    else:
        raise ValueError("id value cannot be empty")


def validate_id_field(text):
    if not text:
        raise ValueError('id value cannot be empty')
    only_alnum = text.replace('-', '').replace('_', '')
    if not only_alnum.isalnum():
        raise ValueError('id must contain only alphanumeric, hyphen (-) and underscore (_)')

# ===============================================================================
# Face Registration Status code definitions
# ===============================================================================
STATUS_CODES = {
    200: LOC['SUCCESS_OK'],
    420: LOC['SMALLFACE'],
    421: LOC['SMALLIMAGE'],
    422: LOC['MULTIFACE'],
    423: LOC['MATCHFACE'],
    424: LOC['NOFACE'],
    425: LOC['INCLINEDFACE'],
    426: LOC['UNREALFACE'],
    500: LOC['ONLYJPG'],
}

SUCCESS_OK = 200
MULTIFACE = 422
NOFACE = 424
SMALLFACE = 420
SMALLIMAGE = 421
MATCHFACE = 423
NOFRONTAL = 425
UNREALFACE = 426
ERROR = 500

# ===============================================================================
# Photo requirements
# ===============================================================================
FACE_MINSIZE = 160
IMAGE_MINSIZE = 160
MUGSHOT_THRESHOLD = 0.25
MUGSHOT_THRESHOLD_LOOSE = 0.4

# ===============================================================================
# global vars
# ===============================================================================
g_shutdown = False  # for shutting-down all threads
