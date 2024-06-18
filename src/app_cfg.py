# -*- coding: utf-8 -*-
import os
import sys

from src.locale_def import *

# ===============================================================================
# Load user config
# ===============================================================================
sys.path.insert(0, os.path.abspath('../../config'))
from src.config import global_cfg


def get_user_config(entry_name, default_value):
    return getattr(global_cfg, entry_name, default_value)


USE_TPU = get_user_config('USE_TPU', False)
USE_TRACK = get_user_config('USE_TRACK', False)

CONTROLER_IP = get_user_config('CONTROLER_IP', 'localhost')
WEBAPP_CFG = get_user_config('WEBAPP_CFG', {})

ENGINE_VER = get_user_config('ENGINE_VER', '2.8')  # mobile int8
FACE_REG_TH = get_user_config('FACE_REG_TH', 0.4)  # threshold for FAR=10^-4?
FACE_REG_LOOSE_TH = get_user_config('FACE_REG_LOOSE_TH', 0.5)  # threshold for FAR=10^-3?

HUMAN_TRACK_KEEP_TH = get_user_config('HUMAN_TRACK_KEEP_TH', 0.5)
HUMAN_DET_TH = get_user_config('HUMAN_DET_TH', 0.4)

APP_TYPE = get_user_config('APP_TYPE', None)
COPYRIGHT = get_user_config('COPYRIGHT', None)
NORMALIZE_ID = get_user_config('NORMALIZE_ID', True)

HTTPS = get_user_config("HTTPS", True)
MAX_FPS = get_user_config("MAX_FPS", 10.0)
LOC = LOCALES[get_user_config("LOCALE", "vn")]

class AppsConfig(object):

    Copyright = {
        'facenet': {
            'COPYRIGHT': 'facenet.vn',
        }
    }

    @staticmethod
    def configure(app):
        cfg = AppsConfig.Copyright.get(COPYRIGHT, {})

        app.config['ADMIN_AVATAR'] = cfg.get('ADMIN_AVATAR', '')
        app.config['COPYRIGHT'] = cfg.get('COPYRIGHT', 'Qualcomm')
        app.config['APP_TITLE'] = cfg.get('APP_TITLE', 'AiHub')
        app.config['APP_TITLE_SHORT'] = cfg.get('APP_TITLE_SHORT', 'SC')
        app.config['LOGO'] = cfg.get('LOGO', '')
        app.config['MAX_CONTENT_LENGTH'] = 1024 * 1024 * 1024  # 1Gb
        app.config['DOMAIN'] = 'localhost'
        app.config['SHOW_LIVE'] = True
        app.config['lang'] = LOC

        for key, val in WEBAPP_CFG.items():
            app.config[key] = val
