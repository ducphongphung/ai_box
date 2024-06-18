import logging
import os

from src.global_def import *
from .. import app_cfg
import requests
import datetime
import werkzeug
from PIL import Image

from src.global_def import *
from src.utils.exifutil import open_oriented_im

from flask import request


def get_controller_ip():
    try:
        controller_ip = request.host.split(':')[0]
        if controller_ip == '127.0.0.1':
            controller_ip = app_cfg.CONTROLER_IP
    except:
        controller_ip = app_cfg.CONTROLER_IP
    return controller_ip


def make_download_link(path, controller_ip=None):
    if controller_ip is None:
        controller_ip = get_controller_ip()

    if not path:
        return 'not_found.html'

    if not path.startswith('http://'):
        return u'http://{}:{}/{}'.format(
            controller_ip, SERVICES[SV_FILE]['port'], os.path.abspath(path))
    else:
        if path.startswith('http://localhost'):
            path = 'http://{}'.format(controller_ip) + path[16:]
        return path


def return_json(user_msg='', ex=None, data=None, ret_code=-1):
    if ex:
        user_msg = '{}: {}'.format(user_msg, ex)
        logging.exception(ex)
    if ret_code in STATUS_CODES:
        user_msg = STATUS_CODES[ret_code]
    if data is None:
        data = {}
    data['message'] = user_msg
    if ret_code != SUCCESS_OK:
        data['err_code'] = ret_code
    return ut.to_utf8(json.dumps(data, indent=4, sort_keys=True, ensure_ascii=False)), 200


def get_params():
    try:
        params = request.json
    except:
        params = None

    if params is None:
        try:
            params = request.values
        except:
            params = None

    if params is None:
        raise Exception('error parsing request data')

    return params


def get_uploaded_image(save_img=0):
    """for compatibility"""
    filename = request.args.get('imagefile', '')
    bgr = None

    img_src = request.values.get('img_src', None)
    if img_src is None:
        img_src = request.values.get('face_img_src', None)

    if img_src:
        bgr = ut.html_img_src_to_bgr(img_src)
        try:
            if save_img:
                filename = "{}/{}.jpg".format(PATH_UPLOAD_DIR, int(time.time()))
                cv2.imwrite(filename, bgr)
        except:
            pass

        return filename, bgr

    if filename:
        if not filename.startswith('http'):
            filename = '/' + filename
        else:
            filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                        werkzeug.secure_filename(filename)
            filename_ = os.path.join(PATH_UPLOAD_DIR, filename_)
            r = requests.get(filename, stream=True, timeout=(1, 1))

            with open(filename_, "wb") as f:
                f.write(r.content)

            bgr = cv2.imread(filename_)
            return filename_, bgr

    if not filename or not os.path.exists(filename):

        imagefile = request.files['imagefile']
        filename_ = str(datetime.datetime.now()).replace(' ', '_') + \
                    werkzeug.secure_filename(imagefile.filename)
        filename = os.path.join(PATH_UPLOAD_DIR, filename_)
        im = open_oriented_im(imagefile)
        image_pil = Image.fromarray((255 * im).astype('uint8'))
        if image_pil.size[0] > MAX_UPLOAD_RES or image_pil.size[1] > MAX_UPLOAD_RES:
            image_pil.thumbnail((MAX_UPLOAD_RES, MAX_UPLOAD_RES), Image.ANTIALIAS)
        image_pil.save(filename)
        bgr = cv2.imread(filename)

    return filename, bgr
