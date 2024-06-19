# -*- coding: utf-8 -*-
import logging
import optparse
import os

from flask import Flask

import sys
# Change the path to folder ai_box
sys.path.append('/home/quangthangggg/Documents/ai-box2/ai_box')

import src.utils.common as ut
from src.app_core.controller_utils import get_params, return_json
from src.cv_core.fall.FallDetector import FallDetector
from src.cv_core.fire.FireDetector import FireDetector
from src.cv_core.family.FamilyDetector import FamilyDetector
from src.app_cfg import AppsConfig

template_dir = os.path.abspath('templates')
static_dir = os.path.abspath('static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = os.urandom(64)



# region WS API
# ==============================================================================

@app.route('/api/detect', methods=['POST'])
def api_detect():
    params = get_params()

    try:
        bgr = ut.html_img_src_to_bgr(params['img_src'])
    except Exception as ex:
        return return_json('Error parsing required parameter: img_src', ex)

    try:
        det_type = params['det_type']
    except Exception as ex:
        return return_json('Error parsing required parameter: det_type', ex)

    try:
        rs = []
        if det_type == 'fall':
            rs = g_fdet.get_fall(bgr).records  # TODO: Confirm to return an objDets
            # return return_json('ok', data={"detections": rs})  # TODO: remove this line if return an objDets
        if det_type == 'fire':
            rs = g_firedet.get_fire(bgr).records
        if det_type == 'family':
            rs = g_familydet.get_stranger(bgr).records
        return return_json('ok', data={"detections": [r.to_json() for r in rs]})
    except Exception as ex:
        return return_json('', ex)



if __name__ == '__main__':
    parser = optparse.OptionParser()

    parser.add_option('--port', help='port to run this program, default=9769', type='int', default=9769)
    parser.add_option('--max_face', help='max number of recognized faces in a frame, default=2', type='int', default=2)

    opts, args = parser.parse_args()
    g_fdet = FallDetector()
    g_firedet = FireDetector()
    g_familydet = FamilyDetector()
    timestamp_ms = 0
    AppsConfig.configure(app)
    # avoid polluting log file with "200 INFO POST /api/..."
    logging.getLogger("tornado.access").setLevel(logging.WARNING)

    # Thread(target=zmq_sink_thread).start()

    ut.start_tornado(app, opts.port)
