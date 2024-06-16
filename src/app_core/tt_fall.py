import logging
import sys
sys.path.append('C:/Users\ducph\PycharmProjects/aibox')

from src.app_core.apps import VideoMonitorApp
from src.app_core.controller_utils import *
from src.app_core.conf import *
from src.utils.common import *

from flask import Flask, render_template, Response, request
import os
import cv2
import json
import time
import requests
from collections import deque
from shapely import geometry

logger = dbg.get_logger("tt_zone")

template_dir = os.path.abspath('templates')
static_dir = os.path.abspath('static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = os.urandom(64)

global previous_time


class RdSwitchStatus:
    FALLEN = 1

class Backend(VideoMonitorApp):
    def __init__(self):
        super(Backend, self).__init__()
        self.zones_cache = {}
        self.msg_throttler.max_age = 5
        self.previous_time = None
        self.tracks = deque(maxlen=global_cfg.FALL_DET_WINDOW_SIZE)

    def detect_batch(self, bgrs):
        rs = []
        for bgr in bgrs:
            bboxes = []  # run detection to get bboxes

            dets = ObjDets([ObjDet(bb) for bb in bboxes])
            regs = ObjRegs(aObjDets=dets)
            rs.append(regs)
        return rs

    def get_detections(self, frame):
        detections = []
        reg = None
        try:
            rs = requests.post(
                'http://localhost:9769/api/detect',
                json={
                    'img_src': ut.html_img_src(frame),
                    'det_type': 'fall'
                }, verify=False)

            if rs.status_code == 200:
                reg = json.loads(rs.content)
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to inference service")
        if reg:
            detections = reg['detections']

        return detections

    def send_fall_events(self, detections):
        # jpg_as_text = ''  # try some others ways to send detect image to HC not using base64 encoded
        try:
            # rdhc_api.send_event_fall_hc(RdSwitchStatus.FALLEN, jpg_as_text, cam_id=self.cam_id)
            self.tracks.clear()
        except:
            self.hc_connected = False


    def set_zone_cache(self):
        try:
            new_zones_cache = {}
            for zone in self.zones:
                zone_id = zone['zone_id']
                if zone_id in self.zones_cache:
                    new_zones_cache[zone_id] = self.zones_cache[zone_id]
                else:
                    new_zones_cache[zone_id] = zone.copy()
                    new_zones_cache[zone_id]['det_seq'] = deque(maxlen=global_cfg.HUMAN_DET_TIME_WINDOW_SIZE)

                self.zones_cache = new_zones_cache
        except Exception as ex:
            logger.exception(ex)

    def send_fall_passing_events(self, detections, frame=None):
        try:
            self.send_fall_events(detections)
        except Exception as ex:
            self.hc_connected = False
            logger.exception(ex)

    def send_fall_event_not_in_zone(self, detections, image):
        try:
            new_zones_cache = {}
            for zone in self.zones:
                zone_id = zone['zone_id']
                if zone_id in self.zones_cache:
                    new_zones_cache[zone_id] = self.zones_cache[zone_id]
                else:
                    new_zones_cache[zone_id] = zone.copy()
                    new_zones_cache[zone_id]['det_seq'] = deque(maxlen=global_cfg.HUMAN_DET_TIME_WINDOW_SIZE)

            self.zones_cache = new_zones_cache
            # build the detection sequence for each zone, the seq contains detections from 9 past frames
            for zone_id, zone in self.zones_cache.items():
                if zone['zone_attributes']['164'] == 0:
                    zone_poly, zone_poly_expanded = expand_zone(zone["coords"])
                    zone_has_motion = False
                    # center of body inside zone_poly_expanded -> zone has detection
                    for bb in detections:
                        pts = [[bb[0], bb[1]], [bb[2], bb[1]], [bb[2], bb[3]], [bb[0], bb[3]]]  # bb to 4 points
                        if zone_poly.intersects(geometry.Polygon(pts)):
                            zone_has_motion = True

                    try:
                        # for each zone, apply the rules to fire events to HC to turn on/off zone-switch
                        if not zone_has_motion:
                            self.send_fall_passing_events(detections, frame= image)

                        else:  # keep current switch status if detection unstable: <= 33% detection
                            logger.warn(f"zone: {zone}: skip sending zone event to HC as detection is not stable")
                    except:
                        self.hc_connected = False
                else:
                    self.send_fall_passing_events(detections, frame= image)
        except Exception as ex:
            logger.exception(ex)

    def process_frame(self, frame, t0, regs, freeze_state):
        show = frame.copy()

        detections = self.get_detections(frame)

        if len(detections):
            for d in detections:
                bb = d['bb']
                dr.draw_box(show, bb, line1="FALLEN"  if d['is_fallen'] == 1 else None,
                            color=(0, 0, 255) if d['is_fallen'] == 1 else None, simple=True)
                if d['is_fallen'] == 1:
                    self.tracks.append(1)

        current_time = time.time()
        # Default passing to 60s
        if self.previous_time is None:
            self.previous_time = current_time
        elif self.previous_time is not None and current_time - self.previous_time > 7 and sum(self.tracks) >= 30:
            self.previous_time = current_time
            self.send_fall_event_not_in_zone(detections=detections, image = show)
            self.tracks.clear()
        return show

    @property
    def zones(self):
        rs = self.conf.get(self.cam_id, {}).get('zone', [])
        if not isinstance(rs, list):
            return []
        else:
            return rs

    def on_conf_update(self, frame):
        super(Backend, self).on_conf_update(frame)
        self.conf['count_margin'] = 0
        self.conf['stopline_y'] = 0
        self.conf = read_json_conf()
        self.set_zone_cache()

    def draw_static_info(self, disp):
        super(Backend, self).draw_static_info(disp)
        for zone in self.zones_cache.values():
            color = dr.RED if zone.get('light') else dr.GREEN
            dr.draw_poly(disp, zone['coords'], zone['zone_name'], color=color)

    def add_cli_opts(self):
        super(Backend, self).add_cli_opts()
        self.parser.add_option('--n_frame', type=int, default=20)

def render_live(err=''):
    return render_template('live.html', current_user='Anonymous', err=err,
                           input_url=backend.input_url if backend.input_url else '',
                           conf_url=backend.conf_url, conf_json=json.dumps(backend.conf, indent=4),
                           sys_info=backend.sys_info())


@app.route('/')
@app.route('/live', methods=['GET', 'POST'])
def live():
    try:
        err = ''
        if request.method == 'POST':
            err = backend.update_input()
        return render_live(err=err)
    except Exception as ex:
        return render_live(err=str(ex))


def gen():
    while True:
        frame = backend.frame_stream
        if frame is not None:
            ret, encoded = cv2.imencode('.jpg', frame)
            frame_mjpeg = encoded.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_mjpeg + b'\r\n\r\n')

        time.sleep(0.1)  # broadcast at 10fps max


@app.route('/video_feed_mjpeg', methods=['GET'])
def video_feed_mjpeg():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/add_cam_v2', methods=['POST'])
def add_cam_v2():
    try:
        camera_infos = read_json_conf()
        all_cams = request.json
        for infos in all_cams:
            username, password, ip_address, url = cam_url_to_info(infos['cam_url'], infos['cam_type'])
            cam_info = {
                "id": infos['id'],
                "cam_ip": infos.get('cam_ip', ''),
                "cam_name": infos.get('cam_name', ''),
                "cam_type": infos.get('cam_type', ''),
                "cam_url": url,
                "zone": infos.get('zone', [])
            }
            camera_infos[str(infos['id'])] = cam_info
            write_cam_info_json(info=camera_infos)
        return return_json("Thành công", data=camera_infos, ret_code=0)
    except Exception as ex:
        return return_json("Lỗi.", ex, ret_code=1)


@app.route('/camera-zone/<camera_ip>/<zone_id>', methods=['DELETE'])
def delete_zone_by_id(camera_ip, zone_id):
    try:
        camera_jsons = read_json_conf()
        if camera_ip not in camera_jsons:
            return return_json(f"Lỗi. Không tồn tại camera: ID : {camera_ip}", ret_code=2)

        zones = camera_jsons[camera_ip].get('zone', [])
        deleted = False

        # Tìm và xóa zone dựa trên zone_id
        for zone in zones:
            if zone.get('zone_id') == zone_id:
                zones.remove(zone)
                deleted = True
                break

        if deleted:
            write_cam_info_json(info=camera_jsons)
            return return_json(f"Thành công. Đã xóa zone có ID {zone_id}", data=camera_jsons[camera_ip], ret_code=0)
        else:
            return return_json(f"Lỗi. Không tìm thấy zone có ID {zone_id} trong camera {camera_ip}", ret_code=3)
    except Exception as ex:
        return return_json("Lỗi.", ex, ret_code=1)


@app.route('/camera-zone/<camera_id>', methods=['GET'])
def get_camera_zone(camera_id):
    try:
        camera_jsons = read_json_conf()
        if camera_id not in camera_jsons:
            return return_json(f"Lỗi. Không tồn tại camera: ID {camera_id}", ret_code=2)
        return return_json("Thành công", data=camera_jsons[camera_id], ret_code=0)
    except Exception as ex:
        return return_json("Lỗi.", ex, ret_code=1)


@app.route('/camera-info', methods=['GET'])
def get_camera_info():
    try:
        camera_info = list(read_json_conf().values())

        return return_json("Thành công", data={"camera_info": camera_info}, ret_code=0)
    except Exception as ex:
        return return_json("Lỗi.", ex, ret_code=1)


@app.route('/synch_cam_zone', methods=['POST'])
def synch_cam_zone():
    try:
        camera_infos = read_json_conf()
        all_cams = request.json
        for infos in all_cams:
            camera_infos[str(infos['id'])] = infos
            write_cam_info_json(info=camera_infos)
        return return_json("Thành công", data=camera_infos, ret_code=0)
    except Exception as ex:
        return return_json("Lỗi.", ex, ret_code=1)


@app.route('/camera-conf', methods=['POST'])
def conf_camera():
    try:
        camera_zone = read_json_conf()
        infos = request.json
        cam_id = infos['cam_id']
        camera_info = camera_zone.get(cam_id, {})
        cam_url = camera_info.get('cam_url')
        cam_ip = camera_info.get('cam_ip')
        if not cam_ip:
            return return_json("Lỗi. Không tìm thấy IP của camera", ret_code=4)
        # Kiểm tra ping đến IP của camera
        if not ping(cam_ip):
            return return_json("Lỗi. Không ping được camera IP", ret_code=2)

        err = backend.update_cam(cam_url, infos['conf_url'], cam_id)

        if err:
            return return_json("Lỗi. Không kích hoạt được camera.", ret_code=3)
        else:
            return return_json("Thành công", ret_code=0)
    except Exception as ex:
        return return_json("Lỗi. ", ex, ret_code=1)

def expand_zone(coords, factor=0.3):
    xs = [i[0] for i in coords]
    ys = [i[1] for i in coords]
    x_center = 0.5 * min(xs) + 0.5 * max(xs)
    y_center = 0.5 * min(ys) + 0.5 * max(ys)
    min_corner = geometry.Point(min(xs), min(ys))
    center = geometry.Point(x_center, y_center)
    distance = center.distance(min_corner) * factor
    zone = geometry.Polygon(coords)
    zone_with_border = zone.buffer(distance)
    return zone, zone_with_border

if __name__ == '__main__':

    AppsConfig.configure(app)

    backend = Backend()

    backend.resume()

    app.config['APP_TITLE_SHORT'] = 'FTR'

    # Run your Flask application
    app.run(host='0.0.0.0', port=8081, debug=False, threaded=True)
