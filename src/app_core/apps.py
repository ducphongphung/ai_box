import time

from src.global_def import *
from src.app_core.cameras import FrameGrabber
from src.app_core.loggers import VideoLogger
from src.config.global_cfg import *
from threading import Thread
from flask import request
import os
import json
from queue import Queue
import cv2
from collections import deque

logger = dbg.get_logger(__name__)


class AppBase(object):
    def __init__(self):
        self.port = 8081
        self.hc_ip = ''
        self.hc_pass = ''
        self.hc_mac = ''
        self.hc_next_retry = MAX_INT
        self.hc_retry_wait = 10
        self.conf = {}
        self.conf_url = ''
        self.conf_has_update = False
        self.cam_id = ''
        self.input_url = None
        self.app_name = os.path.basename(sys.argv[0])
        self.msg_throttler = ExpiringDict(max_age_seconds=2, max_len=100, items={})

    def _init(self):
        pass

    def _update_input_url(self, val):
        if self.input_url != val:
            self.input_url = val
            self._init()
            self.on_input_update()

    def _update_hc(self, hc_ip_val,hc_pass_val,hc_mac_val):
        if self.hc_ip != hc_ip_val or self.hc_pass != hc_pass_val or self.hc_mac != hc_mac_val:
            self.hc_ip = hc_ip_val
            self.hc_pass = hc_pass_val
            self.hc_mac = hc_mac_val
            self._init()
            self.on_hc_update()

    def _update_cam_id(self,cam_id_val):
        if self.cam_id != cam_id_val:
            self.cam_id = cam_id_val
            self._init()
            self.on_cam_id_update()

    def update_input(self):
        input_url = request.values['input_url']

        conf_url = request.values.get('conf_url', '')
        if not conf_url and os.path.exists(input_url + '.json'):
            conf_url = input_url + '.json'

        conf_json = request.values.get('conf_json', '')
        if conf_url == self.conf_url and \
                os.path.exists(conf_url) and \
                conf_url.startswith(SYS_PATH):
            if not conf_json:
                with open(conf_url, 'r') as f:
                    conf_json = f.read()
            try:
                conf_json = json.loads(conf_json)
                for k, v in request.values.items():
                    if k.startswith('th_'):
                        conf_json[k] = v
                with open(conf_url, 'w') as f:
                    json.dump(conf_json, f)
            except Exception as ex:
                logger.exception(ex)

        self._update_input_url(input_url)
        err = self._update_conf(conf_url)

        self._save_states()

        return err

    def update_hc(self, hc_ip, hc_pass, hc_mac):
        self._update_hc(hc_ip, hc_pass, hc_mac)
        self._save_states()

    def update_cam(self, input_url, conf_url, cam_id):
        self._update_input_url(input_url)
        self._update_cam_id(cam_id)
        err = self._update_conf(conf_url)
        self._save_states()
        return err

    def update_zone(self, conf_json_zone):
        input_url = self.input_url

        conf_url = self.conf_url
        if not conf_url and os.path.exists(input_url + '.json'):
            conf_url = input_url + '.json'

        conf_json = conf_json_zone
        if conf_url == self.conf_url and \
                os.path.exists(conf_url) and \
                conf_url.startswith(SYS_PATH):
            if not conf_json:
                with open(conf_url, 'r') as f:
                    conf_json = f.read()
            try:
                conf_json = json.loads(conf_json)
                for k, v in request.values.items():
                    if k.startswith('th_'):
                        conf_json[k] = v
                with open(conf_url, 'w') as f:
                    json.dump(conf_json, f)
            except Exception as ex:
                logger.exception(ex)

        self._update_input_url(input_url)
        err = self._update_conf(conf_url)
        self._save_states()
        return err

    def _update_conf(self, json_url):
        """
        sample content:
        {
            "stopline_y": 0,
            "traffic_light": {
                "red": [1858, 24],
                "yellow": [1856, 95],
                "green": [1865, 176]
            },
            "brt_lane": [1114, 518, 1920, 1080]
        }
        """
        err = ''
        try:
            if json_url.startswith('http'):
                json_str = urllib.urlopen(json_url).read()
                print(json_str)
                self.conf = json.loads(json_str)
            else:
                if json_url.startswith(SYS_PATH):
                    if os.path.isfile(json_url):
                        with open(json_url) as f:
                            self.conf = json.load(f)
                    else:
                        logger.warn('no file at: {}, create new'.format(json_url))
                        try:
                            with open(json_url, 'w') as f:
                                json.dump(self.conf, f)
                        except Exception as ex:
                            logger.exception(ex)
                            err = 'cannot create config file: {}'.format(ex)
                else:
                    err = 'conf file must be in /sc/sources'
                    logger.error(err)
        except Exception as ex:
            err = str(ex)
            logger.exception(ex)
        if err == '':
            self.conf_url = json_url
            self.conf_has_update = True  # async update
        else:
            err = 'Fail to update. Config rolled-back. Exception: {}'.format(err)
        return err

    def on_conf_update(self, frame):
        pass

    def on_input_update(self):
        pass

    def on_hc_update(self):
        pass

    def on_cam_id_update(self):
        pass

    @property
    def hc_connected(self):
        """
        automatically returns to True after some time to trigger connection retry
        """
        if self.hc_next_retry == MAX_INT:
            return True
        else:
            return time.time() - self.hc_next_retry > 0

    @hc_connected.setter
    def hc_connected(self, val: bool):
        if val is False:
            self.hc_next_retry = time.time() + self.hc_retry_wait
        else:
            self.hc_next_retry = MAX_INT

    def _save_states(self):
        file = 'C:\\Users\ducph\PycharmProjects\\ai_box\setup\sc\8081.json'
        with open(file, 'w') as f:
            json.dump({
                'input_url': self.input_url,
                'conf_url': self.conf_url,
                'hc_ip': self.hc_ip,
                'hc_pass': self.hc_pass,
                'hc_mac': self.hc_mac,
                'cam_id': self.cam_id
            }, f)

    def resume(self):
        if not self.input_url:
            file = '/home/quangthangggg/Documents/ai-box2/ai_box/setup/sc/8081.json'
            states = {}
            try:
                if os.path.isfile(file):
                    with open(file, 'r') as f:
                        states = json.load(f)
            except Exception as ex:
                logger.exception(ex)
            self._update_input_url(states.get('input_url', ''))
            self._update_conf(states.get('conf_url', ''))
            self._update_hc(states.get('hc_ip', ''), states.get('hc_pass', ''),states.get('hc_mac',''))
            self._update_cam_id(states.get('cam_id',''))


class VideoMonitorApp(AppBase):
    def __init__(self):
        super(VideoMonitorApp, self).__init__()

        self.last_check_cl = 0
        self.log_notify_time = {}

        self.cam_type = None
        self.cam_ip = None
        self.cam_port = None
        self.thread = None
        self.should_stop = False

        self.min_motion = MIN_MOTION_AREA_RATIO
        self.motion_detector = None
        self.motion_roi = Roi(tl=(0, 0), br=(0, 0))
        self.motion_ratio = 1
        self.motion_scale = 1
        self.motion_mask = None
        self.stop_update_bg_at = 0
        self.stop_update_bg_duration = 3  # second

        self._has_detection = False
        self._has_gui = os.environ.get('DISPLAY', None)
        if self._has_gui:
            logger.info("HAS GUI")

        self.direction = None
        self.roi = None

        self.tracker = None  # to be defined by actual app
        self.det = None
        self.video_logger = None
        self.frame_grabber = None
        self.frame_stream = None

        # debug
        self.fps = 0.0  # average since start
        self.delay = {'all': 0.0, 'det': 0.0, 'track': 0.0}  # moving average
        self.start_time = time.time()
        self.processed_frame_cnt = 0

        self.parser = optparse.OptionParser()
        self.add_cli_opts()
        self.opts = self.parser.parse_args()
        self.get_cli_opts()

        self.det_batch_size = self.opts.batch_size
        self.det_q = Queue(maxsize=self.det_batch_size)

        self.log_dir = self.opts.log_dir

    def _update_cam_ip_port(self):
        if ut.is_webcam_index(self.input_url):
            self.cam_ip = 'webcam{}'.format(self.input_url)
            self.cam_port = None
        else:
            self.cam_ip, self.cam_port = ut.to_ip_port(self.input_url)

    # region virtual methods

    def _init(self):
        if self.video_logger:
            self.video_logger.stop()
        if self.frame_grabber:
            self.frame_grabber.stop()

        self.start_time = time.time()
        self.processed_frame_cnt = 0

        try:
            if self.input_url is None:
                self.input_url = get_cam_service_url(ip=self.cam_ip)
            self._update_cam_ip_port()
        except Exception as ex:
            logger.exception(ex)

        self.video_logger = VideoLogger(
            self.cam_ip, autosave=True,
            lossless=self.flg_write_raw,
            rendered=self.flg_write_rendered, timestamp=self.opts.write_ts,
            log_dir=self.log_dir, cam_port=self.cam_port)
        self.frame_grabber = FrameGrabber(
            self.cam_ip, self.input_url, self.cam_type,
            im_transform_fn=None, fps=self.grab_fps)
        self.motion_detector = cv2.createBackgroundSubtractorMOG2()

        self.should_stop = False

    def on_conf_update(self, frame):
        if self.stopline_y < 0:
            self.conf['stopline_y'] = frame.shape[0] * 2 / 3
        if self.count_margin < 0:
            self.conf['count_margin'] = 135
        if self.left < 0:
            self.conf['left'] = 0
        if self.right < 0:
            self.conf['right'] = frame.shape[0]

        self.roi = Roi(tl=(self.left, self.stopline_y - self.count_margin),
                       br=(self.right, self.stopline_y + self.count_margin))

    # endregion

    @property
    def has_detection(self):
        return self._has_detection

    @has_detection.setter
    def has_detection(self, val):
        if val is True:
            # do not update background for X second when there is detection
            self.stop_update_bg_at = time.time()
        self._has_detection = val

    @property
    def frame_id(self):
        return self.frame_grabber.frame_id

    def stop(self):
        self.should_stop = True
        if self.thread and self.thread.is_alive():
            self.thread.join()

        logger.info("{} gracefully stopped".format(self.__class__.__name__))

    def get_frame(self):
        return self.frame_grabber.get_frame()

    def detect_batch(self, bgrs):
        if self.det is None:
            return [None] * len(bgrs)

        if isinstance(self.det, list):
            batch_regs = [ObjRegs() for i in range(len(bgrs))]
            for det in self.det:
                regs = det.get_ObjRec()
                for i in range(len(bgrs)):
                    batch_regs[i].records.extend(regs[i].records)
        else:
            batch_regs = self.det.get_ObjRec()

        return batch_regs

    def run_detection_thread(self):
        batch = deque(maxlen=self.det_batch_size)
        while not self.should_stop:
            ts = time.time()
            try:
                if self.det_q.maxsize - self.det_q.qsize() >= self.det_batch_size:
                    frame, t0 = self.get_frame()
                    if frame is not None:
                        if self.opts.work_w:
                            frame = ut.resize(frame, dst_w=self.opts.work_w)
                        if 'rot90' in self.conf:
                            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                        if self.conf_has_update:
                            self.on_conf_update(frame)
                            self.conf_has_update = False

                        if ENABLE_MOTION_DETECTION:
                            self.detect_motion(frame, t0)

                        if self.motion_ratio > self.min_motion:
                            has_motion = True
                        else:
                            has_motion = False

                        if not has_motion and self.det_batch_size == 1:
                            self.det_q.put((frame.copy(), t0, None, False))
                            continue

                        batch.append((frame, t0))
                        if len(batch) >= self.det_batch_size:
                            bgrs = [b[0] for b in batch]
                            t0s = [b[1] for b in batch]
                            batch.clear()
                            batch_regs = self.detect_batch(bgrs)
                            for i in range(len(batch_regs)):
                                self.det_q.put((bgrs[i], t0s[i], batch_regs[i], has_motion))

                            # logger.info('det_q size: {}'.format(self.det_q.qsize()))

                            delay = time.time() - t0
                            self.delay['det'] = (self.delay['det'] * 7 + delay) / 8
                    # else:
                    #     logger.warn('frame None')
                else:
                    if self.det_q.maxsize > 1:
                        logger.warn('det_q full')
                    # time.sleep(0.1)
            except Exception as ex:
                logger.exception(ex)
                if self.flg_debug:
                    raise

            ut.limit_fps_by_sleep(MAX_FPS, ts)  # prevent CPU outage

    def run(self):
        self._init()
        det_thread = Thread(target=self.run_detection_thread)
        det_thread.setDaemon(True)
        det_thread.start()
        while not self.should_stop:
            ts = time.time()
            self.has_detection = False
            try:
                if not self.det_q.empty():
                    frame, t0, regs, has_motion = self.det_q.get()

                    if has_motion:
                        self.has_detection = True

                    if frame is not None:
                        show = self.process_frame(frame, t0, regs, freeze_state=not has_motion)
                        self.draw_static_info(show)

                        if self.flg_debug or self.video_logger.rendered:
                            self.draw_debug_info(show)

                        if show.shape[1] > self.opts.stream_w:
                            self.frame_stream = ut.resize(show, dst_w=self.opts.stream_w, fast=True)
                        else:
                            self.frame_stream = show

                        if self.flg_debug and self._has_gui:
                            cv2.imshow(self.app_name, self.frame_stream)
                            if self.opts.img_dir:
                                key = cv2.waitKey(0) & 0xFF
                            else:
                                key = cv2.waitKey(1) & 0xFF

                        if self.flg_write:
                            if self.video_logger.rendered:  # write rendered key clips
                                self.video_logger.update(show, self.has_detection)
                            else:  # write input key clips
                                self.video_logger.update(frame, self.has_detection)

                        t_now = time.time()
                        self.delay['all'] = (self.delay['all'] * 7 + (t_now - t0)) / 8
                        self.delay['track'] = (self.delay['track'] * 7 + (t_now - ts)) / 8
                        self.processed_frame_cnt += 1
                        self.fps = self.processed_frame_cnt / (t_now - self.start_time)

                        if self.processed_frame_cnt % 10 == 0:
                            if self.flg_debug:
                                # wall clock = time: frame grabbed in detection thread -> fully processed
                                logger.info("fps: {:2.1f}, delay: {}".format(self.fps, self.delay))
            except Exception as ex:
                logger.exception(ex)
                if self.flg_debug:
                    raise

            ut.limit_fps_by_sleep(MAX_FPS, ts)  # prevent CPU outage

        # release resource & clean up
        if self.flg_debug:
            try:
                cv2.destroyWindow(self.app_name)
            except:
                pass

        if det_thread.is_alive():
            det_thread.join()
        self.frame_grabber.stop()
        self.video_logger.stop()

    def run_asyn(self):
        self.thread = Thread(target=self.run)
        self.thread.start()

    def detect_motion(self, cv_img, t0):
        """ return ratio of motion area to frame area """
        self.motion_scale = 128. / cv_img.shape[1]
        small = ut.resize(cv_img, scale=self.motion_scale, fast=True)

        # motion mask
        if t0 - self.stop_update_bg_at > self.stop_update_bg_duration:
            bg_learning_rate = -1
        else:
            bg_learning_rate = 0  # should_stop updating background
        self.motion_mask = self.motion_detector.apply(small, learningRate=bg_learning_rate)
        fgmask = self.motion_mask.copy()

        if self.roi and isinstance(self.roi, Roi) and self.roi.a > 0:
            scaled_roi = Roi(bbox=ut.scale_bb(self.roi.bbox, self.motion_scale))
            keep_mask = np.zeros(fgmask.shape, dtype=np.uint8)
            dr.draw_box(keep_mask, scaled_roi.bbox, color=dr.WHITE, simple=True, thickness=-1)
            cv2.bitwise_and(fgmask, keep_mask, fgmask)
            self.motion_ratio = float(cv2.countNonZero(fgmask)) / max(1, scaled_roi.a)
        else:
            self.motion_ratio = ut.nonezero_ratio(fgmask)

        # motion roi
        contours = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        min_t = MAX_INT
        min_l = MAX_INT
        max_b = 0
        max_r = 0
        min_s = max(fgmask.shape) * 0.05
        for c in contours:
            r = cv2.boundingRect(c)
            s = max(r[2], r[3])
            if s < min_s:
                continue
            if r[0] < min_l:
                min_l = r[0]
            if r[1] < min_t:
                min_t = r[1]
            if r[0] + r[2] > max_r:
                max_r = r[0] + r[2]
            if r[1] + r[3] > max_b:
                max_b = r[1] + r[3]
        if min_t != MAX_INT:
            self.motion_roi = Roi(tl=(min_l, min_t), br=(max_r, max_b), scale=1/self.motion_scale)
        else:
            self.motion_roi = Roi(tl=(0, 0), br=(0, 0))

        return self.motion_ratio

    def check_bb_motion(self, bb, percent_th):
        small_bb = [v * self.motion_scale for v in bb]
        if self.motion_mask is not None:
            cnt_nonezero = np.count_nonzero(ut.crop(self.motion_mask, small_bb))
            if cnt_nonezero * 100 / ut.bb_area(small_bb) > percent_th:
                return True
        return False

    @property
    def warp_pts(self):
        if "warp_pts" not in self.cam_cfg[self.cfg_key]:
            return []
        else:
            return [tuple(l) for l in self.cam_cfg[self.cfg_key]["warp_pts"]]

    @property
    def warp_dst_size(self):
        if "warp_dst_size" not in self.cam_cfg[self.cfg_key]:
            return None
        else:
            return self.cam_cfg[self.cfg_key]["warp_dst_size"]

    def warp(self, frame):
        if len(self.warp_pts) == 4:
            warped = ut.four_point_transform(
                frame,
                np.asarray(self.warp_pts, dtype=np.float32),
                dst_size=self.warp_dst_size)
        else:
            warped = frame.copy()
        return warped

    def process_frame(self, cv_img, t0, regs, freeze_state):
        return cv_img.copy()

    def add_cli_opts(self):
        # input options
        self.parser.add_option('--ip', '--cam_ip', default=None,
                               help='camera IP, 0 or \'webcam\' for webcam')
        self.parser.add_option('--ip_suffix', '--cam', type='int', default=None,
                               help='last octet of camera IP (2 - 255), 0 for for webcam')
        self.parser.add_option('--video_dir', default=None,
                               help='simulate input from avi files in specified dir')
        self.parser.add_option('--img_dir', default=None,
                               help='simulate input from image files in specified dir')
        self.parser.add_option('--work_w', type='int', default=0,
                               help='resize input frames to specified width')
        self.parser.add_option('--grab_fps', type='int', default=30,
                               help='depends on video source, for camera, > 30')

        # processing options
        self.parser.add_option('--max_fps', type='int', default=0,
                               help='trade-off resource need and acc, no less than 10')
        self.parser.add_option('--batch_size', type='int', default=1)

        # output options
        self.parser.add_option('--write', type='int', default=0,
                               help="save input key clips to --log_dir")
        self.parser.add_option('--write_raw', type='int', default=0,
                               help="save whole video stream to --log_dir")
        self.parser.add_option('--write_rendered', type='int', default=0,
                               help="save rendered output key clips to --log_dir")
        self.parser.add_option('--log_dir', default=PATH_LOG_DIR)
        self.parser.add_option('--write_ts', type=int, default=0,
                               help='write timestamp to output clips')

        self.parser.add_option('--stream_w', type='int', default=800,
                               help="resize output video stream to specified width")

        # display options
        self.parser.add_option('--debug', type='int', default=0)
        self.parser.add_option('--direct_display', type='int', default=0)

        # web api
        self.parser.add_option('--port', type='int', default=8080)
        self.parser.add_option('--web_port', type='int', default=None)  # backward compatible
        self.parser.add_option('--webui_port', type='int', default=None)  # backward compatible

        # dnn model
        self.parser.add_option('--model', default='')

    def get_cli_opts(self):
        opts, args = self.parser.parse_args()

        self.cam_ip_suffix = opts.ip_suffix
        self.cam_ip = opts.ip
        self.grab_fps = opts.grab_fps

        self.flg_write = bool(opts.write)
        self.flg_write_rendered = bool(opts.write_rendered)
        self.flg_write_raw = bool(opts.write_raw)

        self.flg_debug = bool(opts.debug)
        self.flg_direct_display = bool(opts.direct_display)

        if opts.webui_port:
            self.port = opts.webui_port
        elif opts.web_port:
            self.port = opts.web_port
        else:
            self.port = opts.port

        self.model = opts.model

        if opts.video_dir:
            self.input_url = opts.video_dir

        if opts.img_dir:
            self.input_url = opts.img_dir

        if self.flg_write_rendered or self.flg_write_raw:
            self.flg_write = True

        if opts.max_fps > 0:
            MAX_FPS = opts.max_fps

        self.det_batch_size = opts.batch_size

        self.opts = opts

        return opts

    def draw_debug_info(self, disp):
        if ENABLE_MOTION_DETECTION:
            scale = float(disp.shape[1]) / disp.shape[1]
            l, t, r, b = ut.scale_bb(self.motion_roi.bbox, scale)
            cv2.rectangle(disp, (l, t), (r, b), dr.WHITE, 2)

    def draw_static_info(self, disp):
        dr.draw_text(disp, 'FPS: {:2.1f}, DELAY: {:.2f}'.format(self.fps, self.delay['all']),
                     (30, 40), dr.RED, thickness=2)
        if self.hc_connected is False:
            dr.draw_text(disp, f'HC disconnected, retry in {int(self.hc_next_retry - time.time())} seconds',
                         (30, 80), dr.RED, thickness=2)
        if self.motion_ratio > self.min_motion:
            cv2.circle(disp, (disp.shape[1] // 2, 30), 9, dr.GREEN, thickness=2)
            if self.flg_write and self.has_detection:
                cv2.circle(disp, (disp.shape[1] // 2, 30), 9, dr.RED, -1)

    @staticmethod
    def get_controller_ip():
        try:
            controller_ip = request.host.split(':')[0]
        except Exception as ex:
            logger.exception(ex)
            controller_ip = global_cfg.CONTROLER_IP
        return controller_ip

    @staticmethod
    def sys_info():
        import datetime
        info = {}
        # software info
        try:
            info['expire_date'] = 'N/A'
            license_json = os.path.join(PATH_LOG_DIR, "sys/license.json")
            if os.path.isfile(license_json):
                with open(license_json, "r") as f:
                    data = json.load(f)
                    if 'expire_date' in data:
                        expire_date = datetime.datetime.strptime(data['expire_date'], DATE_FORMAT)
                        if expire_date.year - datetime.datetime.now().year > 2:
                            info['expire_date'] = 'N/A (unlimited)'
                        else:
                            info['expire_date'] = data['expire_date']
        except Exception as ex:
            logger.exception(ex)

        try:
            info['version'] = 0
            version_files = [f for f in os.listdir(PATH_MODEL_DIR) if f.startswith('.')]
            for f in version_files:
                try:
                    re_no = int(f[1:])
                    if re_no > info['version']:
                        info['version'] = re_no
                except:
                    pass
        except Exception as ex:
            logger.exception(ex)

        try:
            info['update_date'] = 0
            exec_dir = '../../../src'
            update = 0
            if os.path.exists(exec_dir):
                update = max(os.stat(root).st_mtime for root, _, _ in os.walk(exec_dir))
            info['update_date'] = datetime.datetime.fromtimestamp(update).strftime("%y%m%d")
        except Exception as ex:
            logger.exception(ex)

        return info

    def resume(self):
        super(VideoMonitorApp, self).resume()
        self.run_asyn()

    @property
    def stopline_y(self):
        return int(self.conf.get('stopline_y', -1))

    @property
    def count_margin(self):
        return int(self.conf.get('count_margin', -1))

    @property
    def left(self):
        return int(self.conf.get('left', -1))

    @property
    def right(self):
        return int(self.conf.get('right', -1))
