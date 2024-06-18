from src.global_def import *

import threading
import six
import cv2
from queue import Queue

logger = dbg.get_logger(__name__)


def is_str(s):
    return isinstance(s, six.string_types)


class FrameGrabber(threading.Thread):
    VIDEO_FILE_EXTS = (
        ".3g2", ".3gp", ".asf", ".asx", ".avi", ".flv", ".m2ts", ".mkv", ".mov", ".mp4", ".mpg", ".mpeg", ".rm", ".swf",
        ".vob", ".wmv")

    def __init__(self, cam_ip=None, src_url=None, cam_type=None, im_transform_fn=None,
                 cap=None, fps=30, **kwargs):
        """
        no_delay: True for responsive app e.g. door-open ASAP, False for high fps app e.g. lpr,
        no_delay can reduce decoding error due to camera buffer overflow when CPU peaks
        """
        threading.Thread.__init__(self)
        # thread safe runtime control
        self.setDaemon(True)
        self.should_stop = False
        self.should_jump = False
        self.should_reset_cap = False  # force reset capture

        self.frame = None
        self.frame_t0 = 0
        self.frame_id = 0
        self.q = Queue(maxsize=1)  # single subscriber interface (auto sync speed)
        self.cam_ip = cam_ip
        self.cam_type = cam_type
        self.cap = cap
        self.video_idx = -1
        self.video_restarted = False
        self.img_idx = -1
        self.id = ut.get_uuid()
        self.actual_fps = 0
        self.consecutive_corrupted_frames = 0
        self.last_corrupted = 0

        self.src_url = src_url
        if self.src_url is None:
            if cam_ip is not None:
                self.src_url = get_cam_service_url(cam_service_id=cam_type, ip=cam_ip)
            else:
                raise ValueError('src_url or cam_ip not specified')
        else:
            if self.cam_ip is None:
                self.cam_ip = ut.to_ip(self.src_url)

        self.im_transform_fn = im_transform_fn

        self.imgs = []
        self.videos = []

        if not ut.is_webcam_index(self.src_url) and os.path.isdir(self.src_url):
            files = ut.get_all_files_recursively(self.src_url)
            self.imgs = [f for f in files if ut.is_valid_image(f)]
            self.videos = [f for f in files if f.endswith(self.VIDEO_FILE_EXTS)]
            # avoid output videos, images
            self.imgs = [f for f in self.imgs if not f.endswith(STR_OUT_JPG)]
            self.videos = [f for f in self.videos if not (STR_ID_RENDERED in f)]
            self.videos.sort(key=os.path.getmtime)

        self.fps = fps
        self.no_delay = kwargs.get('no_delay', True)
        # always reconnect after each 300s if video src is rtsp
        # to prevent connection hang because ffmpeg cannot recover from decoding errors
        self.regular_reconnect = kwargs.get('regular_reconnect', False)
        # only use for video file source
        # grab frame from video file as fast as processing can handle
        self.sync_grab = (fps == 0)
        self.detect_corrupted_frame = kwargs.get('detect_corrupted_frame', False)
        self.reconnect_wait = kwargs.get('reconnect_wait', 3)
        self.video_changed_fn = None

    def grab_all_frames_from_cam_buffer(self):
        buffer_cnt = 0
        while True:
            t = time.time()
            has_frame = self.cap.grab()
            if has_frame:
                if time.time() - t < 0.01:  # when there is still frame in buffer, grab is very fast
                    buffer_cnt += 1
                else:
                    break
            else:
                buffer_cnt = -1  # decode error, cam disconnected
                break
        if buffer_cnt > 4:  # buffer_cnt == 1 is expected for optimal operation
            logger.error('buffer drop: {}, actual fps: {}, limit: {}'.format(
                buffer_cnt, int(self.actual_fps), self.fps))
        return buffer_cnt

    def _decode_grabbed_frame(self, t0):
        if self.detect_corrupted_frame:
            from wurlitzer import pipes
            with pipes() as (out, err):
                has_frame, frame = self.cap.retrieve()
            out_txt = err.read()
            if out_txt.startswith('['):
                self.consecutive_corrupted_frames += 1
                logger.info('consecutive_corrupted_frames: {}'.format(
                    self.consecutive_corrupted_frames))
                self.last_corrupted = t0
            else:
                if t0 - self.last_corrupted > 1:
                    # reset if no corruption in 1 seconds
                    self.consecutive_corrupted_frames = 0
        else:
            has_frame, frame = self.cap.retrieve()
        return has_frame, frame

    def run(self):
        self.frame_id = 0
        self.video_idx = -1
        self.img_idx = -1
        self.should_stop = False
        consecutive_missing_frames = 0
        step_frame_cnt = 0
        step_t0 = 0
        is_rtsp = is_str(self.src_url) and self.src_url.startswith('rtsp')
        is_video = not is_rtsp \
                   and is_str(self.src_url) \
                   and self.src_url.endswith(self.VIDEO_FILE_EXTS)

        if os.path.exists('/opt/intel/install.log'):  # demo version: not allow rtsp
            if is_rtsp:
                return

        if self.cap is None or not self.cap.isOpened():
            self.cap = ut.get_capture(self.src_url)

        if is_video and self.cap and self.cap.isOpened():
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info("Video source: {}. Grab fps: {}".format(self.src_url, self.fps))
        start_time = time.time()
        last_reconnect = start_time
        while self.should_stop is False:
            t0 = time.time()
            frame = None
            has_frame = False

            try:
                # measure actual fps once per 5s
                step_frame_cnt += 1
                if t0 - step_t0 > 5:
                    self.actual_fps = float(step_frame_cnt) / (t0 - step_t0)
                    step_t0 = t0
                    step_frame_cnt = 0

                # allow reset cap from outside thread
                if self.should_reset_cap:
                    logger.error("Handle request to reset capture")
                    self.should_reset_cap = False
                    self.cap = ut.handle_cam_disconnected(self.src_url, self.cap)

                # auto reset cap when ffmpeg cannot recover from decoding error
                # be careful: handle_cam_disconnected hangs if called too frequently
                # so there is a self.reconnect_wait seconds threshold
                if is_rtsp:
                    if self.consecutive_corrupted_frames > self.reconnect_wait * self.fps:
                        logger.error("Cannot recover from decoding error, reset capture")
                        self.cap = ut.handle_cam_disconnected(self.src_url, self.cap)
                        self.consecutive_corrupted_frames = 0
                    if self.regular_reconnect and t0 - last_reconnect > 300:
                        logger.info("Trigger regular reconnection")
                        self.cap = ut.handle_cam_disconnected(self.src_url, self.cap)
                        last_reconnect = t0

                # ========================= Grab frame from sources
                # image folder source
                if self.img_idx < len(self.imgs) - 1:
                    self.img_idx += 1  # so that when crash, idx still moves on
                    frame = cv2.imread(self.imgs[self.img_idx])
                    if frame is not None:
                        has_frame = True
                        # in image mode, manually enter next
                        while not self.q.empty():
                            time.sleep(0.5)

                # video source
                if not has_frame and (self.cap is not None) and self.cap.isOpened():
                    if self.should_jump:
                        for i in range(self.should_jump):
                            self.cap.grab()
                        self.should_jump = 0

                    if is_rtsp and self.no_delay:
                        buffer_cnt = self.grab_all_frames_from_cam_buffer()
                        if buffer_cnt > -1:
                            has_frame = True
                            step_frame_cnt += buffer_cnt
                    elif is_video:
                        wall_time = t0 - start_time
                        video_time = float(self.frame_id) / self.fps

                        if not self.sync_grab:
                            # if wall_time - video_time > 1:
                            #     logger.info('lag: {}'.format(wall_time - video_time))

                            # reduce lag by skipping frames
                            while wall_time > video_time:
                                has_frame = self.cap.grab()
                                if has_frame:
                                    self.frame_id += 1
                                    video_time = float(self.frame_id) / self.fps
                                else:
                                    break
                        else:
                            has_frame = True
                            if not self.q.full():
                                self.cap.grab()
                    else:
                        has_frame = self.cap.grab()

                    if has_frame and not self.q.full():
                        has_frame, frame = self._decode_grabbed_frame(t0)
                # ========================= DONE Grab frame from sources

                if has_frame:
                    consecutive_missing_frames = 0
                    if not self.q.full():
                        if frame is not None:
                            if self.im_transform_fn is not None:
                                cv_img = self.im_transform_fn(frame, self.cam_type, self.cam_ip)
                            else:
                                cv_img = frame.copy()
                            self.q.put((cv_img, t0))
                            # logger.info('frame_q size: {}'.format(self.q.qsize()))
                    else:
                        if self.q.maxsize > 1:
                            logger.warn('frame_q full')
                else:
                    if self.videos:
                        if self.video_idx >= len(self.videos) - 1:
                            logger.info("====================================================")
                            logger.info("Restart from video dir: {}".format(self.src_url))
                            self.video_idx = 0
                            self.video_restarted = True
                        else:
                            self.video_idx = self.video_idx + 1
                            if self.video_changed_fn:
                                self.video_changed_fn()
                        self.cap = ut.get_capture(self.videos[self.video_idx])
                        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
                        logger.info("Running from: {}. Grab fps: {}".format(
                            self.videos[self.video_idx], self.fps))
                    else:
                        if consecutive_missing_frames > self.reconnect_wait * self.fps:  # not reconnect too frequently
                            logger.error("consecutive_missing_frames: {}".format(consecutive_missing_frames))
                            self.cap = ut.handle_cam_disconnected(self.src_url, self.cap)
                            consecutive_missing_frames = 0
                        else:
                            consecutive_missing_frames += 1
                            if consecutive_missing_frames == 1:
                                logger.error("missing frame detected")

                if not is_video:
                    self.frame_id += 1

            except Exception as ex:
                logger.exception(ex)

            ut.limit_fps_by_sleep(self.fps, t0)

        if self.cap:
            self.cap.release()

    # region interface

    def get_frame(self):
        """ returns (frame, t0), if frame is None, t0 is invalid """
        if not self.is_alive():
            self.start()
        if not self.q.empty():
            try:
                self.frame, self.frame_t0 = self.q.get()
            except:
                pass
        return self.frame, self.frame_t0

    def stop(self):
        self.should_stop = True
        if self.is_alive():
            self.join()
        logger.info("FrameGrabber: gracefully stopped: {}".format(self.cam_ip))

    @property
    def current_video(self):
        if self.videos:
            return self.videos[self.video_idx]
        else:
            return None

    # endregion
