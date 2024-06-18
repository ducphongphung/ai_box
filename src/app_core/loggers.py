from queue import Queue

from src.global_def import *
import src.app_cfg as app_cfg
import src.global_def as global_def
from src.utils.services import forward_requests_task

from threading import Thread

import datetime
import requests
import random

try:
    import OutputClipCfg
except:
    class OutputClipCfg:
        FPS = MAX_FPS
        RESOLUTION = LOG_VIDEO_RESOLUTION
        MAX_DURATION = 5  # max duration of event clip
        BEFORE_EVENT = 1  # save how many seconds before event to event clip
        AFTER_EVENT = 3  # save how many seconds after event to event clip
        CODEC = "XVID"
        FILE_EXT = "avi"

logger = dbg.get_logger(__name__)


class CamLogger(object):
    def __init__(self, outdir, cam_ip, log_dir=PATH_LOG_DIR, cam_port=None):
        self.set_output_dir(os.path.join(log_dir, outdir))
        self.cam_ip = cam_ip
        self.cam_port = cam_port

        try:
            self.f_ws_fn = getattr(global_cfg, 'F_WS_FN')
        except AttributeError:
            self.f_ws_fn = None

    def set_output_dir(self, dir):
        ut.mkdir_if_not_existed(dir)
        self.output_dir = dir

    def get_output_dir(self, date=''):
        """ partition by day to speedup processing """
        date = datetime.date.today().strftime('%Y-%m-%d') if date == '' else date
        dir = os.path.join(self.output_dir, date)
        return dir

    @property
    def daily_output_dir(self):
        """ partition by day to speedup processing """
        dir = self.get_output_dir()
        ut.mkdir_if_not_existed(dir)
        return dir

    @property
    def cam_host(self):
        if self.cam_port:
            return '{}_{}'.format(self.cam_ip, self.cam_port)
        else:
            return self.cam_ip

    def get_log_file_path(self, identity, timestamp, file_ext):
        return '{}/{}_{}_{}.{}'.format(
            self.daily_output_dir, timestamp, self.cam_host, identity, file_ext)


class VideoLogger(CamLogger):
    def __init__(self, cam_ip, autosave=False, rendered=False, lossless=False,
                 log_dir=PATH_LOG_DIR, cam_port=None, timestamp=0):
        super(VideoLogger, self).__init__('videos', cam_ip, log_dir, cam_port=cam_port)
        # settings
        self.lossless = lossless
        self.autosave = autosave
        self.rendered = rendered
        self.write_timestamp = timestamp

        self.kcw = KeyClipWriter(bufSize=max(1, OutputClipCfg.FPS * OutputClipCfg.BEFORE_EVENT))
        self.latest_log_file = ''
        self.last_event = 0  # last time we saw event in stream, use to stop recording

    @property
    def fourcc(self):
        return cv2.VideoWriter_fourcc(*OutputClipCfg.CODEC)

    def update(self, frame, has_detection):
        t0 = time.time()

        if has_detection:
            self.last_event = t0
            if self.autosave:
                self.start()

        if t0 - self.last_event < 1. / OutputClipCfg.FPS:  # limit input fps
            return

        if not self.lossless:
            frame = ut.resize(frame, dst_size=OutputClipCfg.RESOLUTION, fast=True)

        if self.write_timestamp:
            dr.draw_text(frame, ut.strftimestamp(t0, TIME_FULL_FORMAT), (10, 30), thickness=2)

        self.kcw.update(frame)

        if t0 - self.last_event > OutputClipCfg.AFTER_EVENT:
            if self.kcw.recording is True:
                self.kcw.finish()

    def start(self):
        self.last_event = int(time.time())

        if self.kcw.recording is True:
            return

        # maintaining 10 Gb free disk-space would be safe
        if ut.get_free_disk_space('../../../src') < 10000000000:
            logger.error('Running out of disk-space for writing more videos.')
        else:
            try:
                log_file = self.get_log_file_path('', self.last_event, OutputClipCfg.FILE_EXT)
                if self.rendered is True:
                    log_file = log_file + STR_ID_RENDERED
                self.latest_log_file = log_file
                self.kcw.start(log_file, self.fourcc)
            except Exception as ex:
                logger.exception('Failed to start logging video: {}'.format(ex))

    def stop(self):
        if self.kcw is not None:
            self.kcw.finish()

    def touch(self):
        """ a cheap operation to make sure self.latest_log_file is not out-dated """
        self.start()
        return self.latest_log_file


class CentralizedCamLogger(CamLogger):
    def __init__(self, cam_ip, saver_ip='localhost', fps=1.0, hub=None):
        super(CentralizedCamLogger, self).__init__('events', cam_ip)
        self.saver_ip = saver_ip
        self.log_notify_time = {}  # control log rate
        self.fps = fps  # control log rate

        if HTTPS:
            protocol = 'https'
        else:
            protocol = 'http'

        if hub is None:
            hub = {}
        self.hub_ip = hub.get('ip', app_cfg.CONTROLER_IP)
        self.hub_port = hub.get('port', SERVICES[SV_CONTROLLER]['port'])
        self.controller_url = "{}://{}:{}".format(
            protocol, self.hub_ip, self.hub_port)

    def get_file_link(self, file_path):
        if file_path:
            return 'http://{}:{}/{}'.format(
                self.saver_ip, SERVICES[SV_FILE]['port'], os.path.abspath(file_path))
        else:
            return ''


class CentralizedEventCamLogger(CentralizedCamLogger):
    def _post_log(self, event_type, crop, video_path, timestamp, tag):
        try:
            if crop.shape[1] > 320:
                save_crop = ut.resize(crop, dst_w=320)
            else:
                save_crop = crop

            if timestamp is None:
                timestamp = ut.get_timestamp()

            image_path = self.get_log_file_path('', timestamp * 1000 + random.randint(1, 1000), 'jpg')
            image_link = self.get_file_link(image_path)
            video_link = self.get_file_link(video_path)
            cv2.imwrite(image_path, save_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            url = '{}/event'.format(self.controller_url)

            data = {
                'event_type': event_type,  # for server to de-duplicate and apply its own log rate control
                'timestamp': timestamp,
                'cam_ip': self.cam_ip,
                'image_link': image_link,
                'video_link': video_link,
                'json_tag': json.dumps(tag),
            }

            Thread(target=forward_requests_task, kwargs=({
                'requests_data': {
                    'url': url, 'data': data, 'timeout': STD_HTTPS_SERVICE_TIMEOUT_SEC
                }
            })).start()
        except requests.Timeout as ex:
            logger.error(ex)
        except Exception as ex:
            logger.exception('Failed to log event: {}'.format(ex))

    def write(self, event_type, video_path='', timestamp=None, crop=None, tag={}):
        try:
            # write log for an identity every 1/self.fps seconds - a cheap way to reduce redundancy
            t0 = time.time()
            if event_type in self.log_notify_time and t0 - self.log_notify_time[event_type] < (1 / self.fps):
                logger.debug('writing log: skipped by log rate controller: identity:  {}'.format(event_type))
                if len(self.log_notify_time) > 512:  # prevent dict growing out of control
                    self.log_notify_time = {}
                    logger.info('reset log_notify_time')
                return

            self.log_notify_time[event_type] = t0

            self._post_log(event_type, crop, video_path, timestamp, tag)
        except Exception as ex:
            logger.exception('Failed to log event: {}'.format(ex))


class CentralizedFaceCamLogger2(CentralizedEventCamLogger):
    def _post_log(self, event_type, crop, video_path, timestamp, tag):
        try:
            if crop.shape[1] > 320:
                save_crop = ut.resize(crop, dst_w=320)
            else:
                save_crop = crop

            if timestamp is None:
                timestamp = ut.get_timestamp()

            image_path = self.get_log_file_path('', timestamp * 1000 + random.randint(1, 1000), 'jpg')
            image_link = self.get_file_link(image_path)
            video_link = self.get_file_link(video_path)
            cv2.imwrite(image_path, save_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            url = '{}/history'.format(self.controller_url)

            data = {
                'user': event_type,
                'timeInOut': timestamp,
                'cam_ip': self.cam_ip,
                'imagePath': image_link,
                'videoPath': video_link,
            }

            Thread(target=forward_requests_task, kwargs=({
                'requests_data': {
                    'url': url, 'data': data, 'timeout': STD_HTTPS_SERVICE_TIMEOUT_SEC
                }
            })).start()

        except requests.Timeout as ex:
            logger.error(ex)
        except Exception as ex:
            logger.exception('Failed to log event: {}'.format(ex))


class CentralizedFaceCamLogger(CentralizedCamLogger):

    class Task(object):
        LIFE_EXPECTANCY_SEC = 1
        LIFE_EXPECTANCY_UNKNOWN_SEC = 3

        def __init__(self, obj_track, overview, door_id, video_path, timestamp, crop):
            self.obj_track = obj_track
            self.overview = overview
            self.door_id = door_id
            self.video_path = video_path
            self.timestamp = timestamp
            self.crop = crop
            # delay write task so we have chance to capture better face picture
            if obj_track.identity == STR_UNKNOWN:
                self.ttd = time.time() + self.LIFE_EXPECTANCY_UNKNOWN_SEC
            else:
                self.ttd = time.time() + self.LIFE_EXPECTANCY_SEC
            self.id = random.getrandbits(32)
            self.current_bb = obj_track.bb

        def __hash__(self):
            return self.id

        def __eq__(self, other):
            return self.id == other.id

        def merge(self, another_task):
            """ keep self.obj_track """
            # extend the start time of track
            start = min(self.obj_track.start, another_task.obj_track.start)
            another_track = another_task.obj_track

            # keep identity that is not Unknown
            identity = self.obj_track.identity
            if identity == STR_UNKNOWN:
                identity = another_track.identity

            # keep other content of cleaner track after merging
            if self.obj_track.clarity < another_track.clarity:
                self.overview = another_task.overview
                self.crop = another_task.crop
                # update id according to overview to avoid duplicated logged faces
                self.id = another_task.id

                # identity of clearer track will prevail, unless it is Unknown
                if another_track.identity != STR_UNKNOWN:
                    identity = another_track.identity

                self.obj_track = another_track

            self.obj_track.identity_track.identity = identity
            self.obj_track.start = start

            # update to latest states
            if self.ttd < another_task.ttd:
                self.ttd = another_task.ttd
                self.current_bb = another_task.current_bb

                self.door_id = another_task.door_id
                self.video_path = another_task.video_path
                self.timestamp = another_task.timestamp

    def _consolidate_identity(self, obj_track):
        return obj_track.identity

    def __init__(self, cam_ip, cam_direction=None, saver_ip='localhost', fps=1.0):
        super(CentralizedFaceCamLogger, self).__init__(cam_ip, saver_ip=saver_ip, fps=fps)
        self.direction = cam_direction
        self.task_q = Queue(100)  # roughly 10 faces in 1 seconds

        Thread(target=self.run_write_optimal_thread).start()

    def _write(self, crop, identity, door_id, video_path, timestamp, wait, score):
        try:
            if crop.shape[1] > 320:
                save_crop = ut.resize(crop, dst_w=320)
            else:
                save_crop = crop

            if timestamp is None:
                timestamp = ut.get_timestamp()

            logfile = self.get_log_file_path(identity, timestamp * 1000 + random.randint(1,1000), 'jpg')
            imagelink = self.get_file_link(logfile)
            videolink = self.get_file_link(video_path)
            cv2.imwrite(logfile, save_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            url = '{}/history'.format(self.controller_url)

            if must_ignore_identity(identity):
                identity = STR_UNKNOWN

            data = {
                'user': identity,
                'timeInOut': timestamp,
                'direction': self.direction,
                'cam_ip': self.cam_ip,
                'door_id': door_id,
                'imagePath': imagelink,
                'videoPath': videolink,
                'wait': wait,
                'score': score
            }

            Thread(target=forward_requests_task, kwargs=({
                'requests_data': {
                    'url': url, 'data': data, 'timeout': STD_HTTPS_SERVICE_TIMEOUT_SEC
                }
            })).start()
        except requests.Timeout as ex:
            logger.error(ex)
        except Exception as ex:
            logger.exception('Failed to log event: {}'.format(ex))

    def write_manual_log(self, face_im, identity, door_id, video_path):
        self._write(face_im, identity, door_id, video_path, None, 8, 100)

    def write(self, obj_track, overview, door_id='', video_path='', timestamp=None, crop=None):
        try:
            # write log for an identity every 1/self.fps seconds - a cheap way to reduce redundancy
            identity = self._consolidate_identity(obj_track)
            t0 = time.time()
            if identity in self.log_notify_time and t0 - self.log_notify_time[identity] < (1/self.fps):
                logger.debug('writing log: skipped by log rate controller: identity:  {}'.format(identity))
                if len(self.log_notify_time) > 512:  # prevent dict growing out of control
                    self.log_notify_time = {}
                    logger.info('reset log_notify_time')
                return

            if not obj_track.is_valid:
                logger.error("invalid face track (no observation): id={} bb={}, ".format(identity, obj_track.bb))
                return

            if identity != STR_UNKNOWN:
                self.log_notify_time[identity] = t0

            bb = Roi(bbox=obj_track.bb).resize(1.25).bbox
            wait = obj_track.identity_wait_sec
            score = int(100 * obj_track.min_match_dist)

            if overview is not None:
                save_crop = ut.crop(overview, bbox=bb)
            else:
                save_crop = crop

            self._write(save_crop, identity, door_id, video_path, timestamp, wait, score)
        except Exception as ex:
            logger.exception('Failed to log event: {}'.format(ex))

    def queue_write_task(self, face_track, overview, door_id='', video_path='', timestamp=None, crop=None):
        try:
            task = self.Task(face_track, overview, door_id, video_path, timestamp, crop)
            if self.task_q.not_full:
                self.task_q.put(task)
            else:
                logger.error('task_q is full')
        except Exception as ex:
            logger.exception(ex)

    def is_valid(self, task):
        t = task.obj_track
        if t.identity == STR_UNKNOWN:
            # BR. write only qualified unknown tracks to avoid unknown overpopulating the history inout
            if t.is_qualified_unknown_track:
                return True
        if (t.identity != STR_UNKNOWN) and (not must_ignore_identity(t.identity)):
            return True
        return False

    def run_write_optimal_thread(self):
        """ selects best pictures to save """
        old_tasks = []
        while global_def.g_shutdown is False:
            t0 = time.time()
            try:
                while not self.task_q.empty():
                    try:
                        # if new task is actually a pending task, update pending task, discard new task
                        # otherwise, add new task to pending task list
                        new_task = self.task_q.get()
                        new_track = new_task.obj_track

                        logger.debug('handle new task: identity={}, bb={}'.format(new_track.identity, new_track.bb))

                        """ track and select the optimal picture to log """
                        keep = [True] * len(old_tasks)
                        # try matching new track with old tracks
                        for i in range(len(old_tasks)):
                            old_task = old_tasks[i]
                            old_track = old_task.obj_track
                            matched = False

                            # 1. match by identity
                            if (old_track.identity == new_track.identity) and (old_track.identity != STR_UNKNOWN):
                                logger.debug('track matched by identity: {}\n'.format(old_track.identity))
                                matched = True
                            else:
                                emb_dist = old_track.dist(new_track)
                                if emb_dist < app_cfg.FACE_REG_LOOSE_TH:
                                    # 2. match by trajectory (match both position & emb)
                                    if new_track.contains(old_track.bb):
                                        logger.debug('track matched by trajectory')
                                        matched = True
                                    # 3. match by emb
                                    elif emb_dist < app_cfg.FACE_REG_TH:
                                        logger.debug('track matched by emb')
                                        matched = True

                                    # # disconnect unknown tracks that connected by emb,
                                    # # so that there are more unknown tracks
                                    # if global_def.SAVE_ALL_UNKNOWN:
                                    #     if old_track.identity == STR_UNKNOWN:
                                    #         if new_track.identity == STR_UNKNOWN:
                                    #             matched = False

                            if matched:
                                logger.debug('merge {}: {} and {}: {}'.format(
                                    new_track.identity, new_track.bb, old_track.identity, old_track.bb))
                                new_task.merge(old_tasks[i])
                                keep[i] = False

                        logger.debug('merge kills {} tasks\n'.format(len(old_tasks) - sum(keep)))
                        old_tasks = [t for t, k in zip(old_tasks, keep) if k is True]
                        old_tasks.append(new_task)
                    except Exception as ex:
                        logger.exception(ex)

                logger.debug('handle pending tasks: {} remaining'.format(len(old_tasks)))
                old_tasks = list(set(old_tasks))  # remove duplication
                logger.debug('handle pending tasks: removed duplications: {} remaining'.format(len(old_tasks)))
                for task in old_tasks:
                    logger.debug(task.current_bb)
                    try:
                        if task.ttd < t0:  # task expired
                            # chi save Unknown > 3s
                            if not (task.obj_track.identity == STR_UNKNOWN and (t0 - task.LIFE_EXPECTANCY_UNKNOWN_SEC - task.obj_track.start < 3)):
                                if self.is_valid(task):
                                    self.write(
                                        task.obj_track, task.overview, task.door_id,
                                        task.video_path, task.timestamp, task.crop)
                    except Exception as ex:
                        logger.exception(ex)

                old_tasks = [t for t in old_tasks if t.ttd > t0]

                ut.limit_fps_by_sleep(MAX_FPS, t0)

            except Exception as ex:
                logger.exception(ex)

        logger.info("EventCamLogger: gracefully stopped")


class TimeSheetLogger(CentralizedFaceCamLogger):
    def is_valid(self, task):
        """ applies specific time-sheet logic """
        t = task.obj_track

        if t.identity != STR_UNKNOWN and not must_ignore_identity(t.identity):
            return True

        return super(TimeSheetLogger, self).is_valid(task)


class VehicleLogger(TimeSheetLogger):
    def __init__(self, lpdb, cam_ip, cam_direction=None, saver_ip='localhost', fps=1.0):
        self.lpdb = lpdb
        super(VehicleLogger, self).__init__(cam_ip, cam_direction, saver_ip, fps)

    def _consolidate_identity(self, obj_track):
        return obj_track.identity


class BaseWriter(object):
    def __init__(self):
        self.writer = None
        self.outputPath = None
        self.thread = None
        self.recording = False

    def flush(self):
        pass

    def write(self):
        pass

    def start_write_thread(self):
        self.recording = True
        self.thread = Thread(target=self.write, args=())
        self.thread.setDaemon(True)
        self.thread.start()

    def finish(self):
        # indicate that we are done recording, join the thread,
        # flush all remaining frames in the queue to file, and
        # release the writer pointer
        self.recording = False
        if self.thread is not None:
            self.thread.join()  # self.recording = False -> thread terminates immediately
            self.thread = None
        self.flush()  # self.writer writes immediately all remaining frames in buffer
        if self.writer:
            self.writer.release()
            self.writer = None
            # write marker file to let 3rd party know writing is done
            with open(self.outputPath + ".done", 'w') as fp:
                pass

        logger.info("{}: gracefully stopped".format(self.__class__.__name__))


class KeyClipWriter(BaseWriter):
    def __init__(self, bufSize=64):
        # store the maximum buffer size of frames to be kept
        # in memory along with the sleep timeout during threading
        self.bufSize = int(bufSize)

        # initialize the buffer of frames, queue of frames that
        # need to be written to file, video writer, writer thread,
        # and boolean indicating whether recording has started or not
        self.frames = deque(maxlen=self.bufSize)
        self.Q = None

        super(KeyClipWriter, self).__init__()

    def update(self, frame):
        # update the frames buffer
        self.frames.appendleft(frame)

        # if we are recording, update the queue as well
        if self.recording:
            self.Q.put(frame)

    def start(self, outputPath, fourcc):
        self.outputPath = outputPath
        self.fourcc = fourcc
        self.Q = Queue()

        # loop over the frames in the deque structure and add them
        # to the queue
        for i in range(len(self.frames), 0, -1):
            self.Q.put(self.frames[i - 1])

        self.start_write_thread()

    def write(self):
        # indicate that we are recording, start the video writer,
        # and initialize the queue of frames that need to be written
        # to the video file
        self.writer = cv2.VideoWriter(
            self.outputPath, self.fourcc, OutputClipCfg.FPS,
            (self.frames[0].shape[1], self.frames[0].shape[0]), True)
        # keep looping
        while True:
            # if we are done recording, exit the thread
            if not self.recording:
                return

            # check to see if there are entries in the queue
            if not self.Q.empty():
                # grab the next frame in the queue and write it
                # to the video file
                frame = self.Q.get()
                self.writer.write(frame)

            # otherwise, the queue is empty, so sleep for a bit
            # so we don't waste CPU cycles
            else:
                time.sleep(0.01)

    def flush(self):
        # empty the queue by flushing all remaining frames to file
        if self.Q is None:
            return
        while not self.Q.empty():
            frame = self.Q.get()
            self.writer.write(frame)
