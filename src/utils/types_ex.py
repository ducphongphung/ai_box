import threading

import src.utils.common as ut
import numpy as np


class Roi(object):
    def __init__(self, bbox=None, cv_rect=None, tl=None, br=None, rois=None, img=None, scale=1., dtype='int'):
        """
        :param bbox: 
        :param cv_rect: 
        :param tl: 
        :param br: 
        :param rois: create new roi by merging bboxes of child rois
        :param scale: scale the roi by scaling tl and br
        """
        # specifying (tl, br) has highest priority
        if tl is None or br is None:
            if bbox is None:  # specifying bbox has 2nd priority
                if rois is not None and len(rois) > 0:
                    points = []
                    for r in rois:
                        points.extend([r.tl, r.br])
                    bbox = ut.get_bb(points)
                if cv_rect is not None:
                    bbox = ut.to_bbox(cv_rect)
        if bbox is not None:
            tl = (bbox[0], bbox[1])
            br = (bbox[2], bbox[3])
        if img is not None:
            tl = (0, 0)
            br = (img.shape[1], img.shape[0])

        if tl is None or br is None:
            raise ValueError('Roi error: topleft or bottomright is not specified')
        else:
            self._tl = tl
            self._br = br

        self.dtype = dtype

        self._tl = (self._tl[0] * scale, self._tl[1] * scale)
        self._br = (self._br[0] * scale, self._br[1] * scale)

    @property
    def tl(self):
        if self.dtype == 'int':
            return tuple([int(e) for e in self._tl])
        return self._tl

    @property
    def br(self):
        if self.dtype == 'int':
            return tuple([int(e) for e in self._br])
        return self._br

    @property
    def bbox(self):
        rs = [self.tl[0], self.tl[1], self.br[0], self.br[1]]
        if self.dtype == 'int':
            return [int(e) for e in rs]
        return rs

    @property
    def cv_rect(self):
        rs = [self.tl[0], self.tl[1], self.w, self.h]
        if self.dtype == 'int':
            return [int(e) for e in rs]
        return rs

    @property
    def w(self):
        rs = self.br[0] - self.tl[0]
        if self.dtype == 'int':
            return int(rs)
        return rs

    @property
    def h(self):
        rs = self.br[1] - self.tl[1]
        if self.dtype == 'int':
            return int(rs)
        return rs

    @property
    def a(self):
        rs = self.w * self.h
        if self.dtype == 'int':
            return int(rs)
        return rs

    @property
    def center(self):
        rs = [self.tl[0] + self.w / 2, self.tl[1] + self.h / 2]
        if self.dtype == 'int':
            return [int(e) for e in rs]
        return rs

    def crop_center_square(self, keep_min=True):
        if keep_min:
            r = min(self.w, self.h) / 2
        else:
            r = max(self.w, self.h) / 2
        return Roi(tl=(self.center[0] - r, self.center[1] - r),
                   br=(self.center[0] + r, self.center[1] + r), dtype=self.dtype)

    def crop_top(self):
        return Roi(tl=self.tl, br=(self.tl[0] + self.w, self.tl[1] + self.w), dtype=self.dtype)

    def intersection(self, another=None, img=None):
        bbox = None

        if img is not None:
            bbox = ut.bb_intersection(self.bbox, [0, 0, img.shape[1] - 1, img.shape[0] - 1])
        elif another is not None:
            bbox = ut.bb_intersection(self.bbox, another.bbox)

        if bbox is not None:
            return Roi(bbox=bbox, dtype=self.dtype)
        else:
            return None

    def resize(self, sx, sy=None):
        """ returns new Roi with extended size, same center """
        c = self.center
        if sy is None:
            sy = sx
        s = (self.w / 2 * sx, self.h / 2 * sy)
        tl = (c[0] - s[0], c[1] - s[1])
        br = (c[0] + s[0], c[1] + s[1])
        return Roi(tl=tl, br=br, dtype=self.dtype)

    def crop(self, cv_rect):
        """ crops this using coordinates relative to this top-left """
        tl = self.tl
        new_tl = (tl[0] + cv_rect[0], tl[1] + cv_rect[1])
        new_br = (new_tl[0] + cv_rect[2], new_tl[1] + cv_rect[3])
        return Roi(tl=new_tl, br=new_br)

    def contains_bb_center(self, bb):
        return ut.is_in_roi(bb, self.tl, self.br)

    def translate(self, dx=0, dy=0):
        new_tl = (self.tl[0] + dx, self.tl[1] + dy)
        new_br = (self.br[0] + dx, self.br[1] + dy)
        return Roi(tl=new_tl, br=new_br, dtype=self.dtype)

    def margin(self, x, y=None):
        """ returns new Roi with extended size, same center """
        if y is None:
            y = x
        return Roi(tl=(self.tl[0] - x, self.tl[1] - y), br=(self.br[0] + x, self.br[1] + y), dtype=self.dtype)


from watchdog.events import FileSystemEventHandler
import os


class DictMapFile(FileSystemEventHandler):
    """
    A small table/dict that is mapped to a file so that 
    whenever the file updates, the table/dict is updated
    """

    def __init__(self, path):
        super(DictMapFile, self).__init__()
        if os.path.exists(path):
            self.filepath = path
            self.lines = []
            self.dict = {}
            self.handle_all('', '')
        else:
            open(path, 'a').close()

    def _update_dict(self):
        pass

    def _get_lines(self):
        self.lines = ut.read_lines(self.filepath)
        self.lines = [l for l in self.lines if len(l.strip()) > 0]

    def handle_all(self, event, op):
        self._get_lines()
        self._update_dict()

    def on_moved(self, event):
        if os.path.abspath(event.dest_path).endswith(os.path.basename(self.filepath)):
            self.handle_all(event, 'MOV')


class ObjDet(object):
    """ Single detection result """

    def __init__(self, bb, crop=None, norm=None, tag=None, landmarks=None, obj_class='obj', confidence=1.):
        self.bb = bb
        self.landmarks = landmarks
        self.crop = crop
        self.norm = norm
        self.obj_class = obj_class
        self.confidence = confidence
        if isinstance(tag, dict):
            self.tag = tag
        else:
            self.tag = {}

    @property
    def rf_score(self):
        return self.tag.get('liveliness', None)

    @rf_score.setter
    def rf_score(self, val):
        self.tag['liveliness'] = val

    def to_json(self):
        final = {}
        if isinstance(self.tag, dict):
            final = self.tag
        if isinstance(self.bb, np.ndarray):
            final['bb'] = self.bb.astype(int).tolist()
        else:
            final['bb'] = [int(e) for e in self.bb]
        final['obj_class'] = self.obj_class
        final['confidence'] = float(self.confidence)
        return final


class ObjDets(object):
    """ Detection result after scanning an image, a list of Det """

    def __init__(self, objDets):
        self.records = objDets

    @property
    def crops(self):
        return [f.crop for f in self.records]

    @crops.setter
    def crops(self, val):
        for i in range(len(self.records)):
            self.records[i].crop = val[i]

    @property
    def norms(self):
        return [f.norm for f in self.records]

    @norms.setter
    def norms(self, val):
        for i in range(len(self.records)):
            self.records[i].norm = val[i]

    @property
    def bboxes(self):
        return [f.bb for f in self.records]

    @bboxes.setter
    def bboxes(self, val):
        for i in range(len(self.records)):
            self.records[i].bb = val[i]

    @property
    def obj_classes(self):
        return [f.obj_class for f in self.records]

    @obj_classes.setter
    def obj_classes(self, val):
        for i in range(len(self.records)):
            self.records[i].obj_class = val[i]

    @property
    def confidences(self):
        return [f.confidence for f in self.records]

    @confidences.setter
    def confidences(self, val):
        for i in range(len(self.records)):
            self.records[i].confidence = val[i]

    @property
    def tags(self):
        return [f.tag for f in self.records]

    @tags.setter
    def tags(self, val):
        for i in range(len(self.records)):
            self.records[i].tag = val[i]

    @property
    def landmark_sets(self):
        return [f.landmarks for f in self.records]

    @landmark_sets.setter
    def landmark_sets(self, val):
        for i in range(len(self.records)):
            self.records[i].landmarks = val[i]

    @property
    def is_empty(self):
        return (self.records is None) or len(self.records) < 1

    def filter(self, keep):
        self.records = [r for r, k in zip(self.records, keep) if k is True]

    @property
    def biggest(self):
        return sorted(self.records, key=lambda x: ut.bb_area(x.bb))[-1]

    @property
    def size(self):
        return len(self.records)

    def filter_keep_topk_biggest(self, topk=128):
        """ sort records: biggest face is indexed at 0, keep topk"""
        self.records = sorted(self.records, key=lambda x: ut.bb_area(x.bb), reverse=True)[:topk]

    def to_json(self):
        return [r.to_json() for r in self.records]


class FallDet(ObjDet):
    def __init__(self, bb, crop=None, norm=None, tag=None, landmarks=None, is_fallen=0, confidence=1.):
        super().__init__(
            bb,
            crop=crop,
            norm=norm,
            tag=tag,
            landmarks=landmarks,
            obj_class='family',
            confidence=confidence,
        )
        self.is_fallen = is_fallen

    def to_json(self):
        final = {}
        if isinstance(self.tag, dict):
            final = self.tag
        if isinstance(self.bb, np.ndarray):
            final['bb'] = self.bb.astype(int).tolist()
        else:
            final['bb'] = [int(e) for e in self.bb]
        final['obj_class'] = self.obj_class
        final['is_fallen'] = self.is_fallen
        final['confidence'] = float(self.confidence)
        return final


class FallDets(ObjDets):
    def __init__(self, fallDets):
        super().__init__(fallDets)
        self.fall_dets = fallDets

class FireDet(ObjDet):
    def __init__(self, bb, crop=None, norm=None, tag=None, landmarks=None, is_fire=0, confidence=1.):
        super().__init__(
            bb,
            crop=crop,
            norm=norm,
            tag=tag,
            landmarks=landmarks,
            obj_class='fire',
            confidence=confidence,
        )
        self.is_fire = is_fire
# Lưu các thuộc tính của kết quả cần dùng
class FamilyDet(ObjDet):
    def __init__(self, bbox_human, bbox_face, stranger, confidence):
        # super().__init__(
        #     confidence=confidence,
        # )
        self.bbox_human = bbox_human,
        self.bbox_face = bbox_face,
        self.confidence = confidence,
        self.stranger = stranger

    def to_json(self):
        # print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
        # print(self.confidence[0])
        final = {}
        # if isinstance(self.bb, np.ndarray):
        #     final['bbox_human'] = self.bbox_human.astype(int).tolist()
        # else:
        #     final['bbox_human'] = [int(e) for e in self.bb]
        final = {
            'bbox_human': [int(e) for e in self.bbox_human[0]],
            'bbox_face': [int(e) for e in self.bbox_face[0]],
            'stranger': self.stranger,
            'confidence': float(self.confidence[0]),
        }
        return final



class FamilyDets(ObjDets):
    def __init__(self, familyDets):
        super().__init__(familyDets)
        self.family_dets = familyDets


    def to_json(self):
        final = {}
        if isinstance(self.tag, dict):
            final = self.tag
        if isinstance(self.bb, np.ndarray):
            final['bb'] = self.bb.astype(int).tolist()
        else:
            final['bb'] = [int(e) for e in self.bb]
        final['obj_class'] = self.obj_class
        final['is_fire'] = self.is_fire
        final['confidence'] = float(self.confidence)
        return final

class FireDets(ObjDets):
    def __init__(self, fireDets):
        super().__init__(fireDets)
        self.fire_dets = fireDets

# FamilyDet Class : lưu output
class FamilyDet(ObjDet):
    def __init__(self, bb, crop=None, norm=None, tag=None, landmarks=None, is_fallen=0, confidence=1.):
        super().__init__(
            bb,
            crop=crop,
            norm=norm,
            tag=tag,
            landmarks=landmarks,
            obj_class='family',
            confidence=confidence,
        )
        self.is_fallen = is_fallen

    def to_json(self):
        final = {}
        if isinstance(self.tag, dict):
            final = self.tag
        if isinstance(self.bb, np.ndarray):
            final['bb'] = self.bb.astype(int).tolist()
        else:
            final['bb'] = [int(e) for e in self.bb]
        final['obj_class'] = self.obj_class
        final['is_fallen'] = self.is_fallen
        final['confidence'] = float(self.confidence)
        return final


class FamilyDets(ObjDets):
    def __init__(self, fallDets):
        super().__init__(fallDets)
        self.fall_dets = fallDets

######################################################################
class HandDet(ObjDet):
    def __init__(self, bb, crop=None, norm=None, tag=None, landmarks=None, gesture=0, confidence=1.):
        super().__init__(
            bb,
            crop=crop,
            norm=norm,
            tag=tag,
            landmarks=landmarks,
            obj_class='hand',
            confidence=confidence
        )
        self.gesture = gesture

    def to_json(self):
        final = {}
        if isinstance(self.tag, dict):
            final = self.tag
        if isinstance(self.bb, np.ndarray):
            final['bb'] = self.bb.astype(int).tolist()
        else:
            final['bb'] = [int(e) for e in self.bb]
        final['obj_class'] = self.obj_class
        final['gesture'] = self.gesture
        final['confidence'] = float(self.confidence)
        return final


class HandDets(ObjDets):
    def __init__(self, handDets):
        super().__init__(handDets)


class ObjReg(ObjDet):
    """ Single face recognition result for a face detection """

    def __init__(self, parent=None, emb=None, match_dist=None, match_id=None):
        if parent is not None:  # copy all values from parent instance
            self.__dict__.update(parent.__dict__)

        self.emb = emb
        self.match_dist = match_dist
        self.match_id = match_id

    def to_json(self):
        final = super(ObjReg, self).to_json()
        final['emb'] = self.emb
        final['match_id'] = self.match_id
        final['match_dist'] = self.match_dist
        return final


class ObjRegs(ObjDets):
    """ Face recognition result after scanning an image, a list of FaceReg """

    def __init__(self, aObjDets=None, listObjReg=None):
        if listObjReg is None:
            if aObjDets:
                listObjReg = [ObjReg(d) for d in aObjDets.records]
            else:
                listObjReg = []
        super(ObjRegs, self).__init__(listObjReg)

    @property
    def embs(self):
        return [f.emb for f in self.records]

    @embs.setter
    def embs(self, val):
        for i in range(len(self.records)):
            self.records[i].emb = val[i]

    @property
    def match_ids(self):
        return [f.match_id for f in self.records]

    @match_ids.setter
    def match_ids(self, val):
        for i in range(len(self.records)):
            self.records[i].match_id = val[i]

    @property
    def match_dists(self):
        return [f.match_dist for f in self.records]

    @match_dists.setter
    def match_dists(self, val):
        for i in range(len(self.records)):
            self.records[i].match_dist = val[i]


# region ExpiringDict
# ==============================================================================
import time
from threading import RLock
import sys
from typing import Any, Union
from collections import OrderedDict


class ExpiringDict(OrderedDict):
    def __init__(self, max_len, max_age_seconds, items=None):
        # type: (Union[int, None], Union[float, None], Union[None,dict,OrderedDict,ExpiringDict]) -> None

        if not self.__is_instance_of_expiring_dict(items):
            self.__assertions(max_len, max_age_seconds)

        OrderedDict.__init__(self)
        self.max_len = max_len
        self.max_age = max_age_seconds
        self.lock = RLock()

        if sys.version_info >= (3, 5):
            self._safe_keys = lambda: list(self.keys())
        else:
            self._safe_keys = self.keys

        if items is not None:
            if self.__is_instance_of_expiring_dict(items):
                self.__copy_expiring_dict(max_len, max_age_seconds, items)
            elif self.__is_instance_of_dict(items):
                self.__copy_dict(items)
            elif self.__is_reduced_result(items):
                self.__copy_reduced_result(items)

            else:
                raise ValueError('can not unpack items')

    def __contains__(self, key):
        """ Return True if the dict has a key, else return False. """
        try:
            with self.lock:
                item = OrderedDict.__getitem__(self, key)
                if time.time() - item[1] < self.max_age:
                    return True
                else:
                    del self[key]
        except KeyError:
            pass
        return False

    def __getitem__(self, key, with_age=False):
        """ Return the item of the dict.

        Raises a KeyError if key is not in the map.
        """
        with self.lock:
            item = OrderedDict.__getitem__(self, key)
            item_age = time.time() - item[1]
            if item_age < self.max_age:
                if with_age:
                    return item[0], item_age
                else:
                    return item[0]
            else:
                del self[key]
                raise KeyError(key)

    def __setitem__(self, key, value, set_time=None):
        """ Set d[key] to value. """
        with self.lock:
            if len(self) == self.max_len:
                if key in self:
                    del self[key]
                else:
                    try:
                        self.popitem(last=False)
                    except KeyError:
                        pass
            if set_time is None:
                set_time = time.time()
            OrderedDict.__setitem__(self, key, (value, set_time))

    def pop(self, key, default=None):
        """ Get item from the dict and remove it.

        Return default if expired or does not exist. Never raise KeyError.
        """
        with self.lock:
            try:
                item = OrderedDict.__getitem__(self, key)
                del self[key]
                return item[0]
            except KeyError:
                return default

    def ttl(self, key):
        """ Return TTL of the `key` (in seconds).

        Returns None for non-existent or expired keys.
        """
        key_value, key_age = self.get(key, with_age=True)  
        if key_age:
            key_ttl = self.max_age - key_age
            if key_ttl > 0:
                return key_ttl
        return None

    def get(self, key, default=None, with_age=False):
        """ Return the value for key if key is in the dictionary, else default. """
        try:
            return self.__getitem__(key, with_age)
        except KeyError:
            if with_age:
                return default, None
            else:
                return default

    def items(self):
        """ Return a copy of the dictionary's list of (key, value) pairs. """
        r = []
        for key in self._safe_keys():
            try:
                r.append((key, self[key]))
            except KeyError:
                pass
        return r

    def items_with_timestamp(self):
        """ Return a copy of the dictionary's list of (key, value, timestamp) triples. """
        r = []
        for key in self._safe_keys():
            try:
                r.append((key, OrderedDict.__getitem__(self, key)))
            except KeyError:
                pass
        return r

    def values(self):
        """ Return a copy of the dictionary's list of values.
        See the note for dict.items(). """
        r = []
        for key in self._safe_keys():
            try:
                r.append(self[key])
            except KeyError:
                pass
        return r

    def fromkeys(self):
        """ Create a new dictionary with keys from seq and values set to value. """
        raise NotImplementedError()

    def iteritems(self):
        """ Return an iterator over the dictionary's (key, value) pairs. """
        raise NotImplementedError()

    def itervalues(self):
        """ Return an iterator over the dictionary's values. """
        raise NotImplementedError()

    def viewitems(self):
        """ Return a new view of the dictionary's items ((key, value) pairs). """
        raise NotImplementedError()

    def viewkeys(self):
        """ Return a new view of the dictionary's keys. """
        raise NotImplementedError()

    def viewvalues(self):
        """ Return a new view of the dictionary's values. """
        raise NotImplementedError()

    def __reduce__(self):
        reduced = self.__class__, (self.max_len, self.max_age, ('reduce_result', self.items_with_timestamp()))
        return reduced

    def __assertions(self, max_len, max_age_seconds):
        self.__assert_max_len(max_len)
        self.__assert_max_age_seconds(max_age_seconds)

    @staticmethod
    def __assert_max_len(max_len):
        assert max_len >= 1

    @staticmethod
    def __assert_max_age_seconds(max_age_seconds):
        assert max_age_seconds >= 0

    @staticmethod
    def __is_reduced_result(items):
        if len(items) == 2 and items[0] == 'reduce_result':
            return True
        return False

    @staticmethod
    def __is_instance_of_expiring_dict(items):
        if items is not None:
            if isinstance(items, ExpiringDict):
                return True
        return False

    @staticmethod
    def __is_instance_of_dict(items):
        if isinstance(items, dict):
            return True
        return False

    def __copy_expiring_dict(self, max_len, max_age_seconds, items):
        # type: (Union[int, None], Union[float, None], Any) -> None
        if max_len is not None:
            self.__assert_max_len(max_len)
            self.max_len = max_len
        else:
            self.max_len = items.max_len

        if max_age_seconds is not None:
            self.__assert_max_age_seconds(max_age_seconds)
            self.max_age = max_age_seconds
        else:
            self.max_age = items.max_age

        [self.__setitem__(key, value, set_time) for key, (value, set_time) in items.items_with_timestamp()]

    def __copy_dict(self, items):
        # type: (dict) -> None
        [self.__setitem__(key, value) for key, value in items.items()]

    def __copy_reduced_result(self, items):
        [self.__setitem__(key, value, set_time) for key, (value, set_time) in items[1]]


# ==============================================================================
# endregion


# threading with return value
# ===============================================================================
from threading import Thread


class ThreadWithReturnValue(Thread):

    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)
        self._return = None

    def run(self):
        try:
            if self._target:
                self._return = self._target(*self._args, **self._kwargs)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def join(self, timeout=None):
        Thread.join(self, timeout)
        return self._return
