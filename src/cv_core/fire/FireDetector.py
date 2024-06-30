import sys

from ultralytics import YOLO
import cv2
from src.utils.types_ex import *

output_width = 1200
output_height = 680


class FireDetector(object):
    def __init__(self):
        self.model = self._load_model()


    def _load_model(self):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = os.path.abspath('src/app_core/models/best.pt')
        model = YOLO( model_path)
        return model

    def _detect(self, bgr):
        results = self.model.predict(bgr, save=False)
        return results

    def _gen_response(self):
        pass

    def _add_fire_label(self, results):
        statuses = []
        conf = []
        for result in results:
            for box in result.boxes:
                if box.conf > 0.3 and (box.cls == 0 or box.cls == 1):  # Assuming class 0 is 'Fire'
                    statuses.append(1)
                    conf.append(box.conf)
                    # self.fall_count += 1  # Increment fire count
                else:
                    conf.append(0)
                    statuses.append(0)
        return conf, statuses



    def get_fire(self, bgr):
        results = self._detect(bgr)
        conf, statuses = self._add_fire_label(results)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        obj_dets = []
        for status, cf, box in zip(statuses, conf, boxes):
            obj_dets.append(FireDet(bb=box, confidence = cf, is_fire=status))
        return FireDets(obj_dets)



if __name__ == "__main__":
    fire_detector = FireDetector()


