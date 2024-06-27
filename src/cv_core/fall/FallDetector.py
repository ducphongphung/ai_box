import sys
from get_output import ObjPred
from ultralytics import YOLO
import cv2
from src.utils.types_ex import *
import json
output_width = 1200
output_height = 680


class FallDetector(object):
    def __init__(self):
        self.model = self._load_model()


    def _load_model(self):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = os.path.abspath('src/app_core/models/model8s_50.pt')
        model = YOLO( model_path)
        return model

    def _detect(self, bgr):
        results = self.model.predict(bgr, save=False)
        return results

    def _gen_response(self):
        pass

    def _add_fall_label(self, results):
        statuses = []
        conf = []
        for result in results:
            for box in result.boxes:
                if box.conf > 0.3 and box.cls == 0:  # Assuming class 0 is 'Fall'
                    statuses.append(1)
                    conf.append(box.conf)
                    # self.fall_count += 1  # Increment fall count
                else:
                    conf.append(0)
                    statuses.append(0)
        return conf, statuses


    def write_to_file(self, new_data):
        file_path = 'output.txt'

        # Đọc nội dung hiện có của tệp
        with open(file_path, 'r') as file:
            existing_content = file.read()

        # Chuyển đổi nội dung hiện có thành danh sách (nếu có)
        try:
            data_list = json.loads(existing_content)
        except json.JSONDecodeError:
            data_list = []

        # Thêm dữ liệu mới vào danh sách
        data_list.append(new_data)

        # Chuyển danh sách sang định dạng JSON
        updated_content = json.dumps(data_list, indent=4)

        # Ghi nội dung mới vào tệp
        with open(file_path, 'w') as file:
            file.write(updated_content)

        print("Dữ liệu đã được thêm vào tệp thành công.")

    def get_fall(self, bgr, timestamp):
        results = self._detect(bgr)
        conf, statuses = self._add_fall_label(results)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        obj_dets = []
        obj_pred_list = []
        for status, cf, box in zip(statuses, conf, boxes):
            obj_dets.append(FallDet(bb=box, confidence = cf, is_fallen=status))
            label = 'Fall' if status == 1 else 'Not Fall'
            obj = ObjPred(label=label, confidence=cf, color=(0, 0, 255), rectangle=box)
            obj.write_to_file(obj.to_json()) 
        return FallDets(obj_dets)



if __name__ == "__main__":
    fall_detector = FallDetector()


