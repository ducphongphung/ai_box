from ultralytics import YOLO
import cv2
from src.utils.types_ex import *

output_width = 1200
output_height = 680


class FallDetector(object):
    def __init__(self):
        self.model = self._load_model()


    def _load_model(self):
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(device)
        model = YOLO("C:/Users\ducph\PycharmProjects/aibox\src\cv_core/fall\model8s_50.pt")
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



    def get_fall(self, bgr):
        results = self._detect(bgr)
        conf, statuses = self._add_fall_label(results)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        obj_dets = []
        for status, cf, box in zip(statuses, conf, boxes):
            obj_dets.append(FallDet(bb=box, confidence = cf, is_fallen=status))
        return FallDets(obj_dets)



if __name__ == "__main__":
    fall_detector = FallDetector()

    video = cv2.VideoCapture(0)
    # video.open("rtsp://admin:Facenet2022@192.168.1.3:554/cam/realmonitor?channel=1&subtype=1")
    if not video.isOpened():
        raise IOError("Cannot open webcam")
    fps = video.get(cv2.CAP_PROP_FPS)
    print('frames per second =', fps)
    print("Start detecting")
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (1200, 680))

    while True:
        ret, frame = video.read()
        if ret:
            fall_dets = fall_detector.get_fall(frame)
            for i, fall_det in enumerate(fall_dets.fall_dets):
                print(f"Detection {i + 1}:")
                print(f"  Box: {fall_det.bb}")
                print(f"  Confidence: {fall_det.confidence}")
                print(f"  Is Fallen: {'Yes' if fall_det.is_fallen == 1 else 'No'}")

            # out.write(fall_detector.get_fall(frame))
            # cv2.imshow("Output", fall_detector.get_fall(frame))
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     print("Exit")
            #     break
        else:
            print("Fail to detect frame")
            break
