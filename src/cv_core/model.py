import cv2
from ultralytics import YOLO


class DemoModel(object):
    def __init__(self):
        self.model = YOLO("C:/Users\ducph\PycharmProjects/aibox\src\models\model8s_50.pt")

    def predict(self, image):
        results = self.model.predict(image, conf = 0.3, stream=True)
        return results

if __name__ == '__main__':

    demo = DemoModel()

    cap = cv2.VideoCapture(1)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("FPS: {}".format(fps))

    if not cap.isOpened():
        raise IOError("Cannot open camera")

    while True:
        ret, frame = cap.read()
        if ret:
            demo.predict(frame)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("capture failed")
            break
