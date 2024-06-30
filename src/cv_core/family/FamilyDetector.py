import sys

from ultralytics import YOLO
import cv2
from src.utils.types_ex import *
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import os

output_width = 1200
output_height = 680


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2

    xi1 = max(x1, x3)
    yi1 = max(y1, y3)
    xi2 = min(x2, x4)
    yi2 = min(y2, y4)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

class FamilyDetector(object):
    def __init__(self):
        self.prototxt_path = 'src/app_core/models/deploy.prototxt.txt'
        self.caffe_model_path = 'src/app_core/models/res10_300x300_ssd_iter_140000.caffemodel'
        self.training_data_path = 'src/uploads'
        self.tflite_model_path = 'src/app_core/models/model.tflite'
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.caffe_model_path)
        self.human_detector = YOLO('src/app_core/models/yolov8n.pt')
        self.stored_embeddings = np.load('src/app_core/models/face_embeddings.npz')['embeddings']
        # Tải mô hình TFLite
        self.interpreter = tf.lite.Interpreter(model_path=self.tflite_model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def preprocess_image(self, face_image):
        try:
            face_image = cv2.resize(face_image, (112, 112))
            face_image = face_image.astype('float32')
            face_image = face_image / 255.0  # Chuẩn hóa ảnh về khoảng [0, 1]
            face_image = np.expand_dims(face_image, axis=0)  # Thêm chiều batch
            return face_image
        except Exception as e:
            return None

    def get_face_embedding(self, face_image):
        # Tiền xử lý ảnh
        preprocessed_image = self.preprocess_image(face_image)
        
        if preprocessed_image is not None:
            # Thiết lập dữ liệu đầu vào
            self.interpreter.set_tensor(self.input_details[0]['index'], preprocessed_image)
            
            # Chạy mô hình
            self.interpreter.invoke()
            
            # Lấy kết quả đầu ra
            embedding_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            if embedding_data is not None:
                return embedding_data[0]
        return None

    def train_data(self):
        faces_embeddings = []

        # Đi qua các tệp dữ liệu huấn luyện
        for root, dirs, files in os.walk(self.training_data_path):
            for file in files:
                path = os.path.join(root, file)
                img = cv2.imread(path)
                (h, w) = img.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
                self.net.setInput(blob)
                detections = self.net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.4:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        face = img[startY:endY, startX:endX]

                        face_embedding = self.get_face_embedding(np.array(face))
                        if face_embedding is not None:
                            faces_embeddings.append(face_embedding)

        # Chuyển đổi danh sách thành numpy array
        faces_embeddings = np.array(faces_embeddings)

        # Lưu các vector nhúng
        np.savez('src/app_core/models/face_embeddings.npz', embeddings=faces_embeddings)
    
        print(f"Đã lưu vector nhúng từ dữ liệu huấn luyện.")
    
    def humandetect(self, frame):
        results = self.human_detector(frame)
        return results, frame

    def process_frame(self, results, frame):
        statuses = []
        conf = []
        humans_bbox = {}
        faces_bbox = []

        for result in results:
            for bbox in result.boxes:
                if int(bbox.cls) == 0:  # Process only human bounding boxes
                    x1, y1, x2, y2 = map(int, bbox.xyxy[0])
                    human_bbox = (x1, y1, x2, y2)  # Use (x1, y1, x2, y2) tuple for dictionary key

                    # Skip if this human_bbox is already processed or overlaps significantly with an existing bbox
                    skip = False
                    for existing_bbox in humans_bbox:
                        if calculate_iou(human_bbox, existing_bbox) > 0.8:
                            skip = True
                            break
                    if skip:
                        continue

                    roi = frame[y1:y2, x1:x2]

                    # Detect faces using DNN
                    (h, w) = roi.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                    self.net.setInput(blob)
                    detections = self.net.forward()
                    print(detections)

                    face_detected = False  # Flag to check if face has been detected for current human_bbox
                    for i in range(detections.shape[2]):
                        if face_detected:  # If a face has already been detected, break the loop
                            break

                        confidence = detections[0, 0, i, 2]
                        if confidence > 0.5:
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            startX += x1
                            startY += y1
                            endX += x1
                            endY += y1

                            face = frame[startY:endY, startX:endX]
                            face_bbox = [startX, endX, startY, endY]
                            # Obtain the face embedding
                            face_embedding = self.get_face_embedding(face)

                            if face_embedding is not None:
                                face_embedding = np.array([face_embedding])
                                similarities = cosine_similarity(face_embedding, self.stored_embeddings)
                                best_match_idx = np.argmax(similarities)
                                best_match_score = similarities[0, best_match_idx]

                                # Ensure that each face gets only one label
                                if face_bbox in faces_bbox:
                                    continue  # Skip if this face_bbox is already processed

                                if best_match_score > 0.3:
                                    statuses.append(1)
                                    conf.append(best_match_score)
                                else:
                                    statuses.append(0)
                                    conf.append(best_match_score)

                                faces_bbox.append(face_bbox)
                                humans_bbox[human_bbox] = face_bbox  # Store human_bbox with corresponding face_bbox
                                face_detected = True  # Mark that a face has been detected for this human_bbox
                                break  # Only take the first detected face for each human_bbox

        # Convert humans_bbox dictionary keys to list for returning
        humans_bbox_list = list(humans_bbox.keys())

        return conf, statuses, humans_bbox_list, faces_bbox

    def get_stranger(self, frame):
        results, frame = self.humandetect(frame)
        conf, statuses, humans_bbox, faces_bbox = self.process_frame(results, frame)
        obj_dets = []
        for status, cf, human_bb, face_bb in zip(statuses, conf, humans_bbox, faces_bbox):
            obj_dets.append(FamilyDet(bbox_human=human_bb, bbox_face=face_bb, confidence=cf, stranger=status))
        return FamilyDets(obj_dets)




if __name__ == "__main__":
    fall_detector = FamilyDetector()
    print(1)