import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  
import DeepFace

class DemoModel(object):
    def __init__(self):
        self.net=cv2.dnn.readNetFromCaffe('src/models/deploy.prototxt.txt', 'src/models/res10_300x300_ssd_iter_140000.caffemodel')
        self.data = np.load('src/models/openface_embeddings.npz', allow_pickle=True)
        self.stored_embeddings = self.data['embeddings']
        self.stored_labels = self.data['labels']
        self.label_ids = self.data['label_ids'].item()

    def get_face_embedding(face_image):
        embedding_data = DeepFace.represent(np.array(face_image), model_name='Facenet512',enforce_detection=False)
        if embedding_data:
            return embedding_data[0]['embedding']
        return None

    def process_fire(self, frame, results, names):
        for r in results:
            for c in r.boxes:
                name = names[int(c.cls)]
                name = str(name).lower()
                if ((name == 'fire') & (float(c.conf) > 0.2)) | ((name == 'smoke') & (float(c.conf) > 0.5)):
                    x1, y1, x2, y2 = map(int, r.boxes.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # return frame

    def process_face(self,frame):
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Điều chỉnh ngưỡng confidence nếu cần
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = frame[startY:endY, startX:endX]

                face_embedding = DemoModel.get_face_embedding(face)

                if face_embedding is not None:
                    face_embedding = np.array([face_embedding])

                    # Compute similarity with stored embeddings
                    # distances = np.array([euclidean_l2_distance(face_embedding, stored_embedding) for stored_embedding in stored_embeddings])

                    # # Find the closest match
                    # best_match_idx = np.argmin(distances)
                    # best_match_score = distances[best_match_idx]
                    similarities = cosine_similarity(face_embedding, self.stored_embeddings)
                    best_match_idx = np.argmax(similarities)
                    best_match_score = similarities[0, best_match_idx]
                    # Set a similarity threshold for matching (adjust as needed)
                    if best_match_score > 0.6:
                        name = self.label_names[self.stored_labels[best_match_idx]]
                        text = f"Known ({best_match_score:.2f})"
                        color = (0, 255, 0)
                    else:
                        text = f"Unknown"
                        color = (0, 0, 255)

                    # Draw rectangle around the face and show the name
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.putText(frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)