import cv2
import os
import numpy as np
from deepface import DeepFace


# Paths to models and training data
prototxt_path = 'src/models/deploy.prototxt.txt'
model_path = 'src/models/res10_300x300_ssd_iter_140000.caffemodel'
training_data_path = 'src/datasets'

# # Create DNN face detector
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Prepare training data storage
labels = []
faces_embeddings = []
current_id = 0
label_ids = {}

# Function to get embeddings via OpenFace
def get_face_embedding(face_image):
    embedding_data = DeepFace.represent(face_image, model_name='Facenet512',enforce_detection=False)
    if embedding_data:
        return embedding_data[0]['embedding']
    return None

# Walk through the training data path
for root, dirs, files in os.walk(training_data_path):
    for file in files:
        path = os.path.join(root, file)
        # print(path)
        label = os.path.basename(root).replace(" ", "-").lower()
        if label not in label_ids:
            label_ids[label] = current_id
            current_id += 1
        id_ = label_ids[label]

        img = cv2.imread(path)
        (h, w) = img.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                face = img[startY:endY, startX:endX]

                # Get embedding via DeepFace OpenFace model
        face_embedding = get_face_embedding(np.array(face))
        if face_embedding is not None:
            faces_embeddings.append(face_embedding)
            labels.append(id_)
            # print("a")

# Save embeddings and labels
np.savez('src/models/openface_embeddings.npz', embeddings=faces_embeddings, labels=labels, label_ids=label_ids)
print('Training done!')
