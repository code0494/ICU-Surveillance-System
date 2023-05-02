import cv2
import time
import pickle
import imutils
import face_recognition


class Face:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.data = pickle.loads(open('faces_dump', 'rb').read())
        self.faces_encodings = self.data["face_data"]
        self.encoded_names = self.data["face_name"]

    def face_det(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=3, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = imutils.resize(frame, height=600)
        return frame, faces

    def rec_face(self, frame):
        encodings = face_recognition.face_encodings(frame, model='small')
        
        name = 'Unknown'
        for encoding in encodings:
            matches = face_recognition.compare_faces(self.faces_encodings, encoding, 0.4)

            if True in matches:
                matched_idxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                for i in matched_idxs:
                    name = self.encoded_names[i]
                    counts[name] = counts.get(name, 0) + 1
                name = max(counts, key=counts.get)
        
        return name
