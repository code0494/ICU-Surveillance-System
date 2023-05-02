import os
import cv2
import pickle
import face_recognition


def update_face(dataset, frame, name, det_mthd='hog'):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model=det_mthd)

    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding in encodings:
        dataset['face_data'].append(encoding)
        dataset['face_name'].append(name)

    return dataset


def fac_recog(test_frame, dataset):
    rgb = cv2.cvtColor(test_frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')
    encodings = face_recognition.face_encodings(rgb, boxes)
    names = []
    for encoding in encodings:
        matches = face_recognition.compare_faces(dataset['face_data'], encoding, 0.5)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matchedIdxs:
                name = dataset['face_name'][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
        names.append(name)
    return names


dataset = {"face_data": list(), "face_name": list()}
path = r'list4'
for (root, dirs, files) in os.walk(path):
    if root.endswith('10'):
        print(root)
        name = root.split(os.path.sep)[-2]
        for file in files:
            frame = cv2.imread(root + '\\' + file)
            print('processing : ', name, file)
            update_face(dataset, frame, name)
print(len(dataset["face_name"]))
f = open('faces_dump', "wb")
f.write(pickle.dumps(dataset))
f.close()

correct_name = 0
incorrect_name = 0
print('recognising')
for (root, dirs, files) in os.walk(path):
    if root.endswith('TEST'):
        name = root.split('\\')[-2]
        for file in files:
            frame = cv2.imread(root + '\\' + file)
            rec_name = fac_recog(frame, dataset)
            if len(rec_name) > 0 and rec_name[0] == name:
                correct_name = correct_name + 1
            else:
                incorrect_name = incorrect_name + 1
                print('incorrect : ', len(rec_name), name, file)

print(correct_name, incorrect_name)


