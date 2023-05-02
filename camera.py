import cv2
import uuid

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('frame', frame)

    if cv2.waitKey(10) == 27:
        break
    elif cv2.waitKey(10) == ord('c'):
        cv2.imwrite('data\image' + str(uuid.uuid1()) + '.jpeg', frame)