import cv2
import torch
import imutils
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', force_reload=True)
def obj_detection(frame):
    results = model(imutils.rotate(frame, 0))
    frame = imutils.resize(np.squeeze(results.render()), width=600)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return( frame, ((results.__str__()).split('\n')[0]).split('854 ')[1])
    #return( frame, ((results.__str__()).split('\n')[0]))