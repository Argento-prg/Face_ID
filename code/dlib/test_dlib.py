import os

import dlib
import datetime
import cv2
from scipy.spatial import distance



sp = dlib.shape_predictor('../../models/shape_predictor_5_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('../../models/dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

path = '../../new_datasets/oleg/f0.jpg'
img = cv2.imread(path)
start = datetime.datetime.now()
faces_1 = detector(img, 1)
shape = sp(img, faces_1[0])
face_chip = dlib.get_face_chip(img, shape)
descriptor = facerec.compute_face_descriptor(face_chip)
end = datetime.datetime.now()
print(str(end - start))


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        start = datetime.datetime.now()
        img = frame
        faces_1 = detector(img, 1)
        shape = sp(img, faces_1[0])
        face_chip = dlib.get_face_chip(img, shape)
        face_descriptor = facerec.compute_face_descriptor(face_chip)
        res = distance.euclidean(descriptor, face_descriptor)
        if res > 0.6:
            print(res)
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
        end = datetime.datetime.now()
        print(str(end - start))
cap.release()
cv2.destroyAllWindows()

