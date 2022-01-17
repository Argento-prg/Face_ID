import dlib
import datetime
import cv2
from scipy.spatial import distance



sp = dlib.shape_predictor('../../models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('../../models/dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

path = '../../new_datasets/oleg/f0.jpg'

img = cv2.imread(path)
start = datetime.datetime.now()
faces_1 = detector(img, 1)
shape = sp(img, faces_1[0])
face_chip = dlib.get_face_chip(img, shape)
face_descriptor1 = facerec.compute_face_descriptor(face_chip)
end = datetime.datetime.now()
print(str(end - start))

path = '../../new_datasets/oleg/f1.jpg'

img = cv2.imread(path)
start = datetime.datetime.now()
faces_1 = detector(img, 1)
shape = sp(img, faces_1[0])
face_chip = dlib.get_face_chip(img, shape)
face_descriptor2 = facerec.compute_face_descriptor(face_chip)
end = datetime.datetime.now()
print(str(end - start))

a = distance.euclidean(face_descriptor1, face_descriptor2)
print(a)

