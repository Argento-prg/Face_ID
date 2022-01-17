import dlib
import cv2
import datetime
import numpy as np
import imagehash
from PIL import Image
from scipy.spatial import distance
import os

def hemming_distance(p_hash, etalon):
    # считаем хеммингово расстояние между двумя хешами
    distance = 0
    cmp_p_hash = str(bin(int(p_hash, base=16)))
    cmp_etalon = str(bin(int(etalon, base=16)))
    for j in range(len(cmp_p_hash)):
        if cmp_etalon[j] != cmp_p_hash[j]:
            distance += 1
    return distance

def euclidean_distance(p_hash, etalon):
    # Евклидово расстояние
    cmp_p_hash = str(bin(int(p_hash, base=16)))
    cmp_etalon = str(bin(int(etalon, base=16)))
    t1 = []
    t2 = []
    for j in range(len(cmp_p_hash)):
        t1.append(int(cmp_p_hash[j], base=16))
        t2.append(int(cmp_etalon[j], base=16))
    dst = distance.euclidean(t1, t2)
    return dst

def putin():
    path = '../test/'
    files = os.listdir(path=path)
    for file in files:
        img = cv2.imread(os.path.join(path + file))
        img = cv2.resize(img, (160, 120))
        faces = detector(img, 1)
        for face in faces:
            landmarks = predictor(img, face)
            # t1 = facerec.compute_face_descriptor(img, landmarks)
            new_image = np.zeros([2, 68])
            for i in range(0, 68):
                new_image[0][i] = landmarks.part(i).x
                new_image[1][i] = landmarks.part(i).y
            temp_image = Image.fromarray(new_image)
            myhash = imagehash.phash(temp_image)
            print(file)
            print(myhash)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('../models/dlib_face_recognition_resnet_model_v1.dat')
path = '../images_know/f2.jpg'
'''
img = cv2.imread(path)
img = cv2.resize(img, (640, 480))
start = datetime.datetime.now()
faces = detector(img, 1)
for face in faces:
    landmarks = predictor(img, face)
    #t1 = facerec.compute_face_descriptor(img, landmarks)
    new_image = np.zeros([2, 68])
    for i in range(0, 68):
        new_image[0][i] = landmarks.part(i).x
        new_image[1][i] = landmarks.part(i).y
    temp_image = Image.fromarray(new_image)
    myhash = imagehash.phash(temp_image)
    print(myhash)
    end = datetime.datetime.now()
    print(end-start)
'''
putin()

print(euclidean_distance('a0c300c300c300c3', 'a0c300c300c30083'))
print(euclidean_distance('a0c300c300c300c3', 'a0c300c300c200c6'))
print(euclidean_distance('a0c300c300c300c3', 'a043004200420042'))
