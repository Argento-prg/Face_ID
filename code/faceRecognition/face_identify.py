import face_recognition
import datetime
from scipy.spatial import distance
import cv2
from PIL import Image
import imagehash
import numpy as np

start = datetime.datetime.now()

path_know = "../images_know/f1.jpg"
path_unknow = "../images_unknow/f4.jpg"

face_1 = face_recognition.load_image_file(path_know)

face_2 = face_recognition.load_image_file(path_unknow)

face_1 = cv2.resize(face_1, (640, 480))
face_2 = cv2.resize(face_2, (640, 480))

face_1_loc = face_recognition.face_locations(face_1)
face_2_loc = face_recognition.face_locations(face_2)

face_1_landmark = face_recognition.face_landmarks(face_1, face_1_loc)
face_2_landmark = face_recognition.face_landmarks(face_2, face_2_loc)

face_marks_list_1 = [[], []]
face_marks_list_2 = [[], []]

for i in face_1_landmark[0].keys():
    for j in face_1_landmark[0][i]:
        face_marks_list_1[0].append(j[0])
        face_marks_list_1[1].append(j[1])

for i in face_2_landmark[0].keys():
    for j in face_2_landmark[0][i]:
        face_marks_list_2[0].append(j[0])
        face_marks_list_2[1].append(j[1])

m1 = np.array(face_marks_list_1)
m2 = np.array(face_marks_list_2)

n1 = Image.fromarray(m1)
n2 = Image.fromarray(m2)

h1 = imagehash.phash(n1)
h2 = imagehash.phash(n2)


print(distance.euclidean(h1, h2))