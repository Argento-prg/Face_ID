import face_recognition
import datetime
from scipy.spatial import distance
# from PIL import Image

path_know = "../images_know/f2.jpg"
path_unknow = "../images_unknow/f2.jpg"

start = datetime.datetime.now()
find_face = face_recognition.load_image_file(path_know)
face_encoding = face_recognition.face_encodings(find_face)[0]
print(datetime.datetime.now() - start)
# compare images
start = datetime.datetime.now()
unknown_picture = face_recognition.load_image_file(path_unknow)
unknown_face_encoding = face_recognition.face_encodings(unknown_picture)
print(len(unknown_face_encoding))
print(datetime.datetime.now() - start)

encoding = unknown_face_encoding[0]  # Обращаемся к 0 элементу, чтобы сравнить
start = datetime.datetime.now()
results = face_recognition.compare_faces([face_encoding], encoding)  # Сравниваем лица

if results[0]:
    print("Это тот же человек")

else:
    print("Это не одинаковые люди!")


print(datetime.datetime.now() - start)

start = datetime.datetime.now()
image = face_recognition.load_image_file(path_know)
face_locations = face_recognition.face_locations(image)
temp_1 = face_recognition.face_landmarks(image, face_locations)
print(datetime.datetime.now() - start)
print(face_locations)

start = datetime.datetime.now()
image = face_recognition.load_image_file(path_unknow)
face_locations = face_recognition.face_locations(image)
print(datetime.datetime.now() - start)
print(face_locations)
temp_2 = face_recognition.face_landmarks(image, face_locations)
print(temp_1)
print(temp_2)
print(distance.euclidean(temp_1, temp_2))