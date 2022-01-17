import face_recognition
import cv2
import datetime

img = cv2.imread('../../images_know/f1.jpg')


start = datetime.datetime.now()

embedings1 = face_recognition.face_encodings(img)
print(len(embedings1[0]))
end = datetime.datetime.now()

print(end - start)

img = cv2.imread('../../images_know/f2.jpg')


start = datetime.datetime.now()

embedings2 = face_recognition.face_encodings(img)
print(len(embedings2[0]))
end = datetime.datetime.now()

print(end - start)
print(face_recognition.compare_faces([embedings1[0]], embedings2[0]))