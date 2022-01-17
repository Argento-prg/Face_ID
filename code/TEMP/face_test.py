from deepface import DeepFace
import datetime

start = datetime.datetime.now()
result = DeepFace.verify("../images_know/papa2.jpg", "../images_unknow/papa2.jpg", 'Facenet')
end = datetime.datetime.now()

print(result)
print(str(end - start))