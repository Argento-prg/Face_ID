import cv2
import numpy as np
import imagehash
from PIL import Image




aHash = []
pHash = []
dHash = []
wHash = []
for i in range(11):
    path = "C:\\Eugene\\" + str(i+1) + ".jpg"
    print(path)
    gray = Image.open(path).convert('LA')
    aHash.append(str(imagehash.average_hash(gray)))
    pHash.append(str(imagehash.phash(gray)))
    dHash.append(str(imagehash.dhash(gray)))
    wHash.append(str(imagehash.whash(gray)))

print(aHash)
print(pHash)
print(dHash)
print(wHash)

hamm = 0
for j in range(len(bin(int(pHash[0], base=16)))):
    hash1 = str(bin(int(pHash[1], base=16)))
    hash2 = str(bin(int(pHash[3], base=16)))
    if hash1[j] != hash2[j]:
        hamm += 1

print(hamm)