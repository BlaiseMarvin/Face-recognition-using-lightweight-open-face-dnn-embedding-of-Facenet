import numpy as np
import pickle
import cv2
import os
import argparse
import imutils




#load our serialized face detector from disk

protopath="E:/FaceRecognitionFaceNet2.0/face_detection_model/deploy.prototxt.txt"
modelpath="E:/FaceRecognitionFaceNet2.0/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

detector=cv2.dnn.readNetFromCaffe(protopath,modelpath)

#load embedding model from disk
embpath="C:/Users/LENOVO/Downloads/openface.nn4.small2.v1.t7"
embedder=cv2.dnn.readNetFromTorch(embpath)

knownEmbeddings=[]
knownNames=[]

total=0
datasetpath="E:/FaceRecognitionFaceNet2.0/train/"

for directory in os.listdir(datasetpath):
    print(directory)
    newpath=datasetpath+directory+'/'
    for filename in os.listdir(newpath):
        #print(filename)
        imgpath=newpath+filename
        image=cv2.imread(imgpath)
        #image=cv2.resize(image,(600,600))
        image = imutils.resize(image, width=600)
        #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        (h,w)=image.shape[:2]

        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        detector.setInput(imageBlob)
        detections=detector.forward()

        #ensure that atleast one face was found
        if len(detections)>0:
            # we're making the assumption that each image has only ONE
            # face, so find the bounding box with the largest probability
            i = np.argmax(detections[0, 0, :, 2])
            confidence = detections[0, 0, i, 2]

            # ensure that the detection with the largest probability also
            # means our minimum probability test (thus helping filter out
            # weak detections
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                face = image[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                if fW < 20 or fH < 20:
                    continue

                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)

                embedder.setInput(faceBlob)
                vec=embedder.forward()

                knownEmbeddings.append(vec.flatten())
                knownNames.append(directory)
                total+=1


from numpy import asarray
from numpy import savez_compressed
ke=asarray(knownEmbeddings)
kn=asarray(knownNames)

savez_compressed('blaise-unknown-embeddings-names.npz',ke,kn)


