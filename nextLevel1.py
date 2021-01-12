import cv2
import numpy as np
from matplotlib import pyplot
from os import listdir
import pathlib
from pathlib import Path
import numpy as np
from numpy import asarray
from numpy import savez_compressed
#load the model and txt file
caffeModel = "C:/Users/LENOVO/Desktop/KMC Internship/releaseTheKraken/res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "C:/Users/LENOVO/Desktop/KMC Internship/releaseTheKraken/deploy.prototxt.txt"

net=cv2.dnn.readNetFromCaffe(prototextPath,caffeModel)

def extract_face(filename,required_size=(160,160)):
    img=cv2.imread(filename)
    (h,w)=img.shape[:2]
    blob=cv2.dnn.blobFromImage(img,1.0,(img.shape[1],img.shape[0]),(104.0, 177.0, 123.0))
    net.setInput(blob)
    detections=net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face=img[startY:endY,startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,required_size)

            return face


def load_faces(directory):
    faces=list()

    #enumerate files
    for filename in listdir(directory):
        #path
        path=directory+filename
        #get face
        face=extract_face(path)
        #store
        faces.append(face)

    return faces

def load_dataset(directory):
    X,y=list(),list()

    #enumerate folders on per class
    for subdir in listdir(directory):
        #path
        path=directory+subdir+'/'

        #skip any files that might be in the dir
        #if not Path.is_dir(path):
            #continue

        #load all faces in the subdirectory
        faces=load_faces(path)

        #create labels
        labels=[subdir for _ in range(len(faces))]

        #summarize progress
        print('> loaded %d examples for class: %s' %(len(faces),subdir))

        #store
        X.extend(faces)
        y.extend(labels)

    return asarray(X),asarray(y)

#We can then call the above function for the Train and Validation folders to load all the data and then save the results in a single numpy array compressed file

#load train dataset

trainX,trainy=load_dataset("E:/FaceRecognitionFaceNet2.0/dataset/")

print(trainX.shape,trainy.shape)

#load test dataset
#testX,testy=load_dataset("C:/Users/LENOVO/Downloads/Compressed/archive/val/")
#print(testX.shape,testy.shape)

#save arrays to one file in a compressed format
#savez_compressed('5-celebrity-faces-dataset.npz', trainX, trainy, testX, testy)

#Created dataset by running this code is ready to be provided to a face detection model


