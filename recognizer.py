import numpy as np
import pickle
import cv2
import os
from sklearn.preprocessing import LabelEncoder
import numpy
import imutils
#load our serialized face detector from disk
protopath="E:/FaceRecognitionFaceNet2.0/face_detection_model/deploy.prototxt.txt"
modelpath="E:/FaceRecognitionFaceNet2.0/face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"

detector=cv2.dnn.readNetFromCaffe(protopath,modelpath)

#load embedding model from disk
embpath="C:/Users/LENOVO/Downloads/openface.nn4.small2.v1.t7"
embedder=cv2.dnn.readNetFromTorch(embpath)

#load the actual face recognition model along with the label encoder
import joblib
recognizer=pickle.load(open('finalized_model.sav','rb'))
#recognizer=joblib.load('finalized1_model.sav')
#le=numpy.load('classes.npy')
le=pickle.load(open('classes.pkl','rb'))
print(le)

#let's load our image and detect faces
image=cv2.imread(r"E:\Oh Faces\IMG_20191225_095938.jpg")
image=imutils.resize(image,width=600)
(h,w)=image.shape[:2]

#construct a blob from the image
imageBlob=cv2.dnn.blobFromImage(cv2.resize(image,(300,300)),1.0,(300,300),(104.0, 177.0, 123.0),swapRB=False, crop=False)

#apply opencv's deep learning detector to localize the faces in the input image
detector.setInput(imageBlob)
detections=detector.forward()

#print(detections)

#loop over the detections
for i in range(0,detections.shape[2]):
    #extract the confidence assosciated with the prediction
    confidence=detections[0,0,i,2]
    #print(confidence)

    #filter out weak detections
    if confidence>0.5:
        #compute the x,y coordinates for the bounding box of the face
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        #extract the face ROI
        face=image[startY:endY,startX:endX]

        (fH,fW)=face.shape[:2]

        #ensure that the face width and height are sufficiently large
        if fW < 20 or fH < 20:
            continue

        #construct a blob for the face ROI
        #then pass the blob through our embedding face model to obtain the 128-d embedding
        #quantification of the face

        faceBlob=cv2.dnn.blobFromImage(face,1.0/255,(96,96),(0,0,0),swapRB=True,crop=False)

        embedder.setInput(faceBlob)
        vec=embedder.forward()

        #perform classification to recognize the face
        preds=recognizer.predict_proba(vec)[0]
        print("preds: ",preds)
        j=np.argmax(preds)
        print("j: ",j)
        proba=preds[j]
        print("proba ",proba)

        #name=le.classes_[j]
        #print("name: ",name)

        prediction=recognizer.predict(vec)
        print("prediction: ",prediction)

        #drawing the bounding box
        text = "{}: {:.2f}%".format(prediction, proba * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10

        cv2.rectangle(image,(startX,startY),(endX,endY),(0,0,255),2)
        cv2.putText(image,text,(startX,y),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,0,255),2)

#show the image
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

