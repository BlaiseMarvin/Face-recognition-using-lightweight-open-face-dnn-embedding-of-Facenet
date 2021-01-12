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

#cap=cv2.VideoCapture(0)
from imutils.video import VideoStream
from imutils.video import FPS
import time

vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while(True):
    frame=vs.read()

    frame = imutils.resize(frame, width=600)
    (h, w) = frame.shape[:2]

    imageBlob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0),swapRB=False, crop=False)
    detector.setInput(imageBlob)
    detections = detector.forward()

    for i in range(0, detections.shape[2]):
        # extract the confidence assosciated with the prediction
        confidence = detections[0, 0, i, 2]
        # print(confidence)

        # filter out weak detections
        if confidence > 0.6:
            # compute the x,y coordinates for the bounding box of the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI
            face = frame[startY:endY, startX:endX]

            (fH, fW) = face.shape[:2]

            # ensure that the face width and height are sufficiently large
            if fW < 20 or fH < 20:
                continue

            faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)

            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # perform classification to recognize the face
            preds = recognizer.predict_proba(vec)[0]
            print("preds: ", preds)
            j = np.argmax(preds)
            print("j: ", j)
            proba = preds[j]
            print("proba ", proba)

            # name=le.classes_[j]
            # print("name: ",name)

            prediction = recognizer.predict(vec)
            print("prediction: ", prediction)

            # drawing the bounding box
            text = "{}: {:.2f}%".format(prediction, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    fps.update()
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.stop()



