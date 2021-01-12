from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import pickle
from sklearn.preprocessing import LabelEncoder
from numpy import load
import joblib
import numpy

data=load('blaise-unknown-embeddings-names.npz')
trainX,trainy=data['arr_0'],data['arr_1']

print(trainX.shape)
print(trainy.shape)

le=LabelEncoder()
labels=le.fit_transform(trainy)

recognizer=SVC(C=1.0,kernel="linear",probability=True)
recognizer.fit(trainX,trainy)



#write the face recognition model to disk
filename='finalized_model.sav'
filename1='finalized1_model.sav'
joblib.dump(recognizer,filename1)
pickle.dump(recognizer,open(filename,'wb'))




#numpy.save('classes.npy', le.classes_)

output=open('classes.pkl','wb')
pickle.dump(labels,output)
output.close()

