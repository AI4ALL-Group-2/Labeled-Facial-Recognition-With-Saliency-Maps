import tensorflow as tf 
import pickle
import numpy as np

# This is used to predict a list of testing datas

model = tf.keras.models.load_model('C:/Users/syngu/GitHub/Facial-Recognition/Labeled-Facial-Recognition-With-Saliency-Maps/models/facial_recognition.keras')

X = open('C:/Users/syngu/GitHub/Facial-Recognition/Labeled-Facial-Recognition-With-Saliency-Maps/data/testing_feature.pickle','rb')
y = open('C:/Users/syngu/GitHub/Facial-Recognition/Labeled-Facial-Recognition-With-Saliency-Maps/data/testing_label.pickle','rb')
X = pickle.load(X)
y = pickle.load(y)

y = np.array(y)


X = X/255.0

loss, accuracy = model.evaluate(X, y)

print( accuracy)