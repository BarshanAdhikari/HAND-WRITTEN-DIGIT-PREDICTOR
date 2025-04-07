
import tensorflow
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
import numpy as np
import os
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

(x_train,y_train),(x_test,y_test)=tensorflow.keras.datasets.mnist.load_data()

x_train=x_train/255
x_test=x_test/255

model=Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))

model.add(Dense(10,activation='softmax'))

model.summary()

model.compile(optimizer=Adam(learning_rate=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
h=model.fit(x_train,y_train,epochs=20,validation_split=0.2)

y_predict=model.predict(x_test)
y_predict=np.argmax(y_predict,axis=1)
accuracy=accuracy_score(y_test,y_predict)
print("\nThe accuracy of the model is:-",accuracy*100)

for i in range(5):
  y_pre=model.predict(x_test)
  y_predict=np.argmax(y_pre,axis=1)
  print(y_predict[i])
  plt.imshow(x_test[i])
  plt.show()