import pickle 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
from tensorflow.keras import models,layers
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import ResNet50


def unpickle(file):
  with open(file,'rb') as fo:
    myDict = pickle.load(fo,encoding='latin1')
  return myDict

trainData = unpickle('cifar-100-python/train')
for item in trainData:
  print(item,type(trainData[item]))

print(len(trainData['data']))
print(len(trainData['data'][0]))

print(np.unique(trainData['fine_labels']))

print(np.unique(trainData['coarse_labels']))

print(trainData['batch_label'])
testData = unpickle('cifar-100-python/test')
metaData = unpickle('cifar-100-python/meta')

print("Fine labels: ",metaData['fine_label_names'],"\n")
print("coarse labels: ",metaData['coarse_label_names'])

category = pd.DataFrame(metaData['coarse_label_names'], columns=['SuperClass'])
subCategory = pd.DataFrame(metaData['fine_label_names'],columns=['SubClass'])
print(category)
print(subCategory)

X_train = trainData['data']
X_train

X_train = X_train.reshape(len(X_train),3,32,32).transpose(0,2,3,1)

def image_show():
  rcParams['figure.figsize'] = 2,2
  imageId = np.random.randint(0,len(X_train))
  plt.imshow(X_train[imageId])
  plt.axis('off')
  print("Image Number :",imageId)
  print("Shape of image :",X_train[imageId].shape)
  print("Image category :",category.iloc[trainData['coarse_labels'][imageId]][0])
  print("Image subcategory :",subCategory.iloc[trainData['fine_labels'][imageId]][0])
image_show()

X_test = testData['data']
X_test = X_test.reshape(len(X_test),3,32,32).transpose(0,2,3,1)

y_train = trainData['fine_labels']
y_test = testData['fine_labels']

n_classes = 100
y_train = to_categorical(y_train,n_classes)
y_test = to_categorical(y_test,n_classes)

res_model = ResNet50(input_shape=(32,32,3),include_top=False,weights="imagenet")

for layer in res_model.layers[:143]:
  layer.trainable = False

model = models.Sequential()
model.add(res_model)
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(100,activation='softmax'))

model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss = 'categorical_crossentropy',
              metrics=['accuracy'],
              )


earlystop_callback = EarlyStopping(
  monitor='val_accuracy', min_delta=0, patience=5)

history = model.fit(X_train,y_train,batch_size=32,epochs=50,validation_data=(X_test,y_test),callbacks=[earlystop_callback])
