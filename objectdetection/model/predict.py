import tensorflow as tf
import numpy as np
import os
import keras

model = tf.keras.models.load_model('model/saved_model/model')

def predict(image,model=model):
    # img = tf.keras.preprocessing.image.load_img(
    #     image, target_size=(32,32,3))
    img = tf.image.resize(image,(32,32))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.expand_dims(img, 0)
    pred = model.predict(img)
    score = tf.nn.softmax(pred[0])
    res = np.argmax(score)
    return res