# -*- coding: utf-8 -*-
# @Time     : 2019/10/20 16:02
# @Author   : Heyangyang

import tensorflow as tf

import random
# print(tf.__version__)
#
# print(tf.test.is_gpu_available())

a = random.sample(range(1, 33), 6)
b = random.randint(1, 16)
print(a)
print(b)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )

# model.fit(x_train, y_train, epochs = 5)
# model.evaluate(x_test, y_test)
