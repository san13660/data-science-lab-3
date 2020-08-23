# Lab 3 - Data Science
# Christopher Sandoval
# Fernanda Estrada
# Luis Delgado
# Estuardo Diaz

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import os
import tensorflow as tf
from keras.datasets import mnist
import keras

img_rows, img_cols = 400, 400 # numero de pixeles despues de resize
train_data = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
train_images = []
train_diagnosis = []

# Se leen las fotos
for dirname, _, filenames in os.walk('/kaggle/input/aptos2019-blindness-detection/train_images/'):
    for filename in filenames:
        image = Image.open(os.path.join(dirname, filename))
        new_image = image.resize((img_rows, img_cols))
        new_image.save('train_'+filename)
        diagnosis = train_data.loc[train_data['id_code']+'.png' == filename,'diagnosis']
        train_images = train_images + [new_image]
        train_diagnosis = train_diagnosis + [diagnosis]


seed=0
np.random.seed(seed) # Se establece la semilla
tf.random.set_seed(seed)
num_classes = 5 # 5 clases

train_diagnosis = keras.utils.to_categorical(np.array(train_diagnosis), num_classes)

train_diagnosis[0]

# Se inicia el modelo
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Se agrega la capa conv
model.add(Conv2D(64, (5, 5), activation='relu'))
    
# Se agrega la capa de pooling
model.add(MaxPooling2D(pool_size=(2, 2)))
    
# Se hace el flattening
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
    
# Capa de softmax
model.add(Dense(num_classes, activation='softmax'))

# Se compila el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Se entrena el modelo
print('Realizando entrenamiento')
model.fit(train_images, train_diagnosis, validation_data=(test_images, test_diagnosis), epochs=3)

# Se evalua el modelo
print('Realizando pruebas')
score = model.evaluate(test_images, test_diagnosis, verbose=1)

print('Test accuracy:', score[1])
