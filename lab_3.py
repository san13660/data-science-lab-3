import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image # Read and resize image
# Import library and dataset
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import os

img_rows, img_cols = 56, 56 # number of pixels 
train_data = pd.read_csv('/kaggle/input/aptos2019-blindness-detection/train.csv')
train_images = []
train_diagnosis = []
for dirname, _, filenames in os.walk('/kaggle/input/aptos2019-blindness-detection/train_images/'):
    for filename in filenames:
        image = Image.open(os.path.join(dirname, filename))
        new_image = image.resize((img_rows, img_cols))
        new_image.save('train_'+filename)
        diagnosis = train_data.loc[train_data['id_code']+'.png' == filename,'diagnosis']
        train_images = train_images + [new_image]
        train_diagnosis = train_diagnosis + [diagnosis]

import tensorflow as tf # tensorflow 2.0
from keras.datasets import mnist
import numpy as np
seed=0
np.random.seed(seed) # fix random seed
tf.random.set_seed(seed)
num_classes = 10 # 10 digits

len(train_images)

import keras
train_diagnosis = keras.utils.to_categorical(np.array(train_diagnosis), 5)

train_diagnosis[0]

model = Sequential()#add model layers
model.add(Conv2D(32, kernel_size=(5, 5),
                     activation='relu',
                     input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
# add second convolutional layer with 20 filters
model.add(Conv2D(64, (5, 5), activation='relu'))
    
# add 2D pooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))
    
# flatten data
model.add(Flatten())
    
# add a dense all-to-all relu layer
model.add(Dense(1024, activation='relu'))
    
# apply dropout with rate 0.5
model.add(Dropout(0.5))
    
# soft-max layer
model.add(Dense(num_classes, activation='softmax'))

#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train the model
print('Realizando entrenamiento')
model.fit(train_images, train_diagnosis, validation_data=(test_images, test_diagnosis), epochs=3)

# evaluate the model
print('Realizando pruebas')
score = model.evaluate(test_images, test_diagnosis, verbose=1)

# print performance
print('Test accuracy:', score[1])