import os
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image
import numpy as np
from keras.preprocessing import image
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 6} )
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


#Inicializamos la red
classifier = Sequential()

#Primero realizamos la convolucion
classifier.add(Convolution2D(32, 3,3, input_shape=(64,64,3),activation='relu'))

#Ahora realizamos el Pooling
classifier.add(MaxPool2D(pool_size=(2,2)))

#Flattening
classifier.add(Flatten())

##Agregamos las dos ultimas capas con un funcion relu como activacion y una sigmoide
#para ver la probabilidad de que sea un gato o perro
classifier.add(Dense(output_dim = 128, activation='relu'))
classifier.add(Dense(output_dim = 3, activation='softmax'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Ingresamos las imagenes para el entrenamiento
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

trainning_set = train_datagen.flow_from_directory(
    'Dataset/training_set',
    target_size=(64,64),
    batch_size=32,
    class_mode='categorical'
)
test_set = test_datagen.flow_from_directory(
    'Dataset/test_set',
    target_size=(64, 64),
    batch_size=32,
        class_mode='categorical'
)

classifier.fit_generator(
    trainning_set,
    steps_per_epoch=8000,
    epochs=10,
    validation_data=test_set,
    validation_steps=8000
)

#Test
files = os.listdir("data/single_prediction")
print(trainning_set.class_indices)
for file in files:
    path = "Dataset/single_prediction/" + file
    test_image = image.load_img(path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    img_class = classifier.predict_classes(test_image)
    classname = img_class[0]
    if classname == 0:
        label = 'Cat'
    elif classname == 1:
        label = 'Dog'
    elif classname == 2:
        label = 'Flower'
    print("Imagen: ", file, " - Class: ", label)
