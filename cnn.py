import sys
from datetime import datetime
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from collections import Counter
from PIL import Image, ImageFont, ImageDraw

from keras.preprocessing import image
import matplotlib.pyplot as plt
import operator

from sklearn.metrics import classification_report, confusion_matrix


filename = "train"

import glob
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# a function to train the model and get the results
# Params: the name of existing model
def train_model(modelname):
        model = load_model(modelname)
        train_datagen = ImageDataGenerator(
                rescale=1./255, # rescale pixel intensities by 1./255
                )

        validate_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                'classified/train',
                target_size=(150, 150),
                batch_size=128,
                class_mode='categorical',
                shuffle=True, 
                seed=42)

        validation_generator = validate_datagen.flow_from_directory(
                'classified/test',
                target_size=(150, 150),
                batch_size=128,
                class_mode='categorical',
                shuffle=True, 
                seed=42)
        test_generator = test_datagen.flow_from_directory(
                directory='validate',
                target_size=(150, 150),
                batch_size=1,
                class_mode=None,
                shuffle=False,
                seed=42
                )

        print(train_generator.n)
        print(validation_generator.n)
        print(test_generator.n)
        path = "validate\*.*"
        for file in glob.glob(path):
        #Opening The Image
                prediction_image_path = file
                img_array = Image.open(prediction_image_path)
                print(list(validation_generator.class_indices.items())[0])
                labels = dict((v,k) for k,v in validation_generator.class_indices.items())
                print(labels)
                #Predicting the Image
                test_img=image.load_img(prediction_image_path,target_size=(150,150))
                test_img = np.expand_dims(test_img, axis = 0)
                result = model.predict(test_img)
                index, value = max(enumerate(result[0]), key=operator.itemgetter(1))
                if value > 0:
                        predictions = labels[index]
                        base_dir='classes'
                        full=os.path.join(base_dir, predictions)
                        full_path=os.path.join(full, file.split('\\')[1])
                        image.save_img(full_path,img_array)
                        im = Image.open(full_path)
                        d = ImageDraw.Draw(im)
                        d.text((10,10), predictions, fill=(255,255,255))
                        image.save_img(full_path,im)
        Y_pred = model.predict_generator(validation_generator, validation_generator.n // 128+1)
        y_pred = np.argmax(Y_pred, axis=1)
        cm = confusion_matrix(validation_generator.classes, y_pred)
        print(cm)
        print('Classification Report')
        print(classification_report(validation_generator.classes, y_pred, target_names=[k for (k,v) in validation_generator.class_indices.items()]))

        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        # Make various adjustments to the plot.
        plt.tight_layout()
        plt.colorbar()
        tick_marks = np.arange(8)
        print(validation_generator.class_indices.items())
        plt.xticks(tick_marks, [k for (k,v) in validation_generator.class_indices.items()])
        plt.yticks(tick_marks, [k for (k,v) in validation_generator.class_indices.items()])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        # Ensure the plot is shown correctly with multiple plots
        plt.title('Confusion matrix')
        plt.show()


# A function to create a model,
# Params: model name to be saved as modelname.h5
def create_model(modelname):
        train_datagen = ImageDataGenerator(
                rescale=1./255, # rescale pixel intensities by 1./255
                )

        validate_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        train_generator = train_datagen.flow_from_directory(
                'classified/train',
                target_size=(150, 150),
                # color_mode="rgb",
                batch_size=128, #32, 64, 128
                class_mode='categorical',
                # save_to_dir='temp_train',
                shuffle=True, # they need to match their id/foldername
                seed=42)

        validation_generator = validate_datagen.flow_from_directory(
                'classified/test',
                target_size=(150, 150),
                # color_mode="rgb",
                batch_size=128,
                class_mode='categorical',
                shuffle=True, # they need to match their id/foldername
                seed=42)
        test_generator = test_datagen.flow_from_directory(
                directory='validate',
                target_size=(150, 150),
                # color_mode="rgb",
                batch_size=1,
                class_mode=None,
                shuffle=False,
                seed=42)

        train_generator = train_datagen.flow_from_directory(
        'classified/train',
        target_size=(150, 150),
        # color_mode="rgb",
        batch_size=128, #32, 64, 128
        class_mode='categorical',
        # save_to_dir='temp_train',
        shuffle=True, # they need to match their id/foldername
        seed=42)
        # neural network model
        model = Sequential()
        model.add(Conv2D(16, (3,3), padding="same", use_bias=True, input_shape = (150, 150, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2,2), strides=None))

        model.add(Conv2D(32, (3,3), padding="valid", use_bias=True, activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=None))

        model.add(Conv2D(64, (3,3), padding="valid", use_bias=True, activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=None))

        model.add(Conv2D(128, (3,3), padding="valid", use_bias=True, activation = 'relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=None))

        model.add(Dense(128, activation = 'relu'))
        model.add(Flatten())
        model.add(Dense(8, activation = 'softmax'))

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
        model.compile(loss = 'categorical_crossentropy',
                optimizer = adam,
                metrics = ['accuracy'])

        model.fit_generator(
                train_generator,
                steps_per_epoch=200,
                epochs=3,
                validation_data=validation_generator,
                validation_steps=400,
        )

        #Printing the model summary 
        print(model.summary())

       # save models and close file for output
        model.save(modelname+'.h5')


if __name__ == '__main__':
        # create_model("newModel")
        train_model("train.h5")
        
