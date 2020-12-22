
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
import pandas as pd
import numpy as np
from keras.optimizers import Adam
from sklearn.metrics import fbeta_score
import keras.applications
print("Donot forget that you have used csv files here")

from keras_preprocessing.image import ImageDataGenerator

df= pd.read_csv('/home/udas/scratch/imageclefdata/Training-Concepts.csv',sep=';',names=["filename", "class"])
ef= pd.read_csv('/home/udas/scratch/imageclefdata/Validation-Concepts.csv',sep=';',names=["image", "label"])
ff= pd.read_csv('/home/udas/scratch/imageclefdata/String-Concepts.csv',sep='\t',header=None)
print('no. of training images:',len(df))
print('no. of validation images:',len(ef))
r=ff.iloc [:,0].values.tolist()
print('Total No. of Concepts:',len(r))
print('Data Type of concepts:',type(r))
def append_ext(fn):
    return fn+".jpg"
df["filename"]=df["filename"].apply(append_ext)
ef["image"]=ef["image"].apply(append_ext)
df["class"]=df["class"].apply(lambda x:x.split(","))
print(df.head())

ef["label"]=ef["label"].apply(lambda x:x.split(","))
print(ef.head())
#the .jpeg is appeded here and string is converted to list

import glob
import pandas as pd
import os
#array = []
# fetch all images from your directory 
# I am assuming .png is the extension of images 
file_names = glob.glob('/home/udas/scratch/imageclefdata/Last/*.jpg') # it will give list of file_names ['a.png','b.png']
images = [i.split("/")[-1]for i in file_names]

hf = pd.DataFrame(images,columns=['picture'])
print(hf.head())
print(len(hf))
datagen=ImageDataGenerator(rescale=1./255.,
                            samplewise_center=True, 
                           samplewise_std_normalization=True,
                            height_shift_range=0.2,                    
                           width_shift_range=0.2,
                           brightness_range=[1.0,1.5],
                           zoom_range=0.5,
                           rotation_range=30,
                           horizontal_flip=True,
                           vertical_flip=True,
                           fill_mode='nearest'                    
                           )

test_datagen=ImageDataGenerator(rescale=1./255.)

train_generator=datagen.flow_from_dataframe(
dataframe=df,
directory="/home/udas/scratch/imageclefdata/training-set/",
x_col="filename",
y_col="class",
color_mode="rgb",
batch_size=16,
shuffle=True,
seed=42,
class_mode="categorical",
classes=r[:],
target_size=(150,150)
)

valid_generator=test_datagen.flow_from_dataframe(
dataframe=ef,
directory="/home/udas/scratch/imageclefdata/validation-set/",
x_col="image",
y_col="label",
color_mode="rgb",
batch_size=16,
shuffle=True,
seed=42,
class_mode="categorical",
classes=r[:],
target_size=(150,150)
)

test_generator=test_datagen.flow_from_dataframe(
dataframe=hf,
directory="/home/udas/scratch/imageclefdata/Last/",
x_col="picture",
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(150,150))

print(type(test_generator))
for x_col in test_generator:
    print(x_col.shape)
   
    break
print(test_generator.dtype)
epoch=75

from keras_contrib.callbacks import CyclicLR
clr_triangular2 = CyclicLR(base_lr=1e-5, max_lr=1e-2,
                    step_size=2000.,mode='triangular2')

callbacks_list=[
                #Earlystopping: Interrupting training when the validation loss is no longer
                #improving (and of course, saving the best model obtained during training)
        keras.callbacks.EarlyStopping(
                       monitor='val_loss',
                        mode='min',
                       verbose=1,
                       patience=10
                        ),
         clr_triangular2,
          keras.callbacks.ModelCheckpoint(
               filepath='/home/udas/scratch/dropout_variation/imagesize_255/model.h5',
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1
           ),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=10),
        keras.callbacks.CSVLogger('/home/udas/scratch/dropout_variation/imagesize_255/csvofrec.csv', 
            separator=',', 
            append=False),
        
        keras.callbacks.TensorBoard(
            log_dir='/home/udas/scratch/dropout_variation/imagesize_255/my_log_dir')

]

from keras import models
from keras.applications.xception import Xception

# Create the model
conv_base=Xception(weights='imagenet',include_top=False,input_shape=(150,150,3))
for layer in conv_base.layers[:-6]:
     layer.trainable = False
# Add the xception convolutional base model
model = models.Sequential()
model.add(conv_base)

model.add(MaxPooling2D(2,2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(5528, activation='sigmoid'))
model.compile(optimizers.Adam(lr=1e-5),loss='binary_crossentropy',metrics=["accuracy"])


STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size    
history=model.fit_generator(generator=train_generator,
                            steps_per_epoch=STEP_SIZE_TRAIN,
                            validation_data=valid_generator,
                            validation_steps=STEP_SIZE_VALID,
                            callbacks=callbacks_list,
                            epochs=epoch)

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

#Train and validation accuracy
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()


#Train and validation loss
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

import pandas as pd
rec= pd.read_csv('/home/udas/scratch/dropout_variation/imagesize_255/csvofrec.csv',sep=',')

import matplotlib.pyplot as plt
plt.plot(rec['epoch'], rec['lr'])
plt.ylabel('Learning Rate')
plt.xlabel('Epoch')
plt.title('Loss with respect to epoch')
plt.legend()
plt.show() 

plt.plot(rec['lr'], rec['acc'])
plt.ylabel('accuracy')
plt.xlabel('learning Rate')
plt.title('accuracy w.r.to learning rate')
plt.legend()
plt.show() 

