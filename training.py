
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
import os
## """iNSIDE ABSPATH ONLY ONE . REFERS TO CURRENT DIRECTORY """
ROOT_DIR = os.path.abspath("./")

import glob

import os

import argparse

parser=argparse.ArgumentParser(description="ETL--Extract, Transform, Load")
parser.add_argument('-tr','--train',type=str,required=True,help='filepath of Training.csv')
parser.add_argument('-vl','--valid',type=str,required=True,help='filepath of validation.csv')
parser.add_argument('-c','--concept',type=str,required=True,help='filepath of strings concept- This file contains all the concepts with their corresponding meaning')
parser.add_argument('-lr','--learningrate',type=float,default=1e-5,required=True,help='learning rate')
parser.add_argument('-ep','--epoch',type=int,required=True,help='Epoch--')
parser.add_argument('-imgz','--imagesize',type=int,default=150,help='Image size, both in height and width')
parser.add_argument('-b','--batch_size',type=int,default=16,help='batch size--Please make sure that the epoch is in power of 2')

args=parser.parse_args()

def append_ext(fn):
    return fn+".jpg"

import os


from keras import models
from keras.applications.xception import Xception

import tensorflow as tf

callbacks_list= [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                        mode='min',
                       verbose=1,
                       patience=10)
]


def etl(train,valid,concept,batch_size,epoch,learningrate,imagesize):
    df= pd.read_csv(train,sep='\t',names=["filename", "class"])
    df["filename"]=df["filename"].apply(append_ext)
    df["class"]=df["class"].apply(lambda x:x.split(";"))    
    #print('no. of training images:',len(df))

    
    ef= pd.read_csv(valid,sep='\t',names=["filename", "class"])
    ef["filename"]=ef["filename"].apply(append_ext)
    ef["class"]=ef["class"].apply(lambda x:x.split(";"))    
    #print('no. of validation images:',len(ef))

    ff= pd.read_csv(concept,sep='\t',header=None)
    r=ff.iloc [:,0].values.tolist()
    #print('Total No. of Concepts:',len(r))
     
    
    TEST_DIR = os.path.join(ROOT_DIR, "test-set/*.jpg") 
    file_names = glob.glob(TEST_DIR) # it will give list of file_names ['a.png','b.png']
    images = [i.split("/")[-1]for i in file_names]
    hf = pd.DataFrame(images,columns=['picture'])
    #hf = pd.DataFrame(images,columns=['filename'])
    #print(hf.head())
   
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
    directory=os.path.join(ROOT_DIR, "training-set"),   
    x_col="filename",
    y_col="class",
    color_mode="rgb",
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    class_mode="categorical",
    classes=r[:],
    target_size=(imagesize,imagesize)
    )

    valid_generator=test_datagen.flow_from_dataframe(
    dataframe=ef,
    directory=os.path.join(ROOT_DIR, "validation-set"),   
    x_col="filename",
    y_col="class",
    color_mode="rgb",
    batch_size=batch_size,
    shuffle=True,
    seed=42,
    class_mode="categorical",
    classes=r[:],
    target_size=(imagesize,imagesize)
    )



 

    conv_base=Xception(weights='imagenet',include_top=False,input_shape=(imagesize,imagesize,3))
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
    model.compile(optimizers.Adam(lr=learningrate),loss='binary_crossentropy',metrics=["accuracy"])


    STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size    
    history=model.fit_generator(generator=train_generator,
                                steps_per_epoch=STEP_SIZE_TRAIN,
                                validation_data=valid_generator,
                                validation_steps=STEP_SIZE_VALID,
                                callbacks=callbacks_list,
                                epochs=epoch)
    
    model.save('./my_model.h5')  

if __name__=='__main__':
    print(etl(args.train,args.valid,args.concept,args.batch_size,args.epoch,args.learningrate,args.imagesize))







