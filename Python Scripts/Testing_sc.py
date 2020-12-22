from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
import pandas as pd
import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator


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
print(len(hf)

datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
dataframe=hf,
directory="/home/udas/scratch/imageclefdata/Last/",
x_col="picture",
batch_size=1,
seed=42,
shuffle=False,
class_mode=None,
target_size=(150,150))


from keras.models import load_model
newmodel = load_model('/home/udas/scratch/projekt01/my_finalmodel.h5')

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


test_generator.reset()
pred=newmodel.predict_generator(test_generator,steps=STEP_SIZE_TEST,verbose=1)
import tensorflow as tf
from keras.layers import Lambda
def top_k(input, k):
  return tf.nn.top_k(input, k=k, sorted=True).values
# if we use .indices, it gives true for all when u print pred_bool
sorted_model = Sequential()
sorted_model.add(Lambda(top_k, input_shape=(5528,), arguments={'k':100}))
ans=sorted_model.predict(pred)
print(ans)
print(ans.shape)
print(ans[0].shape)
print(ans[0])

type(ans)


pred_bool = (ans >0.5)
print(pred_bool)

def top_k(input, k):
  return tf.nn.top_k(input, k=k, sorted=True).indices
model = Sequential()
model.add(Lambda(top_k, input_shape=(5528,), arguments={'k':100}))
sef=model.predict(pred)
print(sef)
print(sef.shape)
#print(sef.type)
print(sef[0])
#everything is fine



predictions=[]
labels = train_generator.class_indices
labels = dict((v,k) for k,v in labels.items())
for i in range(0,10000):
    l=[]
    for j in range(0,100):
        if pred_bool[i,j]==True:
             l.append(labels[sef[i,j]])
    predictions.append(";".join(l))

filenames=test_generator.filenames
results=pd.DataFrame({"Picture":filenames,
                      "predictions":predictions})
results['Picture'] = [x[:-4] for x in results['Picture']]
results=results.sort_values(["Picture"])
print(results.head())
print(results.shape)


results.to_csv("/home/udas/scratch/projekt01/myfinalresults.csv",index=False,sep='\t',header=0)
