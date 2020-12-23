from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
import pandas as pd
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
import glob
import os

import argparse
from keras.models import load_model
parser=argparse.ArgumentParser(description="test")
parser.add_argument('-b','--batch_size',type=int,default=1,help='batch size--Please make sure that the epoch is in power of 2')
parser.add_argument('-tr','--train',type=str,required=True,help='filepath of Training.csv')
parser.add_argument('-c','--concept',type=str,required=True,help='filepath of strings concept- This file contains all the concepts with their corresponding meaning')
args=parser.parse_args()

ROOT_DIR = os.path.abspath("./")

def append_ext(fn):
    return fn+".jpg"


#array = []
# fetch all images from your directory 
# I am assuming .png is the extension of images
def etl(batch_size,train,concept):

	df= pd.read_csv(train,sep='\t',names=["filename", "class"])
	df["filename"]=df["filename"].apply(append_ext)
	df["class"]=df["class"].apply(lambda x:x.split(";"))
	ff= pd.read_csv(concept,sep='\t',header=None)
	r=ff.iloc[:,0].values.tolist()
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
        fill_mode='nearest')

	train_generator=datagen.flow_from_dataframe(
	dataframe=df,
	directory=os.path.join(ROOT_DIR,"training-set"),   
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

	TEST_DIR = os.path.join(ROOT_DIR, "test-set/*.jpg") 
	file_names = glob.glob(TEST_DIR) # it will give list of file_names ['a.png','b.png']
	images = [i.split("/")[-1]for i in file_names]
	hf = pd.DataFrame(images,columns=['picture'])
	print(hf.head())
	test_datagen=ImageDataGenerator(rescale=1./255.)

	test_generator=test_datagen.flow_from_dataframe(
	dataframe=hf,
	directory=os.path.join(ROOT_DIR, "test-set"),   
	x_col="picture",
	batch_size=batch_size,
	seed=42,
	shuffle=False,
	class_mode=None,
	target_size=(150,150)
	)


	newmodel = load_model('./my_model.h5')
	STEP_SIZE_TEST=test_generator.n//test_generator.batch_size		
	print(STEP_SIZE_TEST)
	
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
	pred_bool = (ans >0.5)
	def top_k(input, k):

	  return tf.nn.top_k(input, k=k, sorted=True).indices

	model = Sequential()

	model.add(Lambda(top_k, input_shape=(5528,), arguments={'k':100}))

	sef=model.predict(pred)

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
	results.to_csv("./myfinalresults.csv",index=False,sep='\t',header=0)



if __name__=='__main__':
    print(etl(args.batch_size,args.train,args.concept))





	

