import pandas as pd
import os
## """iNSIDE ABSPATH ONLY ONE . REFERS TO CURRENT DIRECTORY """
ROOT_DIR = os.path.abspath("./")
import argparse

parser=argparse.ArgumentParser(description="calculate accuracy--accuracy is calulated between common elements")
parser.add_argument('-gt','--groundtruth',type=str,required=True,help='filepath of groundtruth.csv')
parser.add_argument('-r','--result',type=str,required=True,help='filepath of predicteed result.csv')
args=parser.parse_args()

def acc(result,groundtruth):
	
	gt= pd.read_csv(groundtruth,sep='\t',header=None)
	r=gt.iloc [:,1].values.tolist()
	pf= pd.read_csv(result,sep='\t',header=None)
	c=pf.iloc [:,1].fillna("")
	d=c.values.tolist()
	#the number of test images in my case is 10000
	N=10000
	sum=0
	for i in range(0,N):
		A = r[i]
		B = d[i]
		AA = list(A.split(";"))
		BB = list(B.split(";"))
		y=[x for x in AA if x in BB]
		acc=len(y)/len(AA)
		sum=acc+sum
	average_accuracy=sum/N
	print("Average accuracy=",average_accuracy)

if __name__=='__main__':
    print(acc(args.result,args.groundtruth))







