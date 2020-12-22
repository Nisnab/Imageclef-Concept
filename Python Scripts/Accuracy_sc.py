import pandas as pd
gt= pd.read_csv('/home/udas/udas/ImageCLEF-ConceptDetection-Evaluation/groundtruth.txt',sep='\t',header=None)
r=gt.iloc [:,1].values.tolist()
pf= pd.read_csv('/home/udas/scratch/dropout_variation/drop_0.2and_0.3/abc.csv',sep='\t',header=None)
c=pf.iloc [:,1].fillna("")
d=c.values.tolist()
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