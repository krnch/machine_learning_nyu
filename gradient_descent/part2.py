import numpy as np
import math
import random
matrix =np.full((11269,61189 ), 0)

file=open("train.data","r")
for line in file:
    val = line.split(" ")

    docid=int(val[0])
    termid=int(val[1])
    count=int(val[2])
    if(count>0):
        count=1
    else:
        count=0
    #store docid termid and count
    matrix[docid-1][termid-1]=count

matrix2=np.full((11269,101),0)
file11=open("train.label","r")
count =0
arr=np.zeros(11269)
for line in file11:
    val=int(line)
    arr[count]=val
    matrix[count][61188]=val
    count=count+1
#open file with the column numbers of first
lol=[]
with open("fileans.txt", "r") as f:
  for line in f:
    pinr=line.strip()
    g=pinr.split(" ")

    z=float(g[0])
    p=int(g[1])
    lol.append((z,p))

zerocount=0
onecount=0
for i in range(0,len(arr-1)):

    val3=[12,13,14,15]
    if(arr[i] in val3 ):
        arr[i]=1
        onecount=onecount+1

    else:
        arr[i]=0
        zerocount=zerocount+1

count=0
for val in lol:
    z=val[1]
    for h in range(0,11269):
        matrix2[h,count]=matrix[h,z]

    count=count+1
for h in range(0,11269):
    matrix2[h,100]=arr[h]
# gradient descent
warray= [0.0]*101
delwarray= [0.0]*101

for i in range(0,101):
    warray[i]=random.uniform(-0.01,0.01)

iter=0
ans=0
while(True):
    for j in range(0,101):
        delwarray[j]=0
    for t in range(0,11269):
        oval=0
        for j in range(1,100):
            oval=oval+(warray[j]*matrix2[t,j])
        oval=oval+warray[0]*1  #for wo

        yval=float(1)/(1+math.exp((-1)*oval))
        for j in range(1,100):
            delwarray[j]=delwarray[j]+ ((matrix2[t,100] - yval)*matrix2[t,j])
        #for wo
        delwarray[0]=delwarray[0]+((matrix2[t,100] - yval))

    oldarray=[0.0]*101
    subarray=[0.0]*101


    for i in range(0,101):
        oldarray[i]=warray[i]

    eta=0.001
    for j in range(0,101):
        warray[j]=warray[j]+eta*delwarray[j]

    check=0
    for j in range(0,101):
        if(oldarray[j]!=warray[j]):
            check=1
    for g in range(0,101):
        subarray[g]=warray[g]-oldarray[g]
    oldans=ans
    ans=0
    for g in range(0,101):
        ans=ans+subarray[g]
    with open("filediff.txt", "a") as fil:
        fil.write(str(ans)+"\n")
        if(oldans>ans):
            fil.write("good\n")
        else:
            fil.write("bad\n")

    iter=iter+1
    if(iter==300):
        break
    if(check==0):
        break

with open ("warray.txt" ,"w") as war:
    for s in warray:
        war.write(str(s)+"\n")
