import numpy as np
import math
matrix =np.full((7505,61189 ), 0)

file=open("test.data","r")
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

matrix2=np.full((7505,101),0)
file11=open("test.label","r")
count =0
arr=np.zeros(7505)
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
    for h in range(0,7505):
        matrix2[h,count]=matrix[h,z]

    count=count+1
for h in range(0,7505):
    matrix2[h,100]=arr[h]

#matrix 2 consists of 100 features and the labels
#loadinf w array into a  list
warray=[]
with open("warray.txt", "r") as f:
  for line in f:
    pinr=line.strip()
    g=float(pinr)
    warray.append(g)

finalval=[]
for j in range(0,7505):
    val = 0
    for i in range(0,100):
        val=val+(warray[i+1]*matrix2[j][i])
    val=val+warray[0]
    yval = float(1) / (1 + math.exp((-1) * val))
    if(yval>0.5):
        finalval.append(1)
    else:
        finalval.append(0)

rightcount=0
wrongcount=0
nono=0
yesyes=0
yesno=0
noyes=0
for i in range(0,7505):
    if(finalval[i]==arr[i]):

        rightcount=rightcount+1
        if(arr[i]==0):
            nono=nono+1
        else:
            yesyes=yesyes+1

    else:
        wrongcount=wrongcount+1
        if(arr[i]==0):
            noyes=noyes+1
        else:
            yesno=yesno+1


print "Confusion Matrix"
print "total=7505\t"+"predicted-\t"+"predicted+"
print "actual-\t   "+str(nono)+"    \t"+str(noyes)
print "actual+\t   "+str(yesno)+"    \t"+str(yesyes)

precision=float(yesyes)/(yesyes+noyes)
recall=float(yesyes)/(yesyes+yesno)

print"\nprecision:"+str(precision)
print"\nrecall:"+str(recall)

weightarr=[0]*101
for i in range(0,101):
    weightarr[i]=(warray[i],i)

z=sorted(weightarr,key=lambda x:(-x[0],x[1]))

print z[:3]