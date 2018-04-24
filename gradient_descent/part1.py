import numpy as np
import math
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

#calculate information gain for class
arr=np.zeros(11269)
file1=open("train.label","r")
count =0
for line in file1:
    val=int(line)
    matrix[count][61188]=val
    arr[count]=val
    count=count+1

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

#finding entropy of parent
totalcount=zerocount+onecount

entroparent= (-1*(float(zerocount)/totalcount)*math.log((float(zerocount)/totalcount)))+(-1*(float(onecount)/totalcount)*math.log(float(onecount)/totalcount))


#find entropy of a row as a feature store it in a dictionary as tuple

list1=[]
countvals=0
for row_values in range(0,61188):
    check = []
    check_dict={}
    zerocount=0
    onecount=0
    counts={}
    for g in range(0,11269):
        val=int(matrix[g][row_values])
        if(val in check):

            check_dict[val]=check_dict[val]+1
            if(arr[g]==0):
                counts[val][0]=counts[val][0]+1
            else:
                counts[val][1]=counts[val][1]+1


        else:
            check.append(val)
            counts[val]=[0,0]
            if (arr[g]==0):
                counts[val][0] = counts[val][0] + 1
            else:
                counts[val][1] = counts[val][1] + 1
            check_dict[val]=1

    entrochild=0
    infogain=0

    for i in check_dict:
        zeroval = counts[i][0]
        oneval = counts[i][1]
        totalval = zeroval + oneval
        enterochild = 0
        if (zeroval == 0 and oneval == 0):
            enterochild = 0
        elif (zeroval == 0):
            enterochild = enterochild + ((-1) * (float(oneval) / totalval) * math.log(float(oneval) / totalval))
        elif (oneval == 0):
            enterochild = enterochild + ((-1) * (float(zeroval) / totalval) * math.log((float(zeroval) / totalval)))
        else:
            enterochild = enterochild + ((-1) * (float(oneval) / totalval) * math.log(float(oneval) / totalval)) + ((-1) * (float(zeroval) / totalval) * math.log((float(zeroval) / totalval)))

        enterochild=enterochild*(float(totalval)/11269)
        infogain = enterochild + infogain


    infogain=entroparent-infogain
    list1.append((infogain,row_values))

    countvals=countvals+1
    with open("countfile.txt","w") as cf:
        cf.write(str(countvals)+"\n")

with open ("anslist1.txt","w") as fil:
    for p in list1:
        fil.write(str(p[0])+" "+str(p[1])+"\n")
listrev=sorted(list1,key=lambda x:(-x[0],x[1]))

list3 =listrev[:100]
with open("fileans.txt", "w") as f:
    for s in list3:
        f.write(str(s[0])+" "+str(s[1])+"\n")
