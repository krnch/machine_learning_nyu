import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_lfw_people
import sklearn.decomposition as skd
import numpy as np
import math

lfw_people = fetch_lfw_people(min_faces_per_person=70)
n_samples, h, w = lfw_people.images.shape
npix = h * w
fea = lfw_people.data


def plt_face(x):


    global h, w
    plt.imshow(x.reshape((h, w)), cmap=plt.cm.gray)
    plt.xticks([])

plt.figure(figsize=(10, 20))
nplt = 4
for i in range(nplt):
    plt.subplot(1, nplt, i + 1)
    plt_face(fea[i])
plt.show()

plt_face(fea[3])
plt.show()
print fea.shape
print len(fea[1])

mean_ans=np.zeros(2914)
index=0
while True:
    val=0
    for i in range(0,1288):
        val=val+fea[i][index]

    mean_ans[index]=val/1288

    index=index+1
    if(index==2913):
        break
print mean_ans
plt_face(mean_ans)
plt.show()

####################
import sklearn.decomposition as skd

pca = skd.PCA(n_components = 5)
skd.PCA.fit(pca,fea)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(fea)

index=0
print "lol"
print Z[3]
print W1
##
while True:
    val=0
    for i in range(2914):
        val=val+math.pow(W1[index][i],2)
    for i in range(2914):
        W1[index][i]=W1[index][i]/math.sqrt(val)
    index=index+1
    if(index==4):
        break
##
print W1

for i in range(5):
    z=np.dot(W1[i],fea[3])
    print z


###########################
X = pca.inverse_transform(Z)

mn = np.mean(fea, axis=0)
fourth_image=np.dot(pca.transform(fea)[:,:150], pca.components_[:150,:])

fourth_image=fourth_image+mn

answer=fourth_image[3]
plt_face(answer)
plt.show()