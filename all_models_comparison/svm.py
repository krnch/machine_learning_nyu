# svm
from sklearn.model_selection import train_test_split
import csv
import sklearn.decomposition as skd
import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#############################################
#helper Methods

def encodeData(dataset, categories):
    return pd.get_dummies(dataset, columns=categories)


def normalizeData(dataset, categories):
    normData = dataset.copy()
    col = dataset[categories]
    col_norm = (col - col.min()) / (col.max() - col.min())
    normData[categories] = col_norm
    return normData


def evaluate(solutions, real):
    predictions = np.array(solutions)
    labels = np.array(real)
    return float((predictions == labels).sum()) / labels.size

#####################################################
# preprocessing for train data

ndf = pd.read_csv('train.csv')
df = ndf.drop(['Unnamed: 0'], axis=1)


columnDrop = ['Issue Date', 'Vehicle Make', 'Violation Code']

oneHotEncoding = ['Registration State', 'Vehicle Body Type', 'Issuing Agency', 'Front Of Or Opposite', 'Plate Type']

columns = ['Summons Number', 'Registration State', 'Plate Type', 'Issue Date', 'Vehicle Body Type', 'Vehicle Make',
           'Issuing Agency', 'Street Code1', 'Street Code2', 'Street Code3', 'Violation Precinct', 'Issuer Precinct',
           'Issuer Code', 'Violation Time', 'Front Of Or Opposite', 'From Hours In Effect', 'To Hours In Effect',
           'Vehicle Year', 'Feet From Curb', 'Violation Code']

finalColumn = [x for x in columns if x not in columnDrop]
normalizedColumn = [x for x in finalColumn if x not in oneHotEncoding]

modified_df = normalizeData(df, normalizedColumn)
new_df = encodeData(modified_df, oneHotEncoding)

new_df['Violation Time'].fillna(value=0, inplace=True)
new_df['From Hours In Effect'].fillna(value=0, inplace=True)
new_df['To Hours In Effect'].fillna(value=0, inplace=True)

columnDrop.append('Registration State_99')
columnDrop.append('Plate Type_999')
new_df11 = new_df.drop(columnDrop, axis=1)

p2=list(new_df11)

features = new_df11.values
labels = df["Violation Code"].astype(int).values

###################################################################
# PCA Calculation
pca = skd.PCA(n_components=50)
val = skd.PCA.fit(pca, features)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(features)

plt.plot(np.cumsum(skd.PCA().fit(features).explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()
################################################################
#preprocessing for test data

ndftest = pd.read_csv('test.csv')
z1 = list(ndftest)
df1 = ndftest.drop(['Unnamed: 0'], axis=1)

sumval=df1['Summons Number']
columnDrop1 = ['Issue Date', 'Vehicle Make']

oneHotEncoding1 = ['Registration State', 'Vehicle Body Type', 'Issuing Agency', 'Front Of Or Opposite', 'Plate Type']

columns1 = ['Summons Number', 'Registration State', 'Plate Type', 'Issue Date', 'Vehicle Body Type', 'Vehicle Make',
            'Issuing Agency', 'Street Code1', 'Street Code2', 'Street Code3', 'Violation Precinct', 'Issuer Precinct',
            'Issuer Code', 'Violation Time', 'Front Of Or Opposite', 'From Hours In Effect', 'To Hours In Effect',
            'Vehicle Year', 'Feet From Curb']

finalColumn1 = [x for x in columns1 if x not in columnDrop1]
normalizedColumn1 = [x for x in finalColumn1 if x not in oneHotEncoding1]
modified_df1 = normalizeData(df1, normalizedColumn1)
new_df1 = encodeData(modified_df1, oneHotEncoding1)

new_df1['Violation Time'].fillna(value=0, inplace=True)
new_df1['From Hours In Effect'].fillna(value=0, inplace=True)
new_df1['To Hours In Effect'].fillna(value=0, inplace=True)

columnDrop1.append('Registration State_99')
columnDrop1.append('Plate Type_999')

new_df111 = new_df1.drop(columnDrop1, axis=1)

p1=list(new_df111)


######################################
#ensuring same columns for test and train datasets

# find intersection
intxn = list(set(p1) & set(p2))

# find substraction to drop
sub1 = list(set(p1) - set(intxn))
test_df = new_df111.drop(sub1, axis=1)
test_features = test_df.values

sub2 = list(set(p2) - set(intxn))
train_df = new_df11.drop(sub2, axis=1)
train_features = train_df.values

features1 = new_df111.values


########################################################
# training a linear SVM classifier

from sklearn.svm import SVC

svm_model_linear = SVC(kernel='linear', C=1).fit(train_features, labels)
svm_predictions = svm_model_linear.predict(test_features)

train_labels_predicted = np.array(svm_predictions)
print train_labels_predicted
print len(train_labels_predicted)



with open("outputlabels.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerow(train_labels_predicted)

with open("outputsummons.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerow(sumval)

