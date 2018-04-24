#decision tree
# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


ndf = pd.read_csv('train.csv')
z=list(ndf)
df=ndf.drop(['Unnamed: 0'],axis=1)

import sklearn.decomposition as skd

columnDrop =['Issue Date','Vehicle Make','Violation Code']

oneHotEncoding=['Registration State','Vehicle Body Type','Issuing Agency','Front Of Or Opposite','Plate Type']

columns=['Summons Number', 'Registration State', 'Plate Type', 'Issue Date', 'Vehicle Body Type', 'Vehicle Make', 'Issuing Agency', 'Street Code1', 'Street Code2', 'Street Code3', 'Violation Precinct', 'Issuer Precinct', 'Issuer Code', 'Violation Time', 'Front Of Or Opposite', 'From Hours In Effect', 'To Hours In Effect', 'Vehicle Year', 'Feet From Curb', 'Violation Code']

finalColumn=[x for x in columns if x not in columnDrop]
normalizedColumn=[x for x in finalColumn if x not in oneHotEncoding]
# print normalizedColumn

#normalizing

modified_df=normalizeData(df,normalizedColumn)
new_df=encodeData(modified_df,oneHotEncoding)

# 'Registration State_99'
# 'Plate Type_999'
columnDrop.append('Registration State_99')
columnDrop.append('Plate Type_999')
new_df['Violation Time'].fillna(value=0,inplace=True)
new_df['From Hours In Effect'].fillna(value=0,inplace=True)
new_df['To Hours In Effect'].fillna(value=0,inplace=True)

features = new_df.drop(columnDrop, axis=1).values
labels = df["Violation Code"].astype(int).values



print features.shape
pca = skd.PCA(n_components =15)
val=skd.PCA.fit(pca,features)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(features)
print Z.shape
# plt.plot(np.cumsum(skd.PCA().fit(features).explained_variance_ratio_))
# plt.xlabel('number of components')
# plt.ylabel('cumulative explained variance');
# plt.show()


# X -> features, y -> label
X= Z
y= labels

# dividing X, y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# training a DescisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

dtree_model = DecisionTreeClassifier(max_depth=10).fit(X_train, y_train)
dtree_predictions = dtree_model.predict(X_test)

# creating a confusion matrix
cm = confusion_matrix(y_test, dtree_predictions)

train_labels=np.array(y_test)
train_labels_predicted= np.array(dtree_predictions)

# print train_labels
# print train_labels_predicted
# print len(train_labels)
# print len(train_labels_predicted)

print evaluate(train_labels,train_labels_predicted)