#random forest

#  importing necessary libraries
HEADERS=['Summons Number', 'Registration State', 'Plate Type', 'Issue Date', 'Vehicle Body Type', 'Vehicle Make', 'Issuing Agency', 'Street Code1', 'Street Code2', 'Street Code3', 'Violation Precinct', 'Issuer Precinct', 'Issuer Code', 'Violation Time', 'Front Of Or Opposite', 'From Hours In Effect', 'To Hours In Effect', 'Vehicle Year', 'Feet From Curb', 'Violation Code']
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, test_x, train_y, test_y
    """

    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage)
    return train_x, test_x, train_y, test_y

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

oneHotEncoding=['Vehicle Body Type','Issuing Agency','Front Of Or Opposite','Plate Type', 'Registration State']

columns=['Summons Number', 'Registration State', 'Plate Type', 'Issue Date', 'Vehicle Body Type', 'Vehicle Make', 'Issuing Agency', 'Street Code1', 'Street Code2', 'Street Code3', 'Violation Precinct', 'Issuer Precinct', 'Issuer Code', 'Violation Time', 'Front Of Or Opposite', 'From Hours In Effect', 'To Hours In Effect', 'Vehicle Year', 'Feet From Curb', 'Violation Code']

finalColumn=[x for x in columns if x not in columnDrop]
normalizedColumn=[x for x in finalColumn if x not in oneHotEncoding]
# print normalizedColumn

#normalizing

modified_df=normalizeData(df,normalizedColumn)
new_df=encodeData(modified_df,oneHotEncoding)
print list(new_df)
new_df['Violation Time'].fillna(value=0,inplace=True)
new_df['From Hours In Effect'].fillna(value=0,inplace=True)
new_df['To Hours In Effect'].fillna(value=0,inplace=True)

columnDrop.append('Registration State_99')
columnDrop.append('Plate Type_999')
feat1= new_df.drop(columnDrop, axis=1)
features=feat1.values

labels = df["Violation Code"].astype(int).values

print features.shape
pca = skd.PCA(n_components =30)
val=skd.PCA.fit(pca,features)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(features)
print Z.shape
plt.plot(np.cumsum(skd.PCA().fit(features).explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.show()
# X -> features, y -> label
X= Z
y= labels

# dividing X, y into train and test data
train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=0)
# Train and Test dataset size details
print "Train_x Shape :: ", train_x.shape
print "Train_y Shape :: ", train_y.shape
print "Test_x Shape :: ", test_x.shape
print "Test_y Shape :: ", test_y.shape

# Create random forest classifier instance
trained_model = random_forest_classifier(train_x, train_y)
print "Trained model :: ", trained_model
predictions = trained_model.predict(test_x)

for i in xrange(0, 5):
    print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])

print "Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))
print "Test Accuracy  :: ", accuracy_score(test_y, predictions)
print " Confusion matrix ", confusion_matrix(test_y, predictions)

