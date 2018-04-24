import sklearn.decomposition as skd
import numpy as np
# .fit computes the principal components (n_components of them)
# The columns of W are the eigenvectors of the covariance matrix of X
X= np.array([[8,5,3],[2, 8, 10],[6,0,1],[8, 2, 6]])
pca = skd.PCA(n_components = 3)
skd.PCA.fit(pca,X)
W1 = pca.components_
W = W1.transpose()
Z = pca.transform(X)
print Z