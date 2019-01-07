
# coding: utf-8

# # Imports

# In[59]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging 
import csv
from sklearn import mixture
import itertools
from scipy import linalg
import matplotlib as mpl

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import *
from keras.wrappers.scikit_learn import KerasClassifier
import keras
import seaborn as sns
import random
import math
from sklearn import tree, preprocessing
from sklearn.neighbors import KNeighborsClassifier, DistanceMetric
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate, train_test_split
from sklearn.ensemble import AdaBoostClassifier
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback, CSVLogger
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import FastICA, PCA
from scipy.stats import kurtosis, skew
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import timeit

#from imblearn.over_sampling import RandomOverSampler


# # Functions

# In[ ]:


def MAPE(pred,act):
    return np.mean(np.abs(pred-act)/act)*100

def Pred_correct(pred,act):
    correct = np.abs(pred-act)
    return len(correct[correct==0])/len(act)

def convert_out(data):
    output = []
    for i in range(np.shape(data)[0]):
        loc = np.where(data[i,]==max(data[i,]))[0]
        output.append(loc[0]+1)
    return(output)

def create_model_nn(neurons=1,optimizer='adam'):
# create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=14, kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def create_model_tree():
    dtree = tree.DecisionTreeClassifier(criterion = "entropy", splitter = 'best', max_leaf_nodes = 90, min_samples_leaf = 1, max_depth=3)

    


# # Preprocessing data

# In[ ]:


data = pd.read_csv('leaf.csv', header=None)
data.drop([1],inplace=True,axis=1)
X = data.values[:,1:]
Y_first = data.values[:,0]
Y = []
out = np.zeros(shape=[36])
for i in range(np.shape(Y_first)[0]):
    out[int(Y_first[i])-1]=1
    Y.append(out)
    out = np.zeros(shape=[36])
Y = np.array(Y)
min_max_scaler = preprocessing.MinMaxScaler()
X_scaled = min_max_scaler.fit_transform(X)
df_normalized = pd.DataFrame(X_scaled)

r,c = np.shape(df_normalized)

total_samples = list(np.arange(0,r,1))
test_samples = random.sample(total_samples,int(np.round(len(total_samples)*0.2)))
train_samples = list(set(total_samples) - set(test_samples))


# # Splitting data

# In[ ]:


trainX = X[train_samples]
trainY = Y[train_samples]
trainY_comp = Y_first[train_samples]
trainX_scaled = X_scaled[train_samples]

testX = X[test_samples]
testY = Y[test_samples]
testY_comp = Y_first[test_samples]
testX_scaled = X_scaled[train_samples]

scalar = StandardScaler()
trainX_S = scalar.fit_transform(trainX)
testX_S = scalar.transform(testX)

values = random.sample(list(np.arange(0,len(trainX_S),1)),len(trainX_S))


# # NN

# In[5]:


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
    def on_epoch_end(self, epoch, logs):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        logs['test_acc'] = acc


    
train_all = []
test_all = []

trainX_use = trainX_S[values,:]
trainY_use = trainY[values,:]
trainY_comp_use = trainY_comp[values]

model = Sequential()
model.add(Dense(250, input_dim=14, kernel_initializer='normal', activation='relu'))
model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# checkpoint
filepath="weights.best_leaf.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
callbacks_list = [checkpoint, TestCallback((testX_S,testY)),CSVLogger('1.log')]
hist = model.fit(trainX_use[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)
plt.plot(100*np.array(1-np.array(hist.history['acc'])),'g')
plt.plot(100*np.array(1-np.array(hist.history['val_acc'])),'b')
plt.plot(100*np.array(1-np.array(hist.history['test_acc'])),'r')
plt.title('Neural Network Misclassification Error for Leaf Data Set')
plt.ylabel('Misclassification Error (%)')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set', 'Testing Set'])
plt.ylim([0,100])
plt.show()

train_pred_NN = convert_out(model.predict(trainX_use))
train_all.append(1-Pred_correct(train_pred_NN,trainY_comp_use))

test_pred_NN = convert_out(model.predict(testX_S))
test_all.append(1-Pred_correct(test_pred_NN,testY_comp))

print('Training Set Misclassification Error is', train_all)
print('Testing Set Misclassification Error is', test_all)    

model1 = Sequential()
model1.add(Dense(250, input_dim=14, kernel_initializer='normal', activation='relu'))
model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model1.load_weights("weights.best_leaf.h5")



train_pred = convert_out(model1.predict(trainX_use[:-82]))
train_act = trainY_comp_use[:-82]
train_acc_1= (1-Pred_correct(train_pred,train_act))

val_pred = convert_out(model1.predict(trainX_use[-82:]))
val_act = trainY_comp_use[-82:]
val_acc_1 = (1-Pred_correct(val_pred,val_act))

test_pred = convert_out(model1.predict(testX_S))
test_acc_1 = (1-Pred_correct(test_pred,testY_comp))

results1 = pd.DataFrame(data = np.transpose([np.array(train_acc_1)*100,np.array(val_acc_1)*100,np.array(test_acc_1)*100]))
print(results1)


# # K means elbow method

# In[219]:


sse = {}
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k,random_state=1).fit(trainX_S)
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()),'o')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.title('Elbow Method for Leaf Data Set')

plt.xticks(list(sse.keys()))
plt.legend()
plt.show()


# In[220]:


kmeans = KMeans(n_clusters=2,random_state=1).fit(trainX_S)
labels =  kmeans.predict(trainX_S)
res=kmeans.__dict__
c = ['red','green']
l = ['Cluster 1', 'Cluster 2']
for i in range(2):
    plt.scatter(trainX_S[:, 6][labels==i], trainX_S[:, 5][labels==i], color=c[i],
             label=l[i]);
plt.xlabel('Feature 2')
plt.ylabel('Feature 5')
plt.legend()
combined = np.hstack((trainX_S,np.reshape(labels,(-1,1))))
df = pd.DataFrame(data=combined)
sns.set(style="ticks")

sns.pairplot(df, hue=14)
plt.show()


# In[221]:


xplot = np.zeros((2,14))
xplot[:,:] = np.arange(0,14,1)
c=['red','green','blue']
l=['Cluster 1', 'Cluster 2', 'Cluster 3']
for i in range(np.shape(res['cluster_centers_'])[0]):
    plt.scatter(xplot[i,:],np.transpose(res['cluster_centers_'][i,:]),color = c[i],label=l[i])
    
plt.xticks(np.arange(0,14,1))
plt.xlabel('Feature Number')
plt.ylabel('Mean Value')
plt.title('Cluster Mean Values for Each Feature for Leaf Data Set')
plt.legend()
plt.show()


# In[222]:


idx_zero = np.where(kmeans.labels_==0)
zeros_array = trainX_S[idx_zero]
vars_zero = np.var(zeros_array,axis=0)
idx_ones = np.where(kmeans.labels_==1)
ones_array = trainX_S[idx_ones]
vars_ones = np.var(ones_array,axis=0)
c=['red','green','blue']
l=['Cluster 1', 'Cluster 2', 'Cluster 3']


plt.scatter(xplot[0,:],np.reshape(vars_zero,[1,-1]),color = c[0],label=l[0])
plt.scatter(xplot[1,:],np.transpose(vars_ones),color = c[1],label=l[1])

plt.xticks(np.arange(0,14,1))
plt.xlabel('Feature Number')
plt.ylabel('Variance')
plt.title('Cluster Variance Values for Each Feature for Leaf Data Set')
plt.legend()
plt.show()


# # K Means Silhouette

# In[207]:


sil_all = []

for n_cluster in range(2, 20):
    kmeans = KMeans(n_clusters=n_cluster,random_state=1).fit(trainX_S)
    label = kmeans.labels_
    sil_coeff = silhouette_score(trainX_S, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    sil_all.append(sil_coeff)
 
plt.scatter(np.arange(2,20,1),sil_all,)
plt.xticks(np.arange(2,20,1))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Score for Leaf Data Set')
plt.legend()
plt.show()  


# # EM + PCA

# In[12]:


gmm = GaussianMixture(n_components=len(unique_Y),random_state=1)
gmm.fit(X)
new_labels = gmm.predict(X)+1
c_size = np.zeros((2,len(unique_Y)))
for i in range(len(unique_Y)):
    idx = np.where(Y_first==unique_Y[i])[0]
    c_size[0,i] = len(Y_first[Y_first==unique_Y[i]])
    new_clusters = new_labels[idx]
    unique_list = (list(set(new_clusters)))
    nc_occ = [len(np.where(new_clusters==unique_list[k])[0]) for k in range(len(unique_list))]
    max_num = unique_list[np.where(nc_occ==np.max(nc_occ))[0][0]]
    c_size[1,i] = len(np.where(new_clusters==max_num)[0])
print(c_size)
print(np.sum(c_size,1))
avg = c_size[1,:]/c_size[0,:]
np.mean(avg)


# In[111]:


lowest_bic = np.infty

bic = []
n_components_range = range(1, 20)
cv_types = ['full']
#cv_types = ['spherical', 'tied', 'diag','full']

for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components,random_state=4)
    gmm.fit(trainX_S)
    bic.append(gmm.bic(trainX_S))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 0.05)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))

plt.xticks(n_components_range)
#plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model for Leaf Data Set')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos+0.5, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of Clusters')
#spl.legend([b[0] for b in bars], cv_types)

Y_ = clf.predict(trainX)
plt.show()

sil_all = []

for n_cluster in range(2, 20):
    clf1 = mixture.GaussianMixture(n_components=n_cluster,random_state=4).fit(trainX_S)
    label = clf1.predict(trainX_S)
    sil_coeff = silhouette_score(trainX_S, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    sil_all.append(sil_coeff)
 
plt.scatter(np.arange(2,20,1),sil_all,)
plt.xticks(np.arange(2,20,1))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Score for Leaf Data Set')
# # Plot the winner
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(trainX)

# for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X[Y_ == i, 5], X[Y_ == i, 6], .8, color=color)

#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[0][1], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)

# plt.xticks(())
# plt.yticks(())
# plt.title('Selected GMM: full model, 2 components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()


# # EM Plot (Means)

# In[114]:


# combined = np.hstack((trainX_scaled,np.reshape(Y_,(-1,1))))
# df = pd.DataFrame(data=combined)
# sns.set(style="ticks")

# sns.pairplot(df, hue=14)
# plt.show()

r,c = np.shape(clf.means_)
print(r)
print(c)
xplot = np.zeros((r,14))
xplot[:,:] = np.arange(0,14,1)
c=['red','green','blue','yellow','purple']
l=['Cluster 1', 'Cluster 2', 'Cluster 3','Cluster 4', 'Cluster 5']
for i in range(r):
    plt.scatter(xplot[i,:],np.transpose(clf.means_[i,:]),color = c[i],label=l[i])
    
plt.xticks(np.arange(0,14,1))
plt.xlabel('Feature Number')
plt.ylabel('Mean Value')
plt.title('Cluster Mean Values for Each Feature for Leaf Data Set')

plt.legend()
plt.show()


# # EM Plot (Deviation)

# In[115]:


r,c,n = np.shape(clf.covariances_)
print(r,c,n)
xplot = np.zeros((r,14))
xplot[:,:] = np.arange(0,14,1)
c=['red','green','blue','yellow','purple']
l=['Cluster 1', 'Cluster 2', 'Cluster 3','Cluster 4', 'Cluster 5']
for i in range(r):
    vals = np.diagonal((clf.covariances_[i,:,:]))
    plt.scatter(xplot[i,:],np.transpose(vals),color = c[i],label=l[i])
    
plt.xticks(np.arange(0,14,1))
plt.xlabel('Feature Number')
plt.ylabel('Variance')
plt.title('Cluster Variance Values for Each Feature for Leaf Data Set')

plt.legend()
plt.show()


# # EM Validation with AIC

# In[16]:


lowest_aic = np.infty

aic = []
n_components_range = range(1, 20)
cv_types = ['full']

for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type,random_state=1)
        gmm.fit(trainX_S)
        aic.append(gmm.aic(trainX_S))
        if aic[-1] < lowest_aic:
            lowest_aic = aic[-1]
            best_gmm = gmm
            

aic = np.array(aic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the AIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 0.05)
    bars.append(plt.bar(xpos, aic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
#plt.ylim([aic.min() * 1.01 - .01 * aic.max(), aic.max()])
plt.title('AIC score per model')
xpos = np.mod(aic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(aic.argmin() / len(n_components_range))
plt.text(xpos, aic.min() * 0.97 + .03 * aic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
spl.legend([b[0] for b in bars], cv_types)


# # Plot the winner
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(X)
# for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X[Y_ == i, 2], X[Y_ == i, 6], .8, color=color)

#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[0][1], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)

# plt.xticks(())
# plt.yticks(())
# plt.title('Selected GMM: full model, 2 components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()


# # ICA

# In[216]:


n_comps_all = np.arange(1,15,1)
e_all = []
kurt_sp = np.zeros((comp_to_check,3))

for i in n_comps_all:
    icat = FastICA(n_components=i)
    t_ = icat.fit_transform(trainX_S)
    pm = icat.inverse_transform(t_)
    loss_all = np.abs((trainX_S - pm)**2).mean()
    e_all.append(loss_all)
    kurt_sp[i-2,:] = np.array([i,np.mean(np.abs(kurtosis(t_,axis=0))),loss_all])
    
# tmp = pd.Series(data = pca.explained_variance_ratio_ ,index = range(1,11+1))
# tmp2 = pd.Series(data = pca.explained_variance_ ,index = range(1,11+1))
# print(tmp2)
fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(1,15))

mat = np.cumsum(tmp2)/np.sum(tmp2)
plt.scatter(n_comps_all,e_all)
plt.xlabel('Number of components')
plt.ylabel('MSE')
plt.title('Reconstruction Error')
plt.show()


fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(1,15))
plt.scatter(kurt_sp[:,0],kurt_sp[:,1])
plt.xlabel('Number of components')
plt.ylabel('Average Absolute Kurtosis')
plt.show()


unique_Y = (list(set(Y_first)))

icat = FastICA(n_components=10)
t_ = icat.fit_transform(trainX_S)

sse_ica = {}
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k,random_state=1).fit(t_)
    #print(data["clusters"])
    sse_ica[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse_ica.keys()), list(sse_ica.values()),'o')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.title('Elbow Method for Leaf Data Set')

plt.xticks(list(sse_ica.keys()))
plt.legend()
plt.show()

sil_all = []

for n_cluster in range(2, 20):
    kmeans = KMeans(n_clusters=n_cluster,random_state=1).fit(t_)
    label = kmeans.labels_
    print(label)
    sil_coeff = silhouette_score(t_, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    sil_all.append(sil_coeff)
 
plt.scatter(np.arange(2,20,1),sil_all,)
plt.xticks(np.arange(2,20,1))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Score for Leaf Data Set')
plt.legend()
plt.show()  



# # Plot the winner
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(trainX)

# for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X[Y_ == i, 5], X[Y_ == i, 6], .8, color=color)

#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[0][1], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)

# plt.xticks(())
# plt.yticks(())
# plt.title('Selected GMM: full model, 2 components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()


# In[215]:


lowest_bic = np.infty

bic = []
n_components_range = range(1, 20)
cv_types = ['full']
#cv_types = ['spherical', 'tied', 'diag','full']

for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components,random_state=4)
    gmm.fit(t_)
    print(gmm.predict(t_))
    bic.append(gmm.bic(t_))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 0.05)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))

plt.xticks(n_components_range)
#plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model for Leaf Data Set')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos+0.5, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of Clusters')
#spl.legend([b[0] for b in bars], cv_types)

Y_ = clf.predict(t_)
plt.show()

sil_all = []

for n_cluster in range(2, 20):
    clf1 = mixture.GaussianMixture(n_components=n_cluster,random_state=4).fit(t_)
    label = clf1.predict(t_)
    sil_coeff = silhouette_score(t_, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    sil_all.append(sil_coeff)
 
plt.scatter(np.arange(2,20,1),sil_all,)
plt.xticks(np.arange(2,20,1))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Score for Leaf Data Set')


# In[17]:


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
    def on_epoch_end(self, epoch, logs):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        logs['test_acc'] = acc
# Compute ICA

# ica = FastICA(n_components=11)
# S_ = ica.fit_transform(trainX_scaled)  # Reconstruct signals
# A_ = ica.mixing_  # Get estimated mixing matrix

# print(kurtosis(S_,axis=0))

ica = FastICA(random_state=1)
comp_to_check = 14
kurt_sp = np.zeros((comp_to_check,3))

train_all = []
test_all = []
train_acc_ica = []
val_acc_ica = []
test_acc_ica = []


for i in range(1,comp_to_check+1):
    print(i)
    ica.set_params(n_components=i)
    S_ = ica.fit_transform(trainX_S)
    proj = ica.inverse_transform(S_)
    loss_all = ((trainX_S - proj)**2).mean()
    kurt_sp[i-2,:] = np.array([i,np.mean(np.abs(kurtosis(S_,axis=0))),loss_all])
    
    trainX_transformed = ica.transform(trainX_S)
    testX_transformed = ica.transform(testX_S)
    
    trainX_S_use = trainX_transformed
    testX_S_use = testX_transformed
    
    trainX_use = trainX_S_use[values,:]
    trainY_use = trainY[values,:]

    

    model = Sequential()
    model.add(Dense(250, input_dim=i, kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint
    filepath="weights.best_leaf.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((testX_S_use,testY)),CSVLogger('1.log')]
    hist = model.fit(trainX_use[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)
    
    model1 = Sequential()
    model1.add(Dense(250, input_dim=i, kernel_initializer='normal', activation='relu'))
    model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.load_weights("weights.best_leaf.h5")


    train_pred = convert_out(model1.predict(trainX_use[:-82,:]))
    train_act = trainY_comp_use[:-82]
    train_acc_ica.append((1-Pred_correct(train_pred,train_act)))

    val_pred = convert_out(model1.predict(trainX_use[-82:,:]))
    val_act = trainY_comp_use[-82:]
    val_acc_ica.append((1-Pred_correct(val_pred,val_act)))

    test_pred = convert_out(model1.predict(testX_S_use))
    test_acc_ica.append((1-Pred_correct(test_pred,testY_comp)))
 


# # ICA Plots

# In[160]:


plt.scatter(kurt_sp[:,0],kurt_sp[:,1])
plt.xlabel('Number of components')
plt.ylabel('Average Absolute Kurtosis')
plt.show()
print(kurtosis(S_,axis=0))
plt.scatter(kurt_sp[:,0],kurt_sp[:,2])
plt.xlabel('Number of components')
plt.ylabel('Reconstruction Error')
plt.show()

fig = plt.figure()
ax = fig.gca()
ax.set_yticks(np.arange(0, 100, 10))
ax.set_xticks(np.arange(1,len(train_acc_ica)+1))

plt.scatter(np.arange(1,len(train_acc_ica)+1),np.array(train_acc_ica)*100,label='Training Set Accuracy')
plt.scatter(np.arange(1,len(train_acc_ica)+1),np.array(val_acc_ica)*100,label='Validation Set Accuracy')
plt.scatter(np.arange(1,len(train_acc_ica)+1),np.array(test_acc_ica)*100,label='Testing Set Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Misclassification Error (%)')
plt.legend()
plt.grid(True)
plt.show()

results = pd.DataFrame(data = np.transpose([np.array(train_acc_ica)*100,np.array(val_acc_ica)*100,np.array(test_acc_ica)*100]),columns=['Training Set', 'Validation Set', 'Testing Set'])
print(results)

best_c = np.where(val_acc_ica==np.min(val_acc_ica))[0]+1
print(best_c[0])


# # K Means + ICA

# In[19]:


ica.set_params(n_components=best_c[0])
S_ = ica.fit_transform(trainX_S)
proj = ica.inverse_transform(S_)
loss_all = ((trainX_S - proj)**2).mean()
kurt_sp[i-2,:] = np.array([i,np.mean(np.abs(kurtosis(S_,axis=0))),loss_all])

trainX_transformed = ica.transform(trainX_S)
testX_transformed = ica.transform(testX_S)

trainX_S_use = trainX_transformed
testX_S_use = testX_transformed

trainX_use = trainX_S_use[values,:]
trainY_use = trainY[values,:]

sse = {}
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k,random_state=1).fit(trainX_use)
    #print(data["clusters"])
    label = kmeans.labels_
    sil_coeff = silhouette_score(trainX_use, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff))
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()),'o')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.xticks(list(sse.keys()))
plt.legend()
plt.show()
    


# # K Means + ICA + NN

# In[20]:


l_use = np.arange(2,10)

train_acc_ica_f = []
val_acc_ica_f = []
test_acc_ica_f = []

for i in range(len(l_use)):
    print(i)
    km = KMeans(n_clusters=l_use[i],random_state=2)
    km.fit(trainX_use)
    # print(km.labels_)
    # plt.hist(km.labels_,l_use)
    # plt.show()
    lab_tr = km.predict(trainX_use)
    lab_te = km.predict(testX_S_use)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(lab_tr.reshape(-1, 1))
    lab_train=enc.transform(lab_tr.reshape(-1, 1)).toarray()
    lab_test = enc.transform(lab_te.reshape(-1, 1)).toarray()

    r,c = np.shape(trainX_use)
    trainX_use_n = np.hstack((trainX_use,lab_train))
    testX_use_n = np.hstack((testX_S_use,lab_test))

    model = Sequential()
    model.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint

    filepath="weights.best_leaf.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((testX_use_n,testY)),CSVLogger('1.log')]
    hist = model.fit(trainX_use_n[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use_n[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)

    model1 = Sequential()
    model1.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.load_weights("weights.best_leaf.h5")


    train_pred = convert_out(model1.predict(trainX_use_n[:-82,:]))
    train_act = trainY_comp_use[:-82]
    train_acc_ica_f.append((1-Pred_correct(train_pred,train_act)))

    val_pred = convert_out(model1.predict(trainX_use_n[-82:,:]))
    val_act = trainY_comp_use[-82:]
    val_acc_ica_f.append((1-Pred_correct(val_pred,val_act)))

    test_pred = convert_out(model1.predict(testX_use_n))
    test_acc_ica_f.append((1-Pred_correct(test_pred,testY_comp)))


# # K Means + ICA + NN Plots

# In[127]:


fig = plt.figure()
ax = fig.gca()
ax.set_yticks(np.arange(0, 100, 10))
ax.set_xticks(np.arange(2,len(train_acc_ica_f)+2))

plt.scatter(np.arange(2,len(train_acc_ica_f)+2),np.array(train_acc_ica_f)*100,label='Training Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_ica_f)+2),np.array(val_acc_ica_f)*100,label='Validation Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_ica_f)+2),np.array(test_acc_ica_f)*100,label='Testing Set Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Misclassification Error (%)')
plt.ylim([0,50])
plt.legend()
plt.grid(True)

results_f = pd.DataFrame(data = np.transpose([np.array(train_acc_ica_f)*100,np.array(val_acc_ica_f)*100,np.array(test_acc_ica_f)*100]),columns=['Training Set', 'Validation Set', 'Testing Set'])
print(results_f)

kmt = KMeans(n_clusters=2,random_state=2)
kmt.fit(trainX_use)
print(kmt.labels_)


# # EM + ICA 

# In[22]:


lowest_bic = np.infty

bic = []
n_components_range = range(1, 20)
cv_types = ['full']
#cv_types = ['spherical', 'tied', 'diag','full']

for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components,random_state=4)
    gmm.fit(trainX_use)
    bic.append(gmm.bic(trainX_use))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 0.05)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))

plt.xticks(n_components_range)
#plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos+0.5, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
#spl.legend([b[0] for b in bars], cv_types)

# # Plot the winner
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(trainX)

# for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X[Y_ == i, 5], X[Y_ == i, 6], .8, color=color)

#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[0][1], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)

# plt.xticks(())
# plt.yticks(())
# plt.title('Selected GMM: full model, 2 components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()


# # EM + ICA + NN 

# In[23]:


l_use = np.arange(2,10)

train_acc_ica_f_2 = []
val_acc_ica_f_2 = []
test_acc_ica_f_2 = []

for i in range(len(l_use)):
    print(i)
    gmm = GaussianMixture(n_components=l_use[i],random_state=1)
    gmm.fit(trainX_use)
    tr_lab = gmm.predict(trainX_use)
    test_lab = gmm.predict(testX_S_use)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(tr_lab.reshape(-1, 1))
    lab_train=enc.transform(tr_lab.reshape(-1, 1)).toarray()
    lab_test = enc.transform(test_lab.reshape(-1, 1)).toarray()

    r,c = np.shape(trainX_use)
    trainX_use_n = np.hstack((trainX_use,lab_train))
    testX_use_n = np.hstack((testX_S_use,lab_test))

    model = Sequential()
    model.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint

    filepath="weights.best_leaf.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((testX_use_n,testY)),CSVLogger('1.log')]
    hist = model.fit(trainX_use_n[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use_n[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)

    model1 = Sequential()
    model1.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.load_weights("weights.best_leaf.h5")


    train_pred = convert_out(model1.predict(trainX_use_n[:-82,:]))
    train_act = trainY_comp_use[:-82]
    train_acc_ica_f_2.append((1-Pred_correct(train_pred,train_act)))

    val_pred = convert_out(model1.predict(trainX_use_n[-82:,:]))
    val_act = trainY_comp_use[-82:]
    val_acc_ica_f_2.append((1-Pred_correct(val_pred,val_act)))

    test_pred = convert_out(model1.predict(testX_use_n))
    test_acc_ica_f_2.append((1-Pred_correct(test_pred,testY_comp)))


# # EM + ICA + NN Plots

# In[24]:


plt.scatter(np.arange(2,len(train_acc_ica_f_2)+2),np.array(train_acc_ica_f_2)*100,label='Training Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_ica_f_2)+2),np.array(val_acc_ica_f_2)*100,label='Validation Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_ica_f_2)+2),np.array(test_acc_ica_f_2)*100,label='Testing Set Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Misclassification Error (%)')
plt.ylim([0,50])
plt.legend()
plt.grid(True)

results_f = pd.DataFrame(data = np.transpose([np.array(train_acc_ica_f_2)*100,np.array(val_acc_ica_f_2)*100,np.array(test_acc_ica_f_2)*100]),columns=['Training Set', 'Validation Set', 'Testing Set'])
print(results_f)


# # PCA

# In[224]:


n_comps_all = np.arange(1,15,1)
e_all = []
for i in n_comps_all:
    pcat = PCA(n_components=i,random_state=1)
    t_ = pcat.fit_transform(trainX_S)
    pm = pcat.inverse_transform(t_)
    loss_all = ((trainX_S - pm)**2).mean()
    e_all.append(loss_all)
    
tmp = pd.Series(data = pca.explained_variance_ratio_ ,index = range(1,14+1))
tmp2 = pd.Series(data = pca.explained_variance_ ,index = range(1,14+1))
print(tmp2)

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(1,14))

mat = np.cumsum(tmp2)/np.sum(tmp2)
plt.scatter(n_comps_all,e_all)
plt.xlabel('Number of components')
plt.ylabel('MSE')
plt.title('Reconstruction Error')
plt.show()

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(1,14))

plt.scatter(n_comps_all,tmp2)
plt.xlabel('Number of components')
plt.ylabel('Eigenvalue')
plt.title('Eigvenvalue of different components')
plt.show()


pcat = PCA(n_components=5,random_state=1)
t_ = pcat.fit_transform(trainX_S)

sse_pca = {}
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k,random_state=1).fit(t_)
    #print(data["clusters"])
    sse_pca[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse_pca.keys()), list(sse_pca.values()),'o')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.title('Elbow Method for Leaf Data Set')

plt.xticks(list(sse_pca.keys()))
plt.legend()
plt.show()

sil_all = []

for n_cluster in range(2, 20):
    kmeans = KMeans(n_clusters=n_cluster,random_state=1).fit(t_)
    label = kmeans.labels_
    print(label)
    sil_coeff = silhouette_score(t_, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    sil_all.append(sil_coeff)
 
plt.scatter(np.arange(2,20,1),sil_all,)
plt.xticks(np.arange(2,20,1))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Score for Leaf Data Set')
plt.legend()
plt.show()  


# In[225]:


lowest_bic = np.infty

bic = []
n_components_range = range(1, 20)
cv_types = ['full']
#cv_types = ['spherical', 'tied', 'diag','full']

for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components,random_state=4)
    gmm.fit(t_)
    print(gmm.predict(t_))
    bic.append(gmm.bic(t_))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 0.05)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))

plt.xticks(n_components_range)
#plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model for Leaf Data Set')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos+0.5, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of Clusters')
#spl.legend([b[0] for b in bars], cv_types)

Y_ = clf.predict(t_)
plt.show()

sil_all = []

for n_cluster in range(2, 20):
    clf1 = mixture.GaussianMixture(n_components=n_cluster,random_state=4).fit(t_)
    label = clf1.predict(t_)
    sil_coeff = silhouette_score(t_, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    sil_all.append(sil_coeff)
 
plt.scatter(np.arange(2,20,1),sil_all,)
plt.xticks(np.arange(2,20,1))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Score for Leaf Data Set')


# In[25]:


# Compute ICA

# ica = FastICA(n_components=11)
# S_ = ica.fit_transform(trainX_scaled)  # Reconstruct signals
# A_ = ica.mixing_  # Get estimated mixing matrix

# print(kurtosis(S_,axis=0))
n_comp = 14
pca = PCA(n_components=n_comp,random_state=1)
trainX_transformed = pca.fit_transform(trainX_S)
testX_transformed = pca.transform(testX_S)

tmp = pd.Series(data = pca.explained_variance_ratio_ ,index = range(1,n_comp+1))
tmp2 = pd.Series(data = pca.explained_variance_ ,index = range(1,n_comp+1))


mat = np.cumsum(tmp2)/np.sum(tmp2)
print(mat)
plt.plot(np.cumsum(tmp2)/np.sum(tmp2))
plt.xlabel('Number of components')
plt.ylabel('Reconstruction Error (%)')
plt.show()


# # PCA + NN

# In[26]:


train_all = []
test_all = []
train_acc = []
val_acc = []
test_acc = []

for i in range(1,15):
    print(i)
    trainX_S_use = trainX_transformed[:,:i]
    testX_S_use = testX_transformed[:,:i]
    
    trainX_use = trainX_S_use[values,:]
    trainY_use = trainY[values,:]

    

    model = Sequential()
    model.add(Dense(250, input_dim=i, kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint
    filepath="weights.best_leaf.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((testX_S_use,testY)),CSVLogger('1.log')]
    hist = model.fit(trainX_use[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)
    
    model1 = Sequential()
    model1.add(Dense(250, input_dim=i, kernel_initializer='normal', activation='relu'))
    model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.load_weights("weights.best_leaf.h5")


    train_pred = convert_out(model1.predict(trainX_use[:-82,:]))
    train_act = trainY_comp_use[:-82]
    train_acc.append((1-Pred_correct(train_pred,train_act)))

    val_pred = convert_out(model1.predict(trainX_use[-82:,:]))
    val_act = trainY_comp_use[-82:]
    val_acc.append((1-Pred_correct(val_pred,val_act)))

    test_pred = convert_out(model1.predict(testX_S_use))
    test_acc.append((1-Pred_correct(test_pred,testY_comp)))


# # PCA + NN Plot

# In[27]:


fig = plt.figure()
ax = fig.gca()
ax.set_yticks(np.arange(0, 100, 10))
ax.set_xticks(np.arange(1,len(train_acc)+1))

plt.scatter(np.arange(1,len(train_acc)+1),np.array(train_acc)*100,label='Training Set Accuracy')
plt.scatter(np.arange(1,len(train_acc)+1),np.array(val_acc)*100,label='Validation Set Accuracy')
plt.scatter(np.arange(1,len(train_acc)+1),np.array(test_acc)*100,label='Testing Set Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Misclassification Error (%)')
plt.legend()
plt.grid(True)
plt.show()

results = pd.DataFrame(data = np.transpose([np.array(train_acc)*100,np.array(val_acc)*100,np.array(test_acc)*100]))
print(results)

best_p = np.where(val_acc==np.min(val_acc))[0]+1
print(best_p[0])


# # K Means + PCA

# In[28]:


trainX_S_use = trainX_transformed[:,:best_p[0]]
testX_S_use = testX_transformed[:,:best_p[0]]

trainX_use = trainX_S_use[values,:]
trainY_use = trainY[values,:]

sse = {}
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k,random_state=1).fit(trainX_use)
    #print(data["clusters"])
    label = kmeans.labels_
    sil_coeff = silhouette_score(trainX_use, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff))
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()),'o')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.xticks(list(sse.keys()))
plt.legend()
plt.show()
    


# # K Means + PCA + NN

# In[29]:


l_use = np.arange(2,10)

train_acc_ica_p = []
val_acc_ica_p = []
test_acc_ica_p = []

for i in range(len(l_use)):
    print(i)
    km = KMeans(n_clusters=l_use[i],random_state=2)
    km.fit(trainX_use)
    # print(km.labels_)
    # plt.hist(km.labels_,l_use)
    # plt.show()
    lab_tr = km.predict(trainX_use)
    lab_te = km.predict(testX_S_use)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(lab_tr.reshape(-1, 1))
    lab_train=enc.transform(lab_tr.reshape(-1, 1)).toarray()
    lab_test = enc.transform(lab_te.reshape(-1, 1)).toarray()

    r,c = np.shape(trainX_use)
    trainX_use_n = np.hstack((trainX_use,lab_train))
    testX_use_n = np.hstack((testX_S_use,lab_test))

    model = Sequential()
    model.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint

    filepath="weights.best_leaf.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((testX_use_n,testY)),CSVLogger('1.log')]
    hist = model.fit(trainX_use_n[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use_n[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)

    model1 = Sequential()
    model1.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.load_weights("weights.best_leaf.h5")


    train_pred = convert_out(model1.predict(trainX_use_n[:-82,:]))
    train_act = trainY_comp_use[:-82]
    train_acc_ica_p.append((1-Pred_correct(train_pred,train_act)))

    val_pred = convert_out(model1.predict(trainX_use_n[-82:,:]))
    val_act = trainY_comp_use[-82:]
    val_acc_ica_p.append((1-Pred_correct(val_pred,val_act)))

    test_pred = convert_out(model1.predict(testX_use_n))
    test_acc_ica_p.append((1-Pred_correct(test_pred,testY_comp)))


# # K Means + PCA + NN Plot

# In[119]:


fig = plt.figure()
ax = fig.gca()
ax.set_yticks(np.arange(0, 100, 10))
ax.set_xticks(np.arange(2,len(train_acc_ica_p)+2))

plt.scatter(np.arange(2,len(train_acc_ica_p)+2),np.array(train_acc_ica_p)*100,label='Training Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_ica_p)+2),np.array(val_acc_ica_p)*100,label='Validation Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_ica_p)+2),np.array(test_acc_ica_p)*100,label='Testing Set Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Misclassification Error (%)')
plt.ylim([0,50])
plt.legend()
plt.grid(True)

results_f = pd.DataFrame(data = np.transpose([np.array(train_acc_ica_p)*100,np.array(val_acc_ica_p)*100,np.array(test_acc_ica_p)*100]),columns=['Training Set', 'Validation Set', 'Testing Set'])
print(results_f)


# # EM + PCA

# In[31]:


lowest_bic = np.infty
print(np.shape(trainX_use))
bic = []
n_components_range = range(1, 20)
cv_types = ['full']
#cv_types = ['spherical', 'tied', 'diag','full']

for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components,random_state=4)
    gmm.fit(trainX_use)
    bic.append(gmm.bic(trainX_use))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 0.05)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))

plt.xticks(n_components_range)
#plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos+0.5, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
#spl.legend([b[0] for b in bars], cv_types)

# # Plot the winner
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(trainX)

# for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X[Y_ == i, 5], X[Y_ == i, 6], .8, color=color)

#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[0][1], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)

# plt.xticks(())
# plt.yticks(())
# plt.title('Selected GMM: full model, 2 components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()


# # EM + PCA + NN

# In[32]:


l_use = np.arange(2,10)

train_acc_ica_p_2 = []
val_acc_ica_p_2 = []
test_acc_ica_p_2 = []

for i in range(len(l_use)):
    print(i)
    gmm = GaussianMixture(n_components=l_use[i],random_state=1)
    gmm.fit(trainX_use)
    tr_lab = gmm.predict(trainX_use)
    test_lab = gmm.predict(testX_S_use)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(tr_lab.reshape(-1, 1))
    lab_train=enc.transform(tr_lab.reshape(-1, 1)).toarray()
    lab_test = enc.transform(test_lab.reshape(-1, 1)).toarray()

    r,c = np.shape(trainX_use)
    trainX_use_n = np.hstack((trainX_use,lab_train))
    testX_use_n = np.hstack((testX_S_use,lab_test))

    model = Sequential()
    model.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint

    filepath="weights.best_leaf.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((testX_use_n,testY)),CSVLogger('1.log')]
    hist = model.fit(trainX_use_n[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use_n[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)

    model1 = Sequential()
    model1.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.load_weights("weights.best_leaf.h5")


    train_pred = convert_out(model1.predict(trainX_use_n[:-82,:]))
    train_act = trainY_comp_use[:-82]
    train_acc_ica_p_2.append((1-Pred_correct(train_pred,train_act)))

    val_pred = convert_out(model1.predict(trainX_use_n[-82:,:]))
    val_act = trainY_comp_use[-82:]
    val_acc_ica_p_2.append((1-Pred_correct(val_pred,val_act)))

    test_pred = convert_out(model1.predict(testX_use_n))
    test_acc_ica_p_2.append((1-Pred_correct(test_pred,testY_comp)))


# # EM + PCA + NN Plot

# In[33]:


plt.scatter(np.arange(2,len(train_acc_ica_p_2)+2),np.array(train_acc_ica_p_2)*100,label='Training Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_ica_p_2)+2),np.array(val_acc_ica_p_2)*100,label='Validation Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_ica_p_2)+2),np.array(test_acc_ica_p_2)*100,label='Testing Set Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Misclassification Error (%)')
plt.ylim([0,50])
plt.legend()
plt.grid(True)

results_p = pd.DataFrame(data = np.transpose([np.array(train_acc_ica_p_2)*100,np.array(val_acc_ica_p_2)*100,np.array(test_acc_ica_p_2)*100]),columns=['Training Set', 'Validation Set', 'Testing Set'])
print(results_p)


# # Random Projections + NN

# In[228]:


n_comps_all = np.arange(1,12,1)
e_all = []
for i in n_comps_all:
    pcat = GaussianRandomProjection(n_components=i,random_state=1)
    t_ = pcat.fit_transform(trainX_S)
    randMat = pcat.components_
    pm = t_.dot(randMat)
    loss_all = ((trainX_S - pm)**2).mean()
    e_all.append(loss_all)






# tmp = pd.Series(data = pcat.explained_variance_ratio_ ,index = range(1,12))
# tmp2 = pd.Series(data = pcat.explained_variance_ ,index = range(1,12))
print( pcat)

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(1,12))

mat = np.cumsum(tmp2)/np.sum(tmp2)
plt.scatter(n_comps_all,e_all)
plt.xlabel('Number of components')
plt.ylabel('MSE')
plt.title('Reconstruction Error')
plt.show()



pcat = GaussianRandomProjection(n_components=4,random_state=1)
t_ = pcat.fit_transform(trainX_S)

sse_pca = {}
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k,random_state=1).fit(t_)
    #print(data["clusters"])
    sse_pca[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse_pca.keys()), list(sse_pca.values()),'o')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.title('Elbow Method for Leaf Data Set')

plt.xticks(list(sse_pca.keys()))
plt.legend()
plt.show()

sil_all = []

for n_cluster in range(2, 20):
    kmeans = KMeans(n_clusters=n_cluster,random_state=1).fit(t_)
    label = kmeans.labels_
    print(label)
    sil_coeff = silhouette_score(t_, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    sil_all.append(sil_coeff)
 
plt.scatter(np.arange(2,20,1),sil_all,)
plt.xticks(np.arange(2,20,1))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Score for Leaf Data Set')
plt.legend()
plt.show()  


# In[229]:


lowest_bic = np.infty

bic = []
n_components_range = range(1, 20)
cv_types = ['full']
#cv_types = ['spherical', 'tied', 'diag','full']

for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components,random_state=4)
    gmm.fit(t_)
    print(gmm.predict(t_))
    bic.append(gmm.bic(t_))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 0.05)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))

plt.xticks(n_components_range)
#plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model for Leaf Data Set')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos+0.5, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of Clusters')
#spl.legend([b[0] for b in bars], cv_types)

Y_ = clf.predict(t_)
plt.show()

sil_all = []

for n_cluster in range(2, 20):
    clf1 = mixture.GaussianMixture(n_components=n_cluster,random_state=4).fit(t_)
    label = clf1.predict(t_)
    sil_coeff = silhouette_score(t_, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    sil_all.append(sil_coeff)
 
plt.scatter(np.arange(2,20,1),sil_all,)
plt.xticks(np.arange(2,20,1))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Score for Leaf Data Set')


# In[34]:


class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data
    def on_epoch_end(self, epoch, logs):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        logs['test_acc'] = acc

comp_to_check = 14
rp_sp = np.zeros((comp_to_check-2,2))

train_all = []
test_all = []
train_acc_rp = []
val_acc_rp = []
test_acc_rp = []


for i in range(1,comp_to_check+1):
    print(i)
    proj = GaussianRandomProjection(n_components=i,random_state=1)
    rp1 = proj.fit(trainX_S)
    rp_ = proj.transform(trainX_S)
    
    trainX_transformed = proj.transform(trainX_S)
    testX_transformed = proj.transform(testX_S)
    
    randMat = proj.components_
    X_Proj = rp_.dot(randMat)
#     rp_sp[i-2,:] = [i,((X_Proj - trainX_scaled)**2).mean()]

    trainX_S_use = trainX_transformed
    testX_S_use = testX_transformed
    print(np.shape(trainX_S_use))
    
    trainX_use = trainX_S_use[values,:]
    trainY_use = trainY[values,:]

    

    model = Sequential()
    model.add(Dense(250, input_dim=i, kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint
    filepath="weights.best_leaf.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((testX_S_use,testY)),CSVLogger('1.log')]
    hist = model.fit(trainX_use[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)
    
    model1 = Sequential()
    model1.add(Dense(250, input_dim=i, kernel_initializer='normal', activation='relu'))
    model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.load_weights("weights.best_leaf.h5")


    train_pred = convert_out(model1.predict(trainX_use[:-82,:]))
    train_act = trainY_comp_use[:-82]
    train_acc_rp.append((1-Pred_correct(train_pred,train_act)))

    val_pred = convert_out(model1.predict(trainX_use[-82:,:]))
    val_act = trainY_comp_use[-82:]
    val_acc_rp.append((1-Pred_correct(val_pred,val_act)))

    test_pred = convert_out(model1.predict(testX_S_use))
    test_acc_rp.append((1-Pred_correct(test_pred,testY_comp)))


# # Random Projections + NN Plot

# In[35]:


fig = plt.figure()
ax = fig.gca()
ax.set_yticks(np.arange(0, 100, 10))
ax.set_xticks(np.arange(1,len(train_acc)+1))

plt.scatter(np.arange(1,len(train_acc_rp)+1),np.array(train_acc_rp)*100,label='Training Set Accuracy')
plt.scatter(np.arange(1,len(train_acc_rp)+1),np.array(val_acc_rp)*100,label='Validation Set Accuracy')
plt.scatter(np.arange(1,len(train_acc_rp)+1),np.array(test_acc_rp)*100,label='Testing Set Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Misclassification Error (%)')
plt.legend()
plt.grid(True)
plt.show()

results_rp = pd.DataFrame(data = np.transpose([np.array(train_acc_rp)*100,np.array(val_acc_rp)*100,np.array(test_acc_rp)*100]))
print(results_rp)

best_rp = np.where(val_acc_rp==np.min(val_acc_rp))[0]+1
print(best_rp[0])

# plt.scatter(rp_sp[:,0],rp_sp[:,1])
# plt.xlabel('Number of Components')
# plt.ylabel('Reconstruction MSE')
# plt.show()


# # K Means + RP

# In[36]:


proj = GaussianRandomProjection(n_components=best_rp[0],random_state=1)
rp1 = proj.fit(trainX_S)
rp_ = proj.transform(trainX_S)

trainX_transformed = proj.transform(trainX_S)
testX_transformed = proj.transform(testX_S)

randMat = proj.components_
X_Proj = rp_.dot(randMat)
#     rp_sp[i-2,:] = [i,((X_Proj - trainX_scaled)**2).mean()]

trainX_S_use = trainX_transformed
testX_S_use = testX_transformed

trainX_use = trainX_S_use[values,:]
trainY_use = trainY[values,:]

sse = {}
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k,random_state=1).fit(trainX_use)
    #print(data["clusters"])
    label = kmeans.labels_
    sil_coeff = silhouette_score(trainX_use, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff))
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()),'o')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.xticks(list(sse.keys()))
plt.legend()
plt.show()
    


# # K Means + RP + NN

# In[37]:


l_use = np.arange(2,10)

train_acc_rp_f = []
val_acc_rp_f = []
test_acc_rp_f = []

for i in range(len(l_use)):
    print(i)
    km = KMeans(n_clusters=l_use[i],random_state=2)
    km.fit(trainX_use)
    # print(km.labels_)
    # plt.hist(km.labels_,l_use)
    # plt.show()
    lab_tr = km.predict(trainX_use)
    lab_te = km.predict(testX_S_use)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(lab_tr.reshape(-1, 1))
    lab_train=enc.transform(lab_tr.reshape(-1, 1)).toarray()
    lab_test = enc.transform(lab_te.reshape(-1, 1)).toarray()

    r,c = np.shape(trainX_use)
    trainX_use_n = np.hstack((trainX_use,lab_train))
    testX_use_n = np.hstack((testX_S_use,lab_test))

    model = Sequential()
    model.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint

    filepath="weights.best_leaf.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((testX_use_n,testY)),CSVLogger('1.log')]
    hist = model.fit(trainX_use_n[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use_n[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)

    model1 = Sequential()
    model1.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.load_weights("weights.best_leaf.h5")


    train_pred = convert_out(model1.predict(trainX_use_n[:-82,:]))
    train_act = trainY_comp_use[:-82]
    train_acc_rp_f.append((1-Pred_correct(train_pred,train_act)))

    val_pred = convert_out(model1.predict(trainX_use_n[-82:,:]))
    val_act = trainY_comp_use[-82:]
    val_acc_rp_f.append((1-Pred_correct(val_pred,val_act)))

    test_pred = convert_out(model1.predict(testX_use_n))
    test_acc_rp_f.append((1-Pred_correct(test_pred,testY_comp)))


# # K Means + RP + NN Plot

# In[123]:


fig = plt.figure()
ax = fig.gca()
ax.set_yticks(np.arange(0, 100, 10))
ax.set_xticks(np.arange(2,len(train_acc_rp_f)+2))

plt.scatter(np.arange(2,len(train_acc_rp_f)+2),np.array(train_acc_rp_f)*100,label='Training Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_rp_f)+2),np.array(val_acc_rp_f)*100,label='Validation Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_rp_f)+2),np.array(test_acc_rp_f)*100,label='Testing Set Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Misclassification Error (%)')
plt.ylim([0,60])
plt.legend()
plt.grid(True)

results_f = pd.DataFrame(data = np.transpose([np.array(train_acc_rp_f)*100,np.array(val_acc_rp_f)*100,np.array(test_acc_rp_f)*100]),columns=['Training Set', 'Validation Set', 'Testing Set'])
print(results_f)


# # EM + RP

# In[39]:


lowest_bic = np.infty

bic = []
n_components_range = range(1, 20)
cv_types = ['full']
#cv_types = ['spherical', 'tied', 'diag','full']

for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components,random_state=4)
    gmm.fit(trainX_use)
    bic.append(gmm.bic(trainX_use))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 0.05)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))

plt.xticks(n_components_range)
#plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos+0.5, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
#spl.legend([b[0] for b in bars], cv_types)

# # Plot the winner
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(trainX)

# for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X[Y_ == i, 5], X[Y_ == i, 6], .8, color=color)

#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[0][1], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)

# plt.xticks(())
# plt.yticks(())
# plt.title('Selected GMM: full model, 2 components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()


# # EM + RP + NN

# In[40]:


l_use = np.arange(2,10)

train_acc_rp_2 = []
val_acc_rp_2 = []
test_acc_rp_2 = []

for i in range(len(l_use)):
    print(i)
    gmm = GaussianMixture(n_components=l_use[i],random_state=1)
    gmm.fit(trainX_use)
    tr_lab = gmm.predict(trainX_use)
    test_lab = gmm.predict(testX_S_use)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(tr_lab.reshape(-1, 1))
    lab_train=enc.transform(tr_lab.reshape(-1, 1)).toarray()
    lab_test = enc.transform(test_lab.reshape(-1, 1)).toarray()

    r,c = np.shape(trainX_use)
    trainX_use_n = np.hstack((trainX_use,lab_train))
    testX_use_n = np.hstack((testX_S_use,lab_test))

    model = Sequential()
    model.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint

    filepath="weights.best_leaf.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((testX_use_n,testY)),CSVLogger('1.log')]
    hist = model.fit(trainX_use_n[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use_n[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)

    model1 = Sequential()
    model1.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.load_weights("weights.best_leaf.h5")


    train_pred = convert_out(model1.predict(trainX_use_n[:-82,:]))
    train_act = trainY_comp_use[:-82]
    train_acc_rp_2.append((1-Pred_correct(train_pred,train_act)))

    val_pred = convert_out(model1.predict(trainX_use_n[-82:,:]))
    val_act = trainY_comp_use[-82:]
    val_acc_rp_2.append((1-Pred_correct(val_pred,val_act)))

    test_pred = convert_out(model1.predict(testX_use_n))
    test_acc_rp_2.append((1-Pred_correct(test_pred,testY_comp)))


# # EM + RP + NN Plot

# In[128]:


plt.scatter(np.arange(2,len(train_acc_rp_2)+2),np.array(train_acc_rp_2)*100,label='Training Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_rp_2)+2),np.array(val_acc_rp_2)*100,label='Validation Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_rp_2)+2),np.array(test_acc_rp_2)*100,label='Testing Set Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Misclassification Error (%)')
plt.ylim([0,50])
plt.legend()
plt.grid(True)

results_f = pd.DataFrame(data = np.transpose([np.array(train_acc_rp_2)*100,np.array(val_acc_rp_2)*100,np.array(test_acc_rp_2)*100]),columns=['Training Set', 'Validation Set', 'Testing Set'])
print(results_f)


# # LDA

# In[230]:


n_comps_all = np.arange(1,15,1)
e_all = []
clft  = LinearDiscriminantAnalysis()
clft.fit(trainX_S, trainY_comp)
t_ = clft.transform(trainX_S)
print(clft.explained_variance_ratio_)


# tmp = pd.Series(data = pcat.explained_variance_ratio_ ,index = range(1,12))
# tmp2 = pd.Series(data = pcat.explained_variance_ ,index = range(1,12))
print( pcat)

fig = plt.figure()
ax = fig.gca()
ax.set_xticks(np.arange(1,15))

plt.scatter(n_comps_all,clft.explained_variance_ratio_)
plt.xlabel('Number of components')
plt.ylabel('Variance Ratio')
plt.title('Variance Ratio for Leaf Data Set')
plt.show()



clft  = LinearDiscriminantAnalysis()
clft.fit(trainX_S, trainY_comp)
t_ = clft.transform(trainX_S)

sse_pca = {}
for k in range(1, 20):
    kmeans = KMeans(n_clusters=k,random_state=1).fit(t_[:,:6])
    #print(data["clusters"])
    sse_pca[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse_pca.keys()), list(sse_pca.values()),'o')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.title('Elbow Method for Leaf Data Set')

plt.xticks(list(sse_pca.keys()))
plt.legend()
plt.show()

sil_all = []

for n_cluster in range(2, 20):
    kmeans = KMeans(n_clusters=n_cluster,random_state=1).fit(t_[:,:6])
    label = kmeans.labels_
    print(label)
    sil_coeff = silhouette_score(t_, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    sil_all.append(sil_coeff)
 
plt.scatter(np.arange(2,20,1),sil_all,)
plt.xticks(np.arange(2,20,1))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Score for Leaf Data Set')
plt.legend()
plt.show()  


# In[241]:


lowest_bic = np.infty

bic = []
n_components_range = range(1, 20)
cv_types = ['full']
#cv_types = ['spherical', 'tied', 'diag','full']

for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components,random_state=4)
    gmm.fit(t_[:,:6])
    print(gmm.predict(t_[:,:6]))
    bic.append(gmm.bic(t_[:,:6]))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 0.05)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))

plt.xticks(n_components_range)
#plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model for Leaf Data Set')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos+0.5, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of Clusters')
#spl.legend([b[0] for b in bars], cv_types)

Y_ = clf.predict(t_[:,:6])
plt.show()

sil_all = []

for n_cluster in range(2, 20):
    clf1 = mixture.GaussianMixture(n_components=n_cluster,random_state=4).fit(t_[:,:6])
    label = clf1.predict(t_[:,:6])
    sil_coeff = silhouette_score(t_[:,:6], label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(n_cluster, sil_coeff))
    sil_all.append(sil_coeff)
 
plt.scatter(np.arange(2,20,1),sil_all,)
plt.xticks(np.arange(2,20,1))
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient')
plt.title('Silhouette Score for Leaf Data Set')


# In[44]:


clf  = LinearDiscriminantAnalysis()
clf.fit(trainX_S, trainY_comp)
trainX_transformed = clf.transform(trainX_S)
testX_transformed = clf.transform(testX_S)

tmp = pd.Series(data = clf.explained_variance_ratio_ ,index = range(1,14+1))


mat = np.cumsum(tmp)/np.sum(tmp)
print(mat)
plt.plot(np.cumsum(tmp)/np.sum(tmp))
plt.xlabel('Number of components')
plt.ylabel('Reconstruction Error (%)')
plt.show()


# # LDA + NN

# In[45]:


train_all = []
test_all = []
train_acc_lda = []
val_acc_lda = []
test_acc_lda = []

for i in range(1,15):
    print(i)
    trainX_S_use = trainX_transformed[:,:i]
    testX_S_use = testX_transformed[:,:i]
    
    trainX_use = trainX_S_use[values,:]
    trainY_use = trainY[values,:]

    

    model = Sequential()
    model.add(Dense(250, input_dim=i, kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint
    filepath="weights.best_leaf.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((testX_S_use,testY)),CSVLogger('1.log')]
    hist = model.fit(trainX_use[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)
    
    model1 = Sequential()
    model1.add(Dense(250, input_dim=i, kernel_initializer='normal', activation='relu'))
    model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.load_weights("weights.best_leaf.h5")


    train_pred = convert_out(model1.predict(trainX_use[:-82,:]))
    train_act = trainY_comp_use[:-82]
    train_acc_lda.append((1-Pred_correct(train_pred,train_act)))

    val_pred = convert_out(model1.predict(trainX_use[-82:,:]))
    val_act = trainY_comp_use[-82:]
    val_acc_lda.append((1-Pred_correct(val_pred,val_act)))

    test_pred = convert_out(model1.predict(testX_S_use))
    test_acc_lda.append((1-Pred_correct(test_pred,testY_comp)))


# # LDA + NN Plot

# In[46]:


fig = plt.figure()
ax = fig.gca()
ax.set_yticks(np.arange(0, 100, 10))
ax.set_xticks(np.arange(1,len(train_acc_lda)+1))

plt.scatter(np.arange(1,len(train_acc_lda)+1),np.array(train_acc_lda)*100,label='Training Set Accuracy')
plt.scatter(np.arange(1,len(train_acc_lda)+1),np.array(val_acc_lda)*100,label='Validation Set Accuracy')
plt.scatter(np.arange(1,len(train_acc_lda)+1),np.array(test_acc_lda)*100,label='Testing Set Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Misclassification Error (%)')
plt.legend()
plt.grid(True)
plt.show()

results = pd.DataFrame(data = np.transpose([np.array(train_acc_lda)*100,np.array(val_acc_lda)*100,np.array(test_acc_lda)*100]))
print(results)

best_ld = np.where(val_acc_lda==np.min(val_acc_lda))[0]+1
print(best_ld[0])


# # LDA + K Means

# In[47]:


trainX_S_use = trainX_transformed[:,:best_ld[0]]
testX_S_use = testX_transformed[:,:best_ld[0]]

trainX_use = trainX_S_use[values,:]
trainY_use = trainY[values,:]

sse = {}
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k,random_state=1).fit(trainX_use)
    #print(data["clusters"])
    label = kmeans.labels_
    sil_coeff = silhouette_score(trainX_use, label, metric='euclidean')
    print("For n_clusters={}, The Silhouette Coefficient is {}".format(k, sil_coeff))
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()),'o')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.xticks(list(sse.keys()))
plt.legend()
plt.show()
    


# # LDA + K Means + NN

# In[48]:


l_use = np.arange(2,10)

train_acc_lda_p = []
val_acc_lda_p = []
test_acc_lda_p = []

for i in range(len(l_use)):
    print(i)
    km = KMeans(n_clusters=l_use[i],random_state=2)
    km.fit(trainX_use)
    # print(km.labels_)
    # plt.hist(km.labels_,l_use)
    # plt.show()
    lab_tr = km.predict(trainX_use)
    lab_te = km.predict(testX_S_use)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(lab_tr.reshape(-1, 1))
    lab_train=enc.transform(lab_tr.reshape(-1, 1)).toarray()
    lab_test = enc.transform(lab_te.reshape(-1, 1)).toarray()

    r,c = np.shape(trainX_use)
    trainX_use_n = np.hstack((trainX_use,lab_train))
    testX_use_n = np.hstack((testX_S_use,lab_test))

    model = Sequential()
    model.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint

    filepath="weights.best_leaf.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((testX_use_n,testY)),CSVLogger('1.log')]
    hist = model.fit(trainX_use_n[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use_n[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)

    model1 = Sequential()
    model1.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.load_weights("weights.best_leaf.h5")


    train_pred = convert_out(model1.predict(trainX_use_n[:-82,:]))
    train_act = trainY_comp_use[:-82]
    train_acc_lda_p.append((1-Pred_correct(train_pred,train_act)))

    val_pred = convert_out(model1.predict(trainX_use_n[-82:,:]))
    val_act = trainY_comp_use[-82:]
    val_acc_lda_p.append((1-Pred_correct(val_pred,val_act)))

    test_pred = convert_out(model1.predict(testX_use_n))
    test_acc_lda_p.append((1-Pred_correct(test_pred,testY_comp)))


# # K Means + NN + LDA Plot

# In[124]:


fig = plt.figure()
ax = fig.gca()
ax.set_yticks(np.arange(0, 100, 10))
ax.set_xticks(np.arange(2,len(train_acc_lda_p)+2))

plt.scatter(np.arange(2,len(train_acc_lda_p)+2),np.array(train_acc_lda_p)*100,label='Training Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_lda_p)+2),np.array(val_acc_lda_p)*100,label='Validation Set Accuracy')
plt.scatter(np.arange(2,len(test_acc_lda_p)+2),np.array(test_acc_lda_p)*100,label='Testing Set Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Misclassification Error (%)')
plt.ylim([0,50])
plt.legend()
plt.grid(True)

results_f = pd.DataFrame(data = np.transpose([np.array(train_acc_lda_p)*100,np.array(val_acc_lda_p)*100,np.array(test_acc_lda_p)*100]),columns=['Training Set', 'Validation Set', 'Testing Set'])
print(results_f)


# # EM + LDA
# 

# In[50]:


lowest_bic = np.infty
print(np.shape(trainX_use))
bic = []
n_components_range = range(1, 20)
cv_types = ['full']
#cv_types = ['spherical', 'tied', 'diag','full']

for n_components in n_components_range:
    # Fit a Gaussian mixture with EM
    gmm = mixture.GaussianMixture(n_components=n_components,random_state=4)
    gmm.fit(trainX_use)
    bic.append(gmm.bic(trainX_use))
    if bic[-1] < lowest_bic:
        lowest_bic = bic[-1]
        best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(figsize=(8, 6))
spl = plt.subplot(2, 1, 1)
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 0.05)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))

plt.xticks(n_components_range)
#plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos+0.5, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
spl.set_xlabel('Number of components')
#spl.legend([b[0] for b in bars], cv_types)

# # Plot the winner
# splot = plt.subplot(2, 1, 2)
# Y_ = clf.predict(trainX)

# for i, (mean, cov, color) in enumerate(zip(clf.means_, clf.covariances_,
#                                            color_iter)):
#     v, w = linalg.eigh(cov)
#     if not np.any(Y_ == i):
#         continue
#     plt.scatter(X[Y_ == i, 5], X[Y_ == i, 6], .8, color=color)

#     # Plot an ellipse to show the Gaussian component
#     angle = np.arctan2(w[0][1], w[0][0])
#     angle = 180. * angle / np.pi  # convert to degrees
#     v = 2. * np.sqrt(2.) * np.sqrt(v)
#     ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
#     ell.set_clip_box(splot.bbox)
#     ell.set_alpha(.5)
#     splot.add_artist(ell)

# plt.xticks(())
# plt.yticks(())
# plt.title('Selected GMM: full model, 2 components')
# plt.subplots_adjust(hspace=.35, bottom=.02)
# plt.show()


# # EM + NN + LDA
# 

# In[51]:


l_use = np.arange(2,10)

train_acc_lda_p_2 = []
val_acc_lda_p_2 = []
test_acc_lda_p_2 = []

for i in range(len(l_use)):
    print(i)
    gmm = GaussianMixture(n_components=l_use[i],random_state=1)
    gmm.fit(trainX_use)
    tr_lab = gmm.predict(trainX_use)
    test_lab = gmm.predict(testX_S_use)

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(tr_lab.reshape(-1, 1))
    lab_train=enc.transform(tr_lab.reshape(-1, 1)).toarray()
    lab_test = enc.transform(test_lab.reshape(-1, 1)).toarray()

    r,c = np.shape(trainX_use)
    trainX_use_n = np.hstack((trainX_use,lab_train))
    testX_use_n = np.hstack((testX_S_use,lab_test))

    model = Sequential()
    model.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # checkpoint

    filepath="weights.best_leaf.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    callbacks_list = [checkpoint, TestCallback((testX_use_n,testY)),CSVLogger('1.log')]
    hist = model.fit(trainX_use_n[:-82,:],trainY_use[:-82,:], validation_data=(trainX_use_n[-82:,:],trainY_use[-82:,:]), epochs=5000, batch_size=40,verbose=0,callbacks=callbacks_list)

    model1 = Sequential()
    model1.add(Dense(250, input_dim=c+l_use[i], kernel_initializer='normal', activation='relu'))
    model1.add(Dense(36, kernel_initializer='normal', activation='softmax'))
    model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model1.load_weights("weights.best_leaf.h5")


    train_pred = convert_out(model1.predict(trainX_use_n[:-82,:]))
    train_act = trainY_comp_use[:-82]
    train_acc_lda_p_2.append((1-Pred_correct(train_pred,train_act)))

    val_pred = convert_out(model1.predict(trainX_use_n[-82:,:]))
    val_act = trainY_comp_use[-82:]
    val_acc_lda_p_2.append((1-Pred_correct(val_pred,val_act)))

    test_pred = convert_out(model1.predict(testX_use_n))
    test_acc_lda_p_2.append((1-Pred_correct(test_pred,testY_comp)))


# # EM + LDA + NN Plot

# In[58]:


fig = plt.figure()
ax = fig.gca()
ax.set_yticks(np.arange(0, 100, 10))
ax.set_xticks(np.arange(2,len(train_acc_lda_p_2)+2))

plt.scatter(np.arange(2,len(train_acc_lda_p_2)+2),np.array(train_acc_lda_p_2)*100,label='Training Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_lda_p_2)+2),np.array(val_acc_lda_p_2)*100,label='Validation Set Accuracy')
plt.scatter(np.arange(2,len(train_acc_lda_p_2)+2),np.array(test_acc_lda_p_2)*100,label='Testing Set Accuracy')
plt.xlabel('Number of Components')
plt.ylabel('Misclassification Error (%)')
plt.ylim([0,50])
plt.legend()
plt.grid(True)

results_f = pd.DataFrame(data = np.transpose([np.array(train_acc_lda_p_2)*100,np.array(val_acc_lda_p_2)*100,np.array(test_acc_lda_p_2)*100]),columns=['Training Set', 'Validation Set', 'Testing Set'])
print(results_f)

