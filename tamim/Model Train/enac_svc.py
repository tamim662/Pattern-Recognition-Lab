# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


# %%
x_train_pos=np.load('Dataset/numpy/A.thaliana5289_pos/ENAC.npy')
x_train_neg=np.load("Dataset/numpy/A.thaliana5289_neg/ENAC.npy")
x_test_pos=np.load("Dataset/numpy/A.thaliana1000indep_pos/ENAC.npy")
x_test_neg=np.load("Dataset/numpy/A.thaliana1000indep_neg/ENAC.npy")
y_train_pos=np.tile(1,5288)
y_train_neg=np.tile(0,5288)
y_test_pos=np.tile(1,999)
y_test_neg=np.tile(0,999)


# %%
print(x_test_neg.shape)
print(y_test_neg.shape)

print(x_train_neg.shape)
print(y_train_neg.shape)


# %%
seed = 40
np.random.seed(seed)


# %%
x_training = np.concatenate((x_train_pos, x_train_neg), axis= 0)
y_training = np.concatenate((y_train_pos, y_train_neg))


# %%
x_test = np.concatenate((x_test_pos, x_test_neg))
y_test = np.concatenate((y_test_pos, y_test_neg))


# %%
print(x_test.shape)
print(y_test.shape)


# %%
print(x_training.shape)


# %%
kf = KFold(n_splits = 5, random_state=seed)


# %%
c = 11
clf = SVC(C=c,  random_state = seed)


# %%
val_accuracy_list = []
train_accuracy_list = []
print(len(val_accuracy_list))
print(len(train_accuracy_list))


# %%

count = 1
for train_index, val_index in kf.split(x_training):
    print("Fold : ",count)
    X_train, X_val = x_training[train_index], x_training[val_index]
    y_train, y_val = y_training[train_index], y_training[val_index]
    
    clf.fit(X_train, y_train)

    y_val_pred = clf.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_accuracy_list.append(val_accuracy)

    y_train_pred = clf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_accuracy_list.append(train_accuracy)

    count += 1 

print("done")


# %%
for i in range(0, len(train_accuracy_list)):
    print("train_accuracy : ", train_accuracy_list[i], "\tVal accuracy : ", val_accuracy_list[i])


# %%
print("Train Acc.: ",(sum(train_accuracy_list))/len(train_accuracy_list))
print("Val Acc.:",(sum(val_accuracy_list))/len(val_accuracy_list))


# %%



