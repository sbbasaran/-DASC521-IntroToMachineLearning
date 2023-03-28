#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# Load the data
X = np.genfromtxt("/Users/ideakadikoy/Desktop/hw01_data_points.csv", delimiter=",", dtype=str)
y = np.genfromtxt("/Users/ideakadikoy/Desktop/hw01_class_labels.csv", delimiter=",", dtype=int)


# In[3]:


# STEP 3
# first 50000 data points should be included to train
# remaining 43925 data points should be included to test
# should return X_train, y_train, X_test, and y_test

    # your implementation starts below
# Split the data into train and test sets
def train_test_split(X, y):
    X_train = X[:50000]
    y_train = y[:50000]
    X_test = X[50000:]
    y_test = y[50000:]
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = train_test_split(X, y)
  
    # your implementation ends above

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[4]:


# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)

def estimate_prior_probabilities(y):
    # your implementation starts below
    K = np.unique(y)
    class_priors = []
    for classes in K:
        class_count = (y == classes).sum()
        class_priors = np.append(class_priors, class_count/len(y))
    
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)


# In[5]:


# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)


# In[7]:


def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
    class_counts = []
    K = np.unique(y)
    D = X.shape[1]
    nucleotides_probabilities = ["A", "C", "G", "T"]
    nucleotide_counts = np.zeros((len(K), D, len(nucleotides_probabilities)))
    for i, classes in enumerate(K):
        labels = y == classes
        class_count = (y == classes).sum()
        class_counts = np.append(class_counts, class_count)
        for j in range(X.shape[1]):
            column = X[labels, j]
            nucleotide_counts[i, j, 0] = np.sum(column == 'A')
            nucleotide_counts[i, j, 1] = np.sum(column == 'C')
            nucleotide_counts[i, j, 2] = np.sum(column == 'G')
            nucleotide_counts[i, j, 3] = np.sum(column == 'T')

    pAcd = (nucleotide_counts[:, :, 0]) / (class_counts[:, np.newaxis])
    pCcd = (nucleotide_counts[:, :, 1]) / (class_counts[:, np.newaxis])
    pGcd = (nucleotide_counts[:, :, 2]) / (class_counts[:, np.newaxis])
    pTcd = (nucleotide_counts[:, :, 3]) / (class_counts[:, np.newaxis])
    # your implementation ends above    
    return pAcd, pCcd, pGcd, pTcd

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)


# In[8]:


# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)


# In[9]:


def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    N = X.shape[0]
    D = X.shape[1]
    K = len(class_priors)
    score_values = np.zeros((N, K))
    
    for n in range(N):
        x = X[n]
        for k in range(K):
            score = np.log(class_priors[k])
            for d in range(D):
                score += np.log((pAcd[k][d])**(int(x[d] == 'A'))*(pCcd[k][d]**(int(x[d] == 'C')))*(pGcd[k][d]**(int(x[d] == 'G')))*(pTcd[k][d]**(int(x[d] == 'T'))))
            score_values[n][k] = score
    # your implementation ends above
    return score_values

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)


# In[10]:


# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)


# In[11]:


def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    K = scores.shape[1]
    confusion_matrix = np.zeros((K, K), dtype=int)
    y_bar = np.argmax(scores, axis=1)
    confusion_matrix[0,0] = np.sum((y_truth == 1) & (y_bar == 0))
    confusion_matrix[0,1] = np.sum((y_truth == 2) & (y_bar == 0))
    confusion_matrix[1,0] = np.sum((y_truth == 1) & (y_bar == 1))
    confusion_matrix[1,1] = np.sum((y_truth == 2) & (y_bar == 1))
    # your implementation ends above
    return confusion_matrix 

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)


# In[ ]:




