#!/usr/bin/env python
# coding: utf-8

# In[1]:


##package needed to do the exercise
import pandas as pd #package for reading and using dataset
import numpy as np #package needed for inserting value in dataset
from sklearn.metrics import roc_auc_score #needed for determing AUC score
import time as t
begin = t.perf_counter()


# In[2]:


df=pd.read_csv('C:/Users/s151675/Documents/Jaar_master_2/Kwartiel 3,4/Master project/Data aangepast/data_for_prediction/data+prediction_final/data_complete_final.csv', delimiter = ';') #read csv file


# In[3]:


df #visualize dataset


# In[4]:


##packages needed for using decision tree classifier and splitting dataset
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation


# In[5]:


#Considered one-hot-encoding in our data, see report.
spec_dummies = pd.get_dummies(df.Specialismecode) #OneHotencoder used for column job
transport_dummies = pd.get_dummies(df.Transport) #OneHotencoder used for column marital
spec_names = spec_dummies.columns
transport_names = transport_dummies.columns
##replace values from categorical to nummerical 
df = df.replace('Vrouw', 0)
df = df.replace('Man', 1)
##replace values from categorical to nummerical (with order)
df = df.replace('Blauw',0) 
df = df.replace('Groen', 1)
df = df.replace('Geel', 2)
df = df.replace('Oranje', 3)
df = df.replace('Rood', 4)

df = pd.concat([df.iloc[:,0],spec_dummies,df.iloc[:,6:9], transport_dummies, df.iloc[:,11], df.iloc[:,12:20]], axis=1) #Dataset with new numerical columns(job and marital)
df


# In[6]:


##define input variables(X) and output variable(Y)
feature_cols2 = df.columns
feature_cols = feature_cols2[0:df.shape[1]-5]
feature_cols_check = feature_cols2[1:df.shape[1]-1]
feature_cols_x = feature_cols2[1:df.shape[1]-5] # all columns excluding 'y' and case_ID
feature_cols_y = feature_cols2[df.shape[1]-5:df.shape[1]] 
# feature_cols = ['to_extern', 'INT', 'CHI', 'Age', 'registering_time', 'MDL', 'ORT', 'History_bin', 'CAR', 'Triage color', 'NEU','Lab']#take relevant features to speed up process
X = df[feature_cols] #input variables
X_ana = df[feature_cols_x]
X_ana_big = df[feature_cols_check]
y = df[feature_cols_y]# Target variable
print(X)
print(y)


# In[7]:


# Split dataset into training set and test set
X_train_t, X_test_t, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # divide dataset into 70% training and 30% test
X_train = X_train_t.iloc[:,1:40]
X_test = X_test_t.iloc[:,1:40]
print(X_train)
y_train_lab = y_train.iloc[:,0]
y_train_rad_to_ext = y_train.iloc[:,1]
y_train_consult = y_train.iloc[:,2]
y_train_opname = y_train.iloc[:,3]
y_test_lab = y_test.iloc[:,0]
y_test_rad_to_ext = y_test.iloc[:,1]
y_test_consult = y_test.iloc[:,2]
y_test_opname = y_test.iloc[:,3]
X_number = X_test_t.iloc[:,0] #take caseID from test set to compare with actual value in later phase
###dataset needed for making predictions with already predicted values
X_train_big = pd.concat([X_train_t.iloc[:,1:40],y_train.iloc[:,0:4]], axis=1)
##choose between these test_data set, where perf is with actual values and other is with predicted values
# X_test_big = X_test #make this out of prediction
X_test_big_perf = pd.concat([X_test_t.iloc[:,1:40],y_test.iloc[:,0:4]], axis=1) #to compare with existing values of columns
y_train_big = y_train.iloc[:,-1]
y_test_big = y_test.iloc[:,-1]


# In[8]:


#Function made to measure the confusing matrix
def perf_measure(y_actual, y_hat):
    ## start with zero values
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    count = 0 
    for i in range(len(y_hat)): #for all values in the column y_test
        count += 1 #count the total number of rows
        if y_actual[i]==1 and y_hat[i]==1:#True positive
            TP += 1
        if y_hat[i]==1 and y_actual[i]==0:#False positive
            FP += 1
        if y_actual[i]==0 and y_hat[i]==0:#True negative
            TN += 1
        if y_hat[i]==0 and y_actual[i]==1:#False negative
            FN += 1

    return(TP/count, FP/count, TN/count, FN/count, count)#return all relevant values in fractions


# In[9]:


def RandomForest(Number_leaves, X_train, y_train, X_test):
    clf = RandomForestClassifier(n_estimators = 100, min_samples_leaf=Number_leaves,class_weight = "balanced") #define decision tree with selected variable
    clf = clf.fit(X_train,y_train) #train the decision tree on training set
    y_pred = clf.predict(X_test) #predict the values of X_test
    
    return(clf, y_pred)


# In[10]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[11]:


def predict_value(y_train, y_test, X_train, X_test, X_ana, feature_cols2, negative):
    ##used for determing best values
    max_score = 0
    max_AUC = 0
    best_leaves = 0
    y_test = np.array(y_test)
    clf_final, y_pred = RandomForest(2, X_train, y_train, X_test)
    ##loop over values from 2 to 100 for selecting best parameter
    for i in range(2,200):
        clf, y_pred = RandomForest(i,X_train, y_train, X_test)
    #     AUC_score = roc_auc_score(y_test,y_pred)
    #     print(accuracy_score(y_test_lab, y_pred))
        TPRate,FPRate,TNRate, FNRate, count = perf_measure(y_test, y_pred) #call confusing matrix function to determine these values in matrix
        if(negative==False):
            accuracy = (TPRate+TNRate)/(TPRate+TNRate+FPRate+FNRate)
            print("accuracy :", accuracy," for num of leaves =", i)
        else:
            accuracy = TNRate/(TNRate+FNRate)
            print("Negative prediction value :", accuracy," for num of leaves =", i)
        ##function used for updating best score if one score exceed previously best scores
        
        if accuracy>=max_score:
            max_score = accuracy
            best_leaves = i
            clf_final = clf
    
    clf, y_pred = RandomForest(best_leaves,X_train, y_train, X_test) #Best decision tree based on highest accuracy 
    TPRate,FPRate,TNRate, FNRate, count = perf_measure(y_test, y_pred) #call confusing matrix function to determine these values in matrix
    p = (((TNRate+FPRate)*count)/count*((TNRate+FNRate)*count)/count) #measure p value for best model based on recall & AUC score
    kappa_score = (metrics.accuracy_score(y_test, y_pred)-p)/(1-p) #measure Kappa score for best model based on recall & AUC score
    AUC_score = roc_auc_score(y_test,y_pred) #measure AUC score for best model based on recall & AUC score
    ##measure Accuracy score for best model based on Recall & AUC score
    accuracy = (TPRate+TNRate)/(TPRate+TNRate+FPRate+FNRate)

    ###feature importance start
    forest = ExtraTreesClassifier(n_estimators=100,min_samples_leaf=best_leaves, class_weight = "balanced")
    forest = forest.fit(X_train,y_train)

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
    indices = np.argsort(importances)[::-1]
    ##print all scores for best model based on recall & AUC score
    print("best number of leaves = ", best_leaves, " with score = ", max_score, " and accuracy = ", accuracy)
    print("kappa score =", kappa_score)
    print("AUC score =", AUC_score)
    print(confusion_matrix(y_test,y_pred))
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(X_ana.shape[1]):
        print("%d. %s (%f)" % (f + 1, feature_cols2[indices[f]], importances[indices[f]]))
    return(clf_final)


# In[12]:


forest_big4 = predict_value(y_train_big, y_test_big, X_train_big, X_test_big_perf, X_ana_big, feature_cols_check, True) ##have to change X_train last features
eind = t.perf_counter()
print("number of seconds =", eind-begin)


# In[ ]:





# In[ ]:




