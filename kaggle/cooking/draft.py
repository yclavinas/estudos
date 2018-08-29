#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 09:39:38 2018

@author: yurilavinas
"""
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_validate
from sklearn import model_selection


with open('train.json', 'r') as f:
    train = json.load(f)

with open('test.json', 'r') as f:
    test = json.load(f)

### processing train data

cuisinelist=[]
ingredientlist=[]
for data in train:
    cuisinelist.append(data['cuisine'])
    ingredientstr=' '.join(data['ingredients'])
    ingredientlist.append(ingredientstr)
    
traindata=pd.concat([pd.Series(cuisinelist),
                     pd.Series(ingredientlist)],
                   axis=1)
traindata.columns=['cuisine','ingredients']
#print(traindata.head())
traindata.groupby('cuisine').size().plot(kind='barh')


vectorizer=TfidfVectorizer()
vectorizer=vectorizer.fit(traindata['ingredients'])
trainvector=vectorizer.transform(traindata['ingredients']).astype('float')
print(trainvector.shape)

### processing test data
testingredientlist=[]
idlist=[]
for data in test:
    idlist.append(data['id'])
    ingredientstr=' '.join(data['ingredients'])
    testingredientlist.append(ingredientstr)

testvector=vectorizer.transform(pd.Series(testingredientlist)).astype('float')




le=LabelEncoder()
label=le.fit_transform(traindata['cuisine'])
x_train, x_test, y_train, y_test = train_test_split(trainvector,
                                                    label,
                                                    test_size=0.1)
# =============================================================================
# clf=XGBClassifier(n_jobs=2)
# clf.fit(x_train,y_train)
# cv_results = cross_validate(clf, x_train, y_train)
# cv_results['test_score'].mean()
# 
# =============================================================================
param_grid = {'n_estimators': [50, 100, 200, 400],
              'max_depth': [2,4,6], #max depth tree can grow; default is none
              'learning_rate': [0.1,0.01,0.03,0.09,0.3, 0.003],
              'random_state': [0] #seed or control random number generator: https://www.quora.com/What-is-seed-in-random-number-generation
             }

# =============================================================================
# cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 ) # run model 10x with 60/30 split intentionally leaving out 10%
# =============================================================================


tune_model = model_selection.GridSearchCV(XGBClassifier(n_jobs=2), param_grid=param_grid, scoring = 'accuracy')
tune_model.fit(x_train,y_train)
cv_results = cross_validate(tune_model, x_train, y_train)
cv_results['test_score'].mean()
tune_model.cv_results_['mean_test_score'].mean()


predictlabel=tune_model.predict(testvector)
predict_label=le.inverse_transform(predictlabel)
testresult=pd.concat([pd.Series(idlist),pd.Series(predict_label)],axis=1)
testresult.columns=['id','cuisine']
testresult.to_csv('result.csv',index=False)