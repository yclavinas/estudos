#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 09:16:40 2018

@author: yurilavinas
"""

import pandas as pd
import re
from nltk import word_tokenize as TK
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.svm import SVC

# substitute the matched pattern
def sub_match(pattern, sub_pattern, ingredients):
    for i in ingredients.index.values:
        for j in range(len(ingredients[i])):
            ingredients[i][j] = re.sub(pattern, sub_pattern, ingredients[i][j].strip())
            ingredients[i][j] = ingredients[i][j].strip()
    re.purge()
    return ingredients

def regex_sub_match(series):
    # remove all units
    p0 = re.compile(r'\s*(oz|ounc|ounce|pound|lb|inch|inches|kg|to)\s*[^a-z]')
    series = sub_match(p0, ' ', series)
    # remove all digits
    p1 = re.compile(r'\d+')
    series = sub_match(p1, ' ', series)
    # remove all the non-letter characters
    p2 = re.compile('[^\w]')
    series = sub_match(p2, ' ', series)
    return series


# remove all the words that are not nouns -- keep the essential ingredients
def lemma(series):
    for i in series.index.values:
        for j in range(len(series[i])):
            # get rid of all extra spaces
            series[i][j] = series[i][j].strip()
            # Tokenize a string to split off punctuation other than periods
            token = TK(series[i][j])
            # set all the plural nouns into singular nouns
            for k in range(len(token)):
                token[k] = lemmatizer.lemmatize(token[k])
            token = ' '.join(token)
            # write them back
            series[i][j] = token
    return series


def submit_csv(y_pred_model, name):
    submit_df = pd.DataFrame()
    submit_df['id'] = rawdf_te['id']
    submit_df['cuisine'] = y_pred_model
    submit_df.to_csv(name, index=False)
    
    
# load the data - test data
rawdf_te = pd.read_json(path_or_buf='test.json')

# load the data - train data
rawdf_tr = pd.read_json(path_or_buf='train.json')

# copy the series from the dataframe
ingredients_tr = rawdf_tr['ingredients']
# do the test.json while at it
ingredients_te = rawdf_te['ingredients']


    
# regex both train and test data
ingredients_tr = regex_sub_match(ingredients_tr)
ingredients_te = regex_sub_match(ingredients_te)

# declare instance from WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# lemmatize both train and test data
ingredients_tr = lemma(ingredients_tr)
ingredients_te = lemma(ingredients_te)


# copy back to the dataframe
rawdf_tr['ingredients_lemma'] = ingredients_tr
rawdf_tr['ingredients_lemma_string'] = [' '.join(_).strip() for _ in rawdf_tr['ingredients_lemma']]
# do the same for the test.json dataset
rawdf_te['ingredients_lemma'] = ingredients_te
rawdf_te['ingredients_lemma_string'] = [' '.join(_).strip() for _ in rawdf_te['ingredients_lemma']]


# DataFrame for training and validation
traindf = rawdf_tr[['cuisine', 'ingredients_lemma_string']].reset_index(drop=True)
print(traindf.shape)
traindf.head()

testdf = rawdf_te[['ingredients_lemma_string']]
print(testdf.shape)
testdf.head()

# training ===================
# X_train
X_train_ls = traindf['ingredients_lemma_string']
vectorizertr = TfidfVectorizer()
X_train = vectorizertr.fit_transform(X_train_ls)

# y_train
y_train = traindf['cuisine']
# for xgboost the labels need to be labeled with encoder
le = LabelEncoder()
y_train_ec = le.fit_transform(y_train)

# predicting =================
# X_test
X_pred_ls = testdf['ingredients_lemma_string']
vectorizerts = TfidfVectorizer(stop_words='english')
X_pred = vectorizertr.transform(X_pred_ls)


clf_mlp = MLPClassifier(activation='logistic', alpha=0.0001, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=True, epsilon=1e-08,
       hidden_layer_sizes=(100,), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=0, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
clf_mlp = clf_mlp.fit(X_train, y_train)
y_pred_mlp = clf_mlp.predict(X_pred)

submit_csv(y_pred_mlp, 'NN_MLP.csv')

# Best estimator after running the grid search cross validation for the XGBoost model
clf_xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=1, learning_rate=0.01, max_delta_step=0,
       max_depth=12, min_child_weight=1, missing=None, n_estimators=1000,
       n_jobs=2, nthread=None, objective='multi:softprob', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=0.8)
clf_xgb = clf_xgb.fit(X_train, y_train_ec)
y_pred_xgb = clf_xgb.predict(X_pred)
# inverse transform the encoded cuisine column
y_pred_xgb = le.inverse_transform(y_pred_xgb)

submit_csv(y_pred_xgb, 'XGBoost.csv')

clf_svc = SVC(degree=3, # default value, not tuned yet
	 			 gamma=1, # kernel coefficient, not tuned yet
	 			 coef0=1, # change to 1 from default value of 0.0
	 			 shrinking=True, # using shrinking heuristics
	 			 tol=0.001, # stopping criterion tolerance 
	      		 probability=False, # no need to enable probability estimates
	      		 cache_size=200, # 200 MB cache size
	      		 class_weight=None, # all classes are treated equally 
	      		 verbose=False, # print the logs 
	      		 max_iter=-1, # no limit, let it run
          		 decision_function_shape=None, # will use one vs rest explicitly 
          		 random_state=None)
parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[1, 100]}

model = GridSearchCV(clf_svc, parameters)
clf_svc = clf_svc.fit(X_train, y_train_ec)
y_pred_svc = clf_svc.predict(X_pred)

y_pred_svc = le.inverse_transform(y_pred_svc)
# 
submit_csv(y_pred_svc, 'SVM-svc.csv')
