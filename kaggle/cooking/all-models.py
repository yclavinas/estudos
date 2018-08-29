#!/anaconda/bin/python

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
import re
from nltk import word_tokenize as TK
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection

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



MLA = [
    # sklearn.neural_network
    # MLPClassifier(),
    # #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    # XGBClassifier(), 
    #SVM
    # svm.SVC(),
    # svm.NuSVC(),
    # svm.LinearSVC(),#no sparse 79
    #Trees    
    # tree.DecisionTreeClassifier(),
    # tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    # discriminant_analysis.LinearDiscriminantAnalysis(),#no sparse
    # discriminant_analysis.QuadraticDiscriminantAnalysis(),#no sparse

    #Nearest Neighbor
    # neighbors.KNeighborsClassifier(),

    # Ensemble Methods
    # ensemble.AdaBoostClassifier(),
    # ensemble.BaggingClassifier(),
    # ensemble.ExtraTreesClassifier(),
    # ensemble.GradientBoostingClassifier(),
    # ensemble.RandomForestClassifier(),

    # #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),#no sparse
    
    #GLM
    # linear_model.LogisticRegressionCV(), 79(?)
    # linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    # linear_model.SGDClassifier(),
    # linear_model.Perceptron(),
    
    # #Navies Bayes
    # naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB()#no sparse
    
    
    
   
    
    

    ]


#split dataset in cross-validation with this splitter class: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.ShuffleSplit.html#sklearn.model_selection.ShuffleSplit
#note: this is an alternative to train_test_split
cv_split = model_selection.ShuffleSplit(n_splits = 1, test_size = .3, train_size = .7, random_state = 0) # run model 10x with 60/30 split intentionally leaving out 10%

#create table to compare MLA metrics
MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
MLA_predict = y_train

#index through MLA and save performance to table
row_index = 0
for alg in MLA:
    print(alg)
    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    if (type(alg) == type(gaussian_process.GaussianProcessClassifier())):
        cv_results = model_selection.cross_validate(alg, X_train.toarray(), y_train_ec, cv  = cv_split)
    elif (type(alg) == type(naive_bayes.GaussianNB())):
        cv_results = model_selection.cross_validate(alg, X_train.toarray(), y_train_ec, cv  = cv_split)
    elif (type(alg) == type(svm.LinearSVC())):
        cv_results = model_selection.cross_validate(alg, X_train.toarray(), y_train_ec, cv  = cv_split)
    elif (type(alg) == type(tree.ExtraTreeClassifier())):
        cv_results = model_selection.cross_validate(alg, X_train.toarray(), y_train_ec, cv  = cv_split)
    elif (type(alg) == type(discriminant_analysis.LinearDiscriminantAnalysis())):
        cv_results = model_selection.cross_validate(alg, X_train.toarray(), y_train_ec, cv  = cv_split)
    elif (type(alg) == type(discriminant_analysis.QuadraticDiscriminantAnalysis())):
        cv_results = model_selection.cross_validate(alg, X_train.toarray(), y_train_ec, cv  = cv_split)
    else:
        cv_results = model_selection.cross_validate(alg, X_train, y_train_ec, cv  = cv_split)
    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3
    print(MLA_compare.loc[row_index])
    alg.fit(X_train, y_train_ec)
    MLA_predict[MLA_name] = le.inverse_transform(alg.predict(X_pred))
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare