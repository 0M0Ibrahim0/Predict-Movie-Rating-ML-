import json
import math
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.model_selection import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


##################################################################
################### Global Variables #############################
##################################################################
distinct_Values = dict()
##################################################################


def draw_missing_data_table(fileName):
    total = fileName.isnull().sum().sort_values(ascending=False)
    percent = (fileName.isnull().sum()/fileName.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

def getCell(colStr):
    data = json.loads(colStr)
    ret = []
    for i in data:
        ret.append(i['name'].lower())
    return ret

def get_Distinct_Values(df, colName):
    rows = len(df[colName])
    mp = dict()
    for i in range(rows):
        if type(df[colName][i]) != type('m3lsh'):
            continue
        tmpList = getCell(df[colName][i])
        for val in tmpList:
            if val in mp:
                mp[val] = mp[val] + 1
            else:
                mp[val] = 1
    ret = []
    for f in mp.items():
        x, y = f
        ret.append((y, x))

    ret.sort()
    ret.reverse()

    if len(ret) > 55:
        ret = ret[:55]

    if colName in distinct_Values:
        return distinct_Values[colName]
    else:
        new_ret = []
        for f in ret:
            value, name = f
            new_ret.append(name)
        distinct_Values[colName] = new_ret
        return new_ret

def create_col(df, colName):
    newCols = get_Distinct_Values(df, colName)
    available = dict()
    rows = len(df[colName])

    zeroes = [0] * rows

    for name in newCols:
        if name not in available:
            available[name] = zeroes

    for i in range(rows):
        if type(df[colName][i]) != type('m3lsh'):
            continue
        tmpList = getCell(df[colName][i])
        for name in tmpList:
            if name in available:
                available[name][i] = 1

    for i in available.items():
        name, values = i
        df[name] = values

    df.drop(colName, axis=1, inplace=True)

def Fill_MissingData(df):
    features = ['budget',
                'popularity',
                'revenue',
                'runtime',
                'vote_count']
    for feature in features:
        df[feature] = df[feature].replace(0,df[feature].mean())

def Normalize_Data(df):
    features = df.columns.values
    for feature in features:
        l = df[feature].values
        mn, mx = float(min(l)), float(max(l))
        for i in range(len(l)):
            if mx - mn == 0:
                l[i] = 0
            else:
                l[i] = float((float(l[i] - mn))/float((mx - mn))) * 2
        df.drop(feature, axis=1, inplace=True)
        df[feature] = l

def PCA(df, yName, no_of_features):
    L = []
    for feature in df.columns.values:
        if feature != yName:
            L.append(feature)

    A, B = df[L], df[yName]
    A = StandardScaler().fit_transform(A)

    pca = PCA(n_components=no_of_features)
    principalComponents = pca.fit_transform(A)

    tmp = []
    for i in range(no_of_features):
        tmp.append('m3lsh ' + str(i + 1))

    principalDf = pd.DataFrame(data=principalComponents, columns=tmp)
    finalDf = pd.concat([principalDf, df[[yName]]], axis=1)

    ##################################################################
    finalDf.dropna(how='any', inplace=True)
    ##################################################################

    return finalDf

def prepro(moviesFile, creditsFile, yName, PCA = False, flag = False):
    #Reading 2 Excel Files
    df1 = None
    try:
        df1 = pd.read_csv(moviesFile)
    except:
        df1 = pd.read_excel(moviesFile)
    df2 = pd.read_csv(creditsFile)
    df2.drop('title', axis=1, inplace=True)

    ##################################################################
    df = pd.merge(df1, df2, left_on='id',right_on='movie_id')
    ##################################################################

    # Dropping non important columns
    drop_list = ['homepage', 'original_title', 'original_language', 'overview', 'status', 'movie_id', 'tagline', 'id', 'title']#, 'keywords', 'production_companies', 'production_countries', 'spoken_languages', 'crew', 'cast']
    for feature in drop_list:
        df.drop(feature, axis=1, inplace=True)

    ######################################################
    if flag == False:
        df = Feature_Encoder(df,[yName])
    ######################################################

    # Handling complex columns
    cmplx_list = ['genres', 'keywords', 'production_companies', 'production_countries', 'spoken_languages', 'crew', 'cast']
    for feature in cmplx_list:
        create_col(df, feature)
        #df.drop(feature, axis=1, inplace=True)

    ##################################################################

    ##################################################################
    df.dropna(how='any', inplace=True)
    ##################################################################

    # Filling Null and noise values with mean and normalizing each column
    Fill_MissingData(df)
    Normalize_Data(df)

    ##################################################################

    # Using Correlation to eleminate non-necessary features
    # corr = df.corr()
    # top_feature = corr.index[abs(corr['rate']>0.5)]
    # top_corr=df[top_feature].corr()
    # print(top_feature)

    # Plotting features with Correlation more than 0.5
    # plt.subplots(figsize=(12,8))
    # sns.heatmap(top_corr,annot=True)
    # plt.show()

    ##################################################################

    ret = df
    if PCA == True:
        ret = PCA(df, yName, 300)
    return ret

def split_Data(yName, df1, df2 = 'm3lsh', flag = False):
    if type(df2) == type('m3lsh'):
        df = df1
        L = []
        for name in df1.columns.values:
            if name != yName:
                L.append(name)

        X = np.asarray(df[L])
        y = np.asarray(df[yName])
        arr, arr2 = [], []

        for k in range(3):
            tmp, tmp2 = [], []
            for i in range(X.shape[0]):
                if (y[i] == k):
                    tmp.append(X[i])
                    tmp2.append(k)

            arr.append(tmp)
            arr2.append(tmp2)

        if flag == True:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, shuffle=True)
        else:
            X0_train, X0_test, y0_train, y0_test = train_test_split(arr[0], arr2[0], test_size=.2, shuffle=True)
            X1_train, X1_test, y1_train, y1_test = train_test_split(arr[1], arr2[1], test_size=.2, shuffle=True)
            X2_train, X2_test, y2_train, y2_test = train_test_split(arr[2], arr2[2], test_size=.2, shuffle=True)

            X_train = list(X0_train) + list(X1_train) + list(X2_train)
            X_test = list(X0_test) + list(X1_test) + list(X2_test)
            y_train = list(y0_train) + list(y1_train) + list(y2_train)
            y_test = list(y0_test) + list(y1_test) + list(y2_test)

            X_train = shuffle(X_train)
            X_test = shuffle(X_test)
            y_train = shuffle(y_train)
            y_test = shuffle(y_test)
        return X_train, X_test, y_train, y_test

    else:
        L = []
        for name in df2.columns.values:
            if name != yName:
                L.append(name)

        X_train = df1[L]
        y_train = df1[yName]
        X_test = df2[L]
        y_test = df2[yName]
        return X_train, X_test, y_train, y_test

df1, df2 = None, None
yName = 'rate'
df1 = prepro('tmdb_5000_movies_classification.csv', 'tmdb_5000_credits.csv', yName)
#df2 = prepro('samples_tmdb_5000_movies_testing_classification.xlsx', 'samples_tmdb_5000_credits_test.csv', yName)

X_train, X_test, y_train, y_test = split_Data(yName, df1)