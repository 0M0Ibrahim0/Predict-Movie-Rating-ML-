import numpy as np
import time
import pandas as pd
from pandas import crosstab
import seaborn as sns
from sklearn import *
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
import preprocessing
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import KMeans
import pickle
from sklearn.ensemble import AdaBoostClassifier
import os
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

class performance:
    def __init__(self, X_train ,predict, y):
        print('Mean Square Error', metrics.mean_squared_error(np.asarray(y), predict))
        r2 = r2_score(np.asarray(y), predict)
        print("R_squared", r2)
        n = len(X_train)
        n -= 1
        p = X_train.shape[1]
        adj_r2 = 1 - ((1 - r2) * ((n) / (n - p)))
        print("Adjusted R-squared", adj_r2)

class  regression:
    def __init__(self):
        self.X_train = preprocessing.X_train
        self.y_train=preprocessing.y_train
        self.X_test=preprocessing.X_test
        self.y_test=preprocessing.y_test
        pass
    #--------------- models ----------------
    def multiple_linear_regression(self):
        muliple_linear_model = linear_model.LinearRegression()
        muliple_linear_model.fit(self.X_train, self.y_train)
        cv=int(self.X_train.shape[0]/10)
        scores = cross_val_score(muliple_linear_model, self.X_train, self.y_train, cv=cv)
        self.prediction = muliple_linear_model.predict(self.X_test)
        performance(self.X_test,self.prediction,self.y_test)
        self.intercept=muliple_linear_model.intercept_
        self.coeff = muliple_linear_model.coef_

    def polynomial(self, degree):
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(self.X_train)
        poly_model = linear_model.LinearRegression()
        poly_model.fit(X_train_poly, self.y_train)
        self.prediction = poly_model.predict(poly_features.fit_transform(self.X_test))
        performance(self.X_test,self.prediction,self.y_test)
        self.intercept = poly_model.intercept_
        self.coeff = poly_model.coef_

    def ridge(self, alpha):
        Ridge_model = linear_model.Ridge(alpha=alpha)
        Ridge_model.fit(self.X_train, self.y_train)
        cv = int(self.X_train.shape[0] / 10)
        scores = cross_val_score(Ridge_model, self.X_train, self.y_train, cv=cv)
        self.prediction = Ridge_model.predict(self.X_test)
        performance(self.X_test,self.prediction,self.y_test)
        self.intercept = Ridge_model.intercept_
        self.coeff = Ridge_model.coef_

    def lasso(self, alpha):
        lasso_model = linear_model.Lasso(alpha=alpha)
        lasso_model.fit(self.X_train, self.y_train)
        cv = int(self.X_train.shape[0] / 10)
        scores = cross_val_score(lasso_model, self.X_train, self.y_train, cv=cv)
        self.prediction = lasso_model.predict(self.X_test)
        performance(self.X_test, self.prediction, self.y_test)
        self.intercept = lasso_model.intercept_
        self.coeff = lasso_model.coef_


    def elasticNet(self):
        cv = int(self.X_train.shape[0] / 10)
        ElasticNet_model = linear_model.ElasticNetCV(cv=cv)
        ElasticNet_model.fit(self.X_train, self.y_train)
        self.prediction = ElasticNet_model.predict(self.X_test)
        performance(self.X_test, self.prediction, self.y_test)
        self.intercept = ElasticNet_model.intercept_
        self.coeff = ElasticNet_model.coef_


class classification:
    def __init__(self):
        self.X_train = preprocessing.X_train
        self.y_train = preprocessing.y_train
        self.X_test = preprocessing.X_test
        self.y_test = preprocessing.y_test
        self.Models=['SVM OVR','SVM OVO','Logistic','KMeans','KNN','Adaboost','DecisionTree']
        self.trainTime=[0,0,0,0,0,0,0]
        self.testTime = [0,0,0,0,0,0,0]
        self.accuracy=[0,0,0,0,0,0,0]
        pass

    def SupportVectorMachine(self):
        if (os.path.exists('OVR_model.sav.meta')):
            svm_model_linear_ovr= pickle.load(open('OVR_model.sav', 'rb'))
        else :
            c = [1, 1000, 1000000]
            ker = ['linear', 'poly', 'rbf']
            for i in range(len(c)):
                start = time.time()
                svm_model_linear_ovr = OneVsRestClassifier(SVC(kernel='rbf',gamma=0.4, C=c[i])).fit(self.X_train, self.y_train)
                end = time.time()
                self.trainTime[0]=(end - start)/60
                accuracyTest = svm_model_linear_ovr.score(self.X_test, self.y_test)
                accuracyTrain = svm_model_linear_ovr.score(self.X_train, self.y_train)
                print("accuracyTrain " ,accuracyTrain*100, " accuracyTest " , accuracyTest*100)
            filename = 'OVR_model.sav'
            #pickle.dump(svm_model_linear_ovr, open(filename, 'wb'))

        accuracy = svm_model_linear_ovr.score(self.X_test, self.y_test)

        start = time.time()
        prediction= svm_model_linear_ovr.predict(self.X_test)
        end = time.time()
        self.testTime[0]=(end - start)/60

        print('One VS Rest SVM accuracy: ' + str(accuracy * 100))
        self.accuracy[0]=accuracy
        print(confusion_matrix(self.y_test, prediction))

        if (os.path.exists('OVO_model.sav.meta')):
            svm_model_linear_ovo= pickle.load(open('OVO_model.sav', 'rb'))
        else :
            for i in [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,1]:
                start = time.time()
                svm_model_linear_ovo = SVC(kernel='rbf', gamma=i, C=0.1).fit(self.X_train, self.y_train)
                end = time.time()
                self.trainTime[1]=(end - start) / 60
                accuracyTest= svm_model_linear_ovo.score(self.X_test, self.y_test)
                accuracyTrain = svm_model_linear_ovo.score(self.X_train, self.y_train)
                print("accuracyTrain ",accuracyTrain," accuracyTest ",accuracyTest)
            filename = 'OVO_model.sav'
            #pickle.dump(svm_model_linear_ovo, open(filename, 'wb'))

        accuracy = svm_model_linear_ovo.score(self.X_test, self.y_test)

        start = time.time()
        prediction = svm_model_linear_ovo.predict(self.X_test)
        end = time.time()
        self.testTime[1]=(end - start) / 60

        print('One VS One SVM accuracy: ' + str(accuracy * 100))
        self.accuracy[1]= accuracy
        print(confusion_matrix(self.y_test, prediction))

    def LogisticRegression(self):
        if (os.path.exists('Logistic_model.sav.meta')):
            logistic_regression_model = pickle.load(open('Logistic_model.sav', 'rb'))
        else :
            logistic_regression_model = LogisticRegression(multi_class='auto')

            start = time.time()
            logistic_regression_model.fit(self.X_train,self.y_train)
            end = time.time()
            self.trainTime[2]=(end - start) / 60

            filename = 'Logistic_model.sav'
            #pickle.dump(logistic_regression_model, open(filename, 'wb'))


        start = time.time()
        prediction = logistic_regression_model.predict(self.X_test)
        end = time.time()
        self.testTime[2]=(end - start) / 60

        accuracy = logistic_regression_model.score(self.X_test, self.y_test)
        #accuracy = logistic_regression_model.score(self.X_train, self.y_train)
        print("Logistic Regression accuracy: "+ str(accuracy * 100))
        self.accuracy[2]=accuracy
        print(confusion_matrix(self.y_test, prediction))

    def KMeans(self):
        if (os.path.exists('kmeans_model.sav.meta')):
            kmeans_model = pickle.load(open('kmeans_model.sav', 'rb'))
        else :
            for i in range(10):
                kmeans_model = KMeans(n_clusters=i)
                start = time.time()
                kmeans_model.fit(self.X_train)
                end = time.time()
                self.trainTime[3]=(end - start) / 60
                accuracy = kmeans_model.score(self.X_test, self.y_test)
                print("K-Means Accuracy: "+" with #clusters "+i +" "+ str(accuracy * 100))
            filename = 'kmeans_model.sav'
            pickle.dump(kmeans_model, open(filename, 'wb'))

        start = time.time()
        prediction = kmeans_model.predict(self.X_test)
        end = time.time()
        self.testTime[3]=(end - start) / 60

        accuracy = kmeans_model.score(self.X_test, self.y_test)
        self.accuracy[3]=accuracy
        print("K-Means Accuracy: " + str(accuracy * 100))
        print(confusion_matrix(self.y_test, prediction))

    def KNearstNeighbors(self):
        scaler = StandardScaler()
        scaler.fit(self.X_train)

        X_train = scaler.transform(self.X_train)
        X_test = scaler.transform(self.X_test)

        if (os.path.exists('knn_model.sav.meta')):
            knn_model = pickle.load(open('knn_model.sav', 'rb'))
        else :
            for i in range (40):
                knn_model = KNeighborsClassifier(n_neighbors=i)
                start = time.time()
                knn_model.fit(X_train, self.y_train)
                end = time.time()
                accuracy = knn_model.score(self.X_test, self.y_test)
                print("K-NN Accuracy: " +"with #k neighbours = "+i+" "+ str(accuracy * 100))
            self.trainTime[4]=(end - start) / 60
            filename = 'knn_model.sav'
            pickle.dump(knn_model, open(filename, 'wb'))

        start = time.time()
        prediction = knn_model.predict(X_test)
        end = time.time()
        self.testTime[4]=(end - start) / 60

        accuracy = knn_model.score(self.X_test, self.y_test)
        print("K-NN Accuracy: " + str(accuracy * 100))
        self.accuracy[4]=accuracy
        print(confusion_matrix(self.y_test, prediction))

    def DecisionTree_ADAboost (self):

        scaler = StandardScaler()
        scaler.fit(self.X_train)
        X_train = scaler.transform(self.X_train)
        X_test = scaler.transform(self.X_test)

        if (os.path.exists('bdt.sav.meta')):
            bdt = pickle.load(open('bdt.sav', 'rb'))
        else :
            bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=100),
                                     algorithm="SAMME.R",
                                     n_estimators=1000)

            start = time.time()
            bdt.fit(X_train, self.y_train)
            end = time.time()
            self.trainTime[5]=(end - start) / 60
            filename = 'bdt.sav'
            pickle.dump(bdt, open(filename, 'wb'))


        start = time.time()
        prediction = bdt.predict(X_test)
        end = time.time()
        self.testTime[5]=(end - start) / 60

        accuracy = np.mean(prediction == self.y_test)
        print("Adaboost Accuracy: " + str(accuracy * 100))
        self.accuracy[5]=accuracy
        print(confusion_matrix(self.y_test, prediction))


        if (os.path.exists('clf.sav.meta')):
            bdt = pickle.load(open('clf.sav', 'rb'))
        else :
            clf = tree.DecisionTreeClassifier(max_depth=1000)
            start = time.time()
            clf.fit(X_train, self.y_train)
            end = time.time()
            self.trainTime[6]=(end - start) / 60
            filename = 'clf.sav'
            pickle.dump(clf, open(filename, 'wb'))

        start = time.time()
        prediction = clf.predict(X_test)
        end = time.time()
        self.testTime[6]=(end - start) / 60


        accuracy = np.mean(prediction == self.y_test)
        print("Decision-Tree  Accuracy: " + str(accuracy * 100))
        self.accuracy[6]=accuracy
        print(confusion_matrix(self.y_test, prediction))

    def bars(self):
        y_pos=np.arange(len(self.Models))
        plt.bar(y_pos,self.trainTime,align='center',alpha=0.5)
        plt.xticks(y_pos,self.Models)
        plt.ylabel('Train Time')
        plt.title("Train time without PCA")
        plt.show()

        #########################################

        y_pos = np.arange(len(self.Models))
        plt.bar(y_pos, self.testTime, align='center', alpha=0.5)
        plt.xticks(y_pos, self.Models)
        plt.ylabel('test Time')
        plt.title("Test time without PCA")
        plt.show()

        ##########################################

        y_pos = np.arange(len(self.Models))
        plt.bar(y_pos, self.accuracy, align='center', alpha=0.5)
        plt.xticks(y_pos, self.Models)
        plt.ylabel('Accuracy ')
        plt.title("Accuracy without PCA")
        plt.show()

c = classification()
c.SupportVectorMachine()
#c.LogisticRegression()
#c.KMeans()
#c.KNearstNeighbors()
#c.DecisionTree_ADAboost()
#c.bars()


'''
c=regression()
c.multiple_linear_regression()
for i in range(len(preprocessing.L)):
    feature = preprocessing.L[i]

    plt.scatter(preprocessing.df[feature], preprocessing.df['vote_average'])
    mean = preprocessing.df[feature].median()
    featureList = np.array([preprocessing.df[feature].min(), preprocessing.df[feature].max()])
    # print(c.coeff, c.intercept)
    # y = featureList * c.coeff[i] + c.intercept
    # print(preprocessing.df[feature].shape[1])
    y=np.dot(featureList,c.coeff[i])+c.intercept
    # print(y)
    # y = np.sum(y, np.ones((y.shape[0])))
    y += mean
    plt.plot(np.array(featureList),y,color='red',linewidth=3)
    plt.show()
'''































