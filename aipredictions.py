import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as seabornInstance
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import  GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict

dataset=pd.read_csv('C://Users//Daddy//Desktop//datasets//train.csv')
test=pd.read_csv('C://Users//Daddy//Desktop//datasets//test.csv')
print(dataset.head())
print(dataset.shape)
print(dataset.describe())
count_classes = pd.value_counts(dataset['booker'],sort=True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Advertising or not advertising")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()
print(dataset.isnull().sum()) #to view any null values present at the column level
print(dataset.dtypes) #to see the data type of variables which present in the dataset
correlation = dataset.corr()# to see the correlation between the independent and dependent variables
plt.figure(figsize=(1, 1))
seabornInstance.heatmap(correlation, vmax=1, square=True,annot=True)
plt.title('Correlation between different fearures')
plt.show()#to view the correlation between the variable

advertisingtosite = dataset[dataset.booker == 1].index#sampling of the data to eliminate the imbalancing problem
random_indices = np.random.choice(advertisingtosite, len(dataset.loc[dataset.booker == 1]), replace=False)
advertising_sample = dataset.loc[random_indices]
not_advertising = dataset[dataset.booker == 0].index
random_indices = np.random.choice(not_advertising, sum(dataset['booker']), replace=False)
not_advertising_sample = dataset.loc[random_indices]
dataset_new = pd.concat([not_advertising_sample, advertising_sample], axis=0)
print("Percentage of bookers not advertisng into the site: ", len(dataset_new[dataset_new.booker == 0])/len(dataset_new))
print("Percentage of bookers on advertisements: ", len(dataset_new[dataset_new.booker == 1])/len(dataset_new))
print("Total number of records in resampled data: ", len(dataset_new))
dataset_new_new = dataset_new.sample(frac=1).reset_index(drop=True)
X = dataset_new['clickouts'].values.reshape(-1,1) #all columns except the last one
y = dataset_new['booker'].values.reshape(-1,1)

scaler = StandardScaler()#to normalize the data means if any big variances in range of values it will normalize
X=scaler.fit_transform(X)
print(X)
pca=PCA(svd_solver='auto', whiten=False)#exploratory data analysis it is used to identifyy the variations and to bring strong data patterns
pca.fit(X)
var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=2)*100)
plt.ylabel('variance of variables')
plt.xlabel('Features')
plt.title('PCA Analysis')
plt.style.context('seaborn-whitegrid')
#plt.show(var)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape
pca = PCA()
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
classifier = RandomForestClassifier(max_depth=2, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print(y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test,y_pred))
print('Accuracy of  classifier on training set: {:.2f}'.format(classifier.score(X_train, y_train)))
print('Accuracy of classifier on testing set: {:.2f}'.format(classifier.score(X_test, y_test)))
Y_pred = cross_val_predict(classifier, X, y)
for i in range(1036121):
    print(X_test[i], Y_pred[i])
quality_of_prediction=matthews_corrcoef(Y_pred.astype(int), y.astype(int))
print(quality_of_prediction)#to print the quality of predictions
logReg = LogisticRegression()
logReg.fit(X_train, y_train)
pipe = Pipeline([('pca', pca), ('logistic', logReg)])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test,y_pred))
print('Accuracy of logistic regression on training set: {:.2f}'.format(logReg.score(X_train, y_train)))
print('Accuracy of logistic regression on testing set: {:.2f}'.format(logReg.score(X_test, y_test)))
gnb=GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(classification_report(y_test,y_pred))
print(cm)
print('Accuracy of GNB classifier on training set: {:.2f}'.format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on testing set: {:.2f}'.format(gnb.score(X_test, y_test)))




