from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
from sklearn import model_selection
#from scipy.interpolate import spline

######################################################################
# This python code is inspired from:

# https://www.datacamp.com/community/tutorials/random-forests-classifier-python
# and
# https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
######################################################################

data = pd.read_csv('Data/features_30_sec.csv')
print(data.head())


X=data.drop(['filename', 'label'],axis=1).values
scaler = StandardScaler()
X = scaler.fit_transform(X)
y=data['label'].values

# Fitting Random Forest Classification to the Training set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#n_estimators: This is the number of trees in the random forest classification
#criterion: This is the loss function used to measure the quality of the split


a= []
b= []
for i in range(20, 260, 20):
    classifier = RandomForestClassifier(n_estimators = i, criterion = 'entropy', random_state = 42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    a.append(accuracy_score(y_test, y_pred))
    b.append(i)


print(a)
print(b)

plt.scatter(b,a)
plt.title('Evolution of the accuracy', fontsize=20)
plt.xlabel('Numbers of Trees', fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.savefig("graphs/RF_nbrtree.png")


#best model

#split test and train data
kf = model_selection.KFold(n_splits=5, shuffle=True)
accuracies=[]
for train_index, test_index in kf.split(X):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    classifier = RandomForestClassifier(n_estimators=140, criterion='entropy', random_state=42)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
    accuracies.append(accuracy)

print("average accuracy with KFOLD = 5 :", sum(accuracies)/len(accuracies))

# Confusion Matrix
confusion_matr = confusion_matrix(y_test, y_pred) #normalize = 'true'
plt.figure(figsize = (16, 9))
ax= sns.heatmap(confusion_matr, cmap="Blues", annot=True, fmt='d',
            xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
           yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
                annot_kws={"size": 20});
ax.set(xlabel="Actual class",
       ylabel="Predicted class")
sns.set(font_scale= 1.4)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 18, rotation=45)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 18, rotation=45)
plt.savefig("RF_CF.png")