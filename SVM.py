
import os
import pandas as pd
#import numpy as np
#import matplotlib.pyplot as plt
#import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
import lightgbm as lgbm
from sklearn.svm import SVC
import seaborn as sns

os.getcwd()

######################################################################
# This python code is inspired from:

# https://github.com/chittalpatel/Music-Genre-Classification-GTZAN
######################################################################


data = pd.read_csv("Data/features_30_sec.csv")
print(data.head())
data.head


X=data.drop(['filename', 'label'],axis=1).values
y=data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,test_size=0.2,random_state=42)
print(X_train.shape)
print(y_train.shape)

params = {
    "cls__C": [0.5, 1, 5, 10],
    "cls__kernel": ['rbf', 'sigmoid','poly'],
}

pipe_svm = Pipeline([
    ('scale', StandardScaler()),
    ('var_tresh', VarianceThreshold(threshold=0.1)),
    ('feature_selection', SelectFromModel(lgbm.LGBMClassifier())),
    ('cls', SVC())
])

grid_svm = GridSearchCV(pipe_svm, params, scoring='accuracy', n_jobs=-1, cv=5,verbose=2)
grid_svm.fit(X_train, y_train)
print(grid_svm)


preds = grid_svm.predict(X_test)
print("Best score on validation set (accuracy) = {:.4f}".format(grid_svm.best_score_))
print("Best score on test set (accuracy) = {:.4f}".format(accuracy_score(y_test, preds)))

# Confusion Matrix
confusion_matr = confusion_matrix(y_test, preds) #normalize = 'true'
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
plt.savefig("SVM_CF.png")
