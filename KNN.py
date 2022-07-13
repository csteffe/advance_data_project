import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
import random
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(10)

data=pd.read_csv("Data/features_30_sec.csv")
data.head()
print(data.shape)

X=data.drop(['filename', 'label'],axis=1).values
y=data['label'].values

scaler=StandardScaler()

X=scaler.fit_transform(X)

print(X.shape)
print(y.shape)

a= []
b= []
for i in range(1, 50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    y_pred = knn.predict(X_test)
    a.append(accuracy_score(y_test, y_pred))
    b.append(i)

plt.scatter(b,a)
plt.title('Evolution of the accuracy', fontsize=20)
plt.xlabel('Numbers of K', fontsize=12)
plt.ylabel('Accuracy',fontsize=12)
plt.savefig("KNN_nbrk.png")


#split test and train data
kf = model_selection.KFold(n_splits=5, shuffle=True)
accuracies=[]
for train_index, test_index in kf.split(X):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    y_predicted = knn.predict(X_test)
    accuracy=accuracy_score(y_true=y_test, y_pred=y_predicted)
    accuracies.append(accuracy)

# confusion matrix + accuracy
print(confusion_matrix(y_true=y_test, y_pred=y_predicted))
print("average accuracy with KFOLD = 5 :", sum(accuracies)/len(accuracies))

# Confusion Matrix
confusion_matr = confusion_matrix(y_test, y_predicted) #normalize = 'true'
plt.figure(figsize = (16, 9))
ax= sns.heatmap(confusion_matr, cmap="Blues", annot=True, fmt='d',
            xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
           yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
                annot_kws={"size": 20});
ax.set(xlabel="Actual class",
       ylabel="Predicted class")
sns.set(font_scale= 18)
ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 18, rotation=45)
ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 18, rotation=45)
plt.savefig("knn_CF.png")

#plot
#export_graphviz(tree,out_file=None,feature_names = X.columns)
