import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import export_graphviz
import random
from sklearn import model_selection
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

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

#split test and train data
kf = model_selection.KFold(n_splits=5, shuffle=True)
accuracies=[]
for train_index, test_index in kf.split(X):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = DecisionTreeClassifier(criterion='entropy', max_depth=10)
    clf.fit(X_train,y_train)
    y_predicted = clf.predict(X_test)
    accuracy=accuracy_score(y_true=y_test, y_pred=y_predicted)
    accuracies.append(accuracy)

# confusion matrix + accuracy
print(confusion_matrix(y_true=y_test, y_pred=y_predicted))
print("average accuracy with KFOLD = 5 :", sum(accuracies)/len(accuracies))
#plot the tree
tree.plot_tree(clf)
plt.savefig("DT_plot.png")

# Confusion Matrix
confusion_matr = confusion_matrix(y_test, y_predicted) #normalize = 'true'
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

plt.savefig("DT_CF.png")



#plot
#export_graphviz(tree,out_file=None,feature_names = X.columns)
