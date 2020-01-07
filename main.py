import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import preprocessing
import graphviz

le = preprocessing.LabelEncoder()
dwelling_data = pd.read_stata('data/Dwelling.dta')
sns.countplot(dwelling_data['h13_2'], label="Count")
print(dwelling_data.groupby('h13_2').size())

target = list(dwelling_data['h13_2'])
# plt.show()
# plt.savefig('h13_2 distribution')

feature_names = [
    'h1',
    'h2',
    'h3',
    'h4',
    'h23',
    'h21_1',
    'h13_2'
    # 'h21_2',
    # 'h21_3',
    # 'h21_4',
    # 'h21_5',
    # 'h21_6',
    # 'h21_7',
    # 'h21_8',
    # 'h21_9',
    # 'h21_10',
    # 'h21_11',
    # 'h21_12',
    # 'h21_13',
    # 'h21_15',
    # 'h21_16',
    # 'h21_17',
    # 'h21_18',
    # 'h21_19',
    # 'h21_20',
    # 'h21_21',
]

X_features = [
    'h1',
    'h2',
    'h3',
    'h4',
    'h23',
    'h21_1',
]
dataset = dwelling_data
for feature in feature_names:
    dataset[feature] = le.fit_transform(dwelling_data[feature].astype(str))

X = dataset[X_features]

# for feature in feature_names:
#     X[feature] = pd.Categorical(X[feature]).codes

y = dataset['h13_2']
from sklearn import tree

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Create Decision Tree classifer object and Train it
clf = DecisionTreeClassifier(criterion="entropy", max_depth=40).fit(X_train, y_train)
tree.plot_tree(clf)
# Predict the response for test dataset
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=X_features,
                                class_names=target,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("tree")