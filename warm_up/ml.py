import numpy as np # linear algebra
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, RFE, RFECV
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.ensemble import RandomForestClassifier

np.random.seed(239)

train = pd.read_csv("input/train.csv", index_col = 0)
train_data = train.values[:, :27]
train_target = train.values[:, -1]

exch = dict(zip(['A', 'B', 'C', 'D', 'E', 'F', 'G'], range(7)))
for i in range(0, train_data.shape[0]):
    for j in range(20, train_data.shape[1]):
        train_data[i, j] = exch[train_data[i, j]]

train_data, test_data, train_target, test_target = train_test_split(train_data, train_target, test_size = 0.2)

clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
clf.fit(train_data, train_target)
ans = clf.predict(test_data)
true = 0
for i in range(len(ans)):
    if (ans[i] == test_target[i]):
        true += 1
print("RandomForest", true / len(ans) * 100)

clf = SVC()
svc = SVC(kernel="linear")
slc =  RFECV(estimator = svc, cv=10).fit(train_data, train_target)
clf.fit(slc.transform(train_data), train_target)
ans = clf.predict(slc.transform(test_data))
true = 0
for i in range(len(ans)):
    if (ans[i] == test_target[i]):
        true += 1
print("RFECV + SVC", true / len(ans) * 100)

clf = SVC()
svc = SVC(kernel="linear")
best = 0
for i in range (1, 27):
    slc = SelectKBest(k = i)
    clf = SVC()
    slc.fit(train_data, train_target)
    clf.fit(slc.transform(train_data), train_target)
    ans = clf.predict(slc.transform(test_data))
    true = 0;
    for i in range(len(ans)):
        if (ans[i] == test_target[i]):
            true += 1
    best = max(best, true / len(ans) * 100)

print("SelectKBest + SVC", best)

best = 0
for i in range (1, 27):
    slc = SelectKBest(k = i)
    for j in range (1, 20):
        clf = KNeighborsClassifier(n_neighbors=j)
        slc.fit(train_data, train_target)
        clf.fit(slc.transform(train_data), train_target)
        ans = clf.predict(slc.transform(test_data))
        true = 0;
        for i in range(len(ans)):
            if (ans[i] == test_target[i]):
                true += 1
        best = max(best, true / len(ans) * 100)

print("SelectKBest + KNN", best)