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
test = pd.read_csv("input/test.csv", index_col = 0)
test_data = test.values[:, :27]
train_data = train.values[:, :27]
train_target = train.values[:, -1]

exch = dict(zip(['A', 'B', 'C', 'D', 'E', 'F', 'G'], range(7)))
for i in range(0, train_data.shape[0]):
    for j in range(20, train_data.shape[1]):
        train_data[i, j] = exch[train_data[i, j]] 

for i in range(0, test_data.shape[0]):
    for j in range(20, test_data.shape[1]):
        test_data[i, j] = exch[test_data[i, j]] 


f = open("log.csv", "w")
print("id,class", file = f)

best = 0
slc = SelectKBest(k = 10)
clf = KNeighborsClassifier(n_neighbors=j)
slc.fit(train_data, train_target)
clf.fit(slc.transform(train_data), train_target)
ans = clf.predict(slc.transform(test_data))
true = 0;
for q in range(len(ans)):
    print("%d,%s" % (2 * q + 1, ans[q]), file = f)
