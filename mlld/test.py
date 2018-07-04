print(__doc__)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from utils.data_reader import DataReader as dr

# Build a forest and compute the feature importances
forest = RandomForestClassifier(n_estimators = 17, max_depth = 5, criterion = 'entropy', random_state = 0, class_weight = 'balanced')
forest = AdaBoostClassifier(n_estimators = 4000, learning_rate = 1, algorithm = 'SAMME.R')
forest = AdaBoostClassifier(n_estimators = 50, learning_rate = 1, algorithm = 'SAMME.R')

data = dr('corpus_scores\\v2_5_raw_inv.txt').read_data(precision = -1)
features = data[:, 2:-1]
labels = data[:, -1].astype(int)

forest.fit(features, labels)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(features.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
fig = plt.figure()
plt.title("Feature importances")
plt.bar(range(features.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
plt.xticks(range(features.shape[1]), indices)
plt.xlim([-1, features.shape[1]])
fig.savefig('test.png')