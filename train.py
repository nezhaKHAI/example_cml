from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import os
import numpy as np

# Read in data
X_train = np.genfromtxt("data/train_features.csv")
y_train = np.genfromtxt("data/train_labels.csv")
X_test = np.genfromtxt("data/test_features.csv")
y_test = np.genfromtxt("data/test_labels.csv")

# Fit a model
depth = 5
clf = RandomForestClassifier(max_depth=depth)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print(acc)
#out_file = open("metrics.json","w")
#json.dump("Accuracy: " + str(acc) + "\n", out_file, indent=6)
with open("metrics.txt", "w") as outfile:
    outfile.write("Accuracy: " + str(acc) + "\n")

# Plot it
#predictions = clf.predict(X_test)
#cm = confusion_matrix(y_test, predictions, labels=clf.classes_)
#disp = ConfusionMatrixDisplay(
#    confusion_matrix=cm, display_labels=clf.classes_)
disp = plot_confusion_matrix(clf, X_test, y_test, normalize="true")
disp.plot()
plt.show
plt.savefig("plot.png")
