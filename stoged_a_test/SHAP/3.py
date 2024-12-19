# Census income classification with scikit-learn
# https://shap.readthedocs.io/en/latest/example_notebooks/tabular_examples/model_agnostic/Census%20income%20classification%20with%20scikit-learn.html

import sklearn
import numpy as np
import shap

X, y = shap.datasets.adult()
X["Occupation"] *= 1000  # to show the impact of feature scale on KNN predictions
X_display, y_display = shap.datasets.adult(display=True)
X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
    X, y, test_size=0.2, random_state=7
)

knn = sklearn.neighbors.KNeighborsClassifier()
knn.fit(X_train, y_train)

def f(x):
    return knn.predict_proba(x)[:, 1]


med = X_train.median().values.reshape((1, X_train.shape[1]))

explainer = shap.Explainer(f, med)
shap_values = explainer(X_valid.iloc[0:1000, :])