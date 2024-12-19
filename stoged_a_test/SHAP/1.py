import shap
import numpy as np
import sklearn


X_adult,y_adult = shap.datasets.adult()
model_adult = sklearn.linear_model.LogisticRegression(max_iter=10000)
model_adult.fit(X_adult, y_adult)
def model_adult_proba(x):
    return model_adult.predict_proba(x)[:,1]

background_adult = shap.maskers.Independent(X_adult, max_samples=100)
explainer = shap.Explainer(model_adult_proba, background_adult)
shap_values_adult = explainer(X_adult[:10])

print(1)


