import numpy as np
import shap

data_bkg = np.array([[1,2,3],[10,20,30]])
x = np.array([[100,200,300],])

maskers = shap.maskers.Independent(data_bkg, max_samples=100, fixed_background=[False,False,True])
explainer = shap.Explainer(lambda x: x.sum(axis=-1), maskers,algorithm='permutation')
shap_values = explainer(x)
print(1)