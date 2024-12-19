import shap
import numpy as np
import sklearn

from stoged_a_test._MyIndependent import MyIndependent
# MyIndependent = shap.maskers.Independent

X_adult,y_adult = shap.datasets.adult()
model_adult = sklearn.linear_model.LogisticRegression(max_iter=10000)
model_adult.fit(X_adult.values, y_adult)


# 未魔改
def model_adult_probaV1(x):
    x = [np.hstack(item) for item in x]
    return model_adult.predict_proba(x)[:,1]
tempV1 = X_adult.values

background_adultV1 = MyIndependent(tempV1[:10], max_samples=10)
explainerV1 = shap.Explainer(model_adult_probaV1, background_adultV1, algorithm='exact')
shap_values_adultV1 = explainerV1(tempV1[:5])

print(shap_values_adultV1.values)
print(f'shape:{str(shap_values_adultV1.values.shape):s}')
delta,base = shap_values_adultV1.values.sum(1), shap_values_adultV1.base_values
result = base + delta
predict = model_adult_probaV1(tempV1[:5])
print(f' {"base:":>8s}{str(base):s}\n {"delta:":>8s}{str(delta):s}\n')
print(f' {"predict:":>8s}{str(predict):s}\n {"result:":>8s}{str(result):s}')

# 魔改(联合)
def model_adult_probaV2(x):
    x = [np.hstack(item) for item in x]
    return model_adult.predict_proba(x)[:,1]

tempV2 = [[item[:3],item[3:9],item[9:]] for item in X_adult.values]
tempV2 = np.array(tempV2,dtype=object)

background_adultV2 = MyIndependent(tempV2[:10], max_samples=10)
explainerV2 = shap.Explainer(model_adult_probaV2, background_adultV2, algorithm='exact')
shap_values_adultV2 = explainerV2(tempV2[:5])

print(shap_values_adultV2.values)
print(f'shape:{str(shap_values_adultV2.values.shape):s}')
delta,base = shap_values_adultV2.values[:,:-1].sum(1), shap_values_adultV2.base_values
result = base + delta
predict = model_adult_probaV2(tempV2[:5])
print(f' {"base:":>8s}{str(base):s}\n {"delta:":>8s}{str(delta):s}\n')
print(f' {"predict:":>8s}{str(predict):s}\n {"result:":>8s}{str(result):s}')

# 魔改(联合+Fix)
def model_adult_probaV3(x):
    x = [np.hstack(item) for item in x]
    return model_adult.predict_proba(x)[:,1]

tempV3 = [[item[:3],item[3:9],item[9:]] for item in X_adult.values]
tempV3 = np.array(tempV3,dtype=object)

background_adultV3 = MyIndependent(tempV3[:10], max_samples=10, fixed_background=[False,False,True])
explainerV3 = shap.Explainer(model_adult_probaV3, background_adultV3, algorithm='permutation')
shap_values_adultV3 = explainerV3(tempV3[:5])

print(shap_values_adultV3.values)
print(f'shape:{str(shap_values_adultV3.values.shape):s}')
delta,base = shap_values_adultV3.values.sum(1), shap_values_adultV3.base_values
result = base + delta
predict = model_adult_probaV3(tempV3[:5])
print(f' {"base:":>8s}{str(base):s}\n {"delta:":>8s}{str(delta):s}\n')
print(f' {"predict:":>8s}{str(predict):s}\n {"result:":>8s}{str(result):s}')
print(1)
