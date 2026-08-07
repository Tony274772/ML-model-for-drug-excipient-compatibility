#%%
import time
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Fix #4: Missing closing quote on filename string
df = pd.read_csv('mol2vec_data.csv')
print(df)
print(f"Loaded mol2vec data: {df.shape}")

#%%
# Fix #6: Select label by column name, not position.
# Original: y = df.iloc[:,-1].values  — this grabs 'Excipient_Smiles', NOT 'Outcome1'
y = df['Outcome1'].values
X = df.iloc[:, 0:200].values
print(f"y shape: {y.shape}, X shape: {X.shape}")
print(f"Label distribution: {np.bincount(y)}")

#%% Training set, validation set and test set
# Fix #16: Added random_state for reproducibility
# Fix #17: Added stratify=y given the ~9-10% positive rate
from sklearn.model_selection import train_test_split
X_train, X_remain, y_train, y_remain = train_test_split(
    X, y, test_size=.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_remain, y_remain, test_size=.5, random_state=42, stratify=y_remain
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

#%%
# Fix #16: Added random_state to SVMSMOTE
from imblearn.over_sampling import SVMSMOTE
svmsmote = SVMSMOTE(random_state=42)
X_train_resampled, y_train_resampled = svmsmote.fit_resample(X_train, y_train)
print(f"Post-SMOTE: {X_train_resampled.shape}, class dist: {np.bincount(y_train_resampled)}")

#%%
# Random Forest
# Fix #16: Added random_state to estimator
from sklearn.ensemble import RandomForestClassifier

param_grid_RF = {
    'n_estimators': [50, 100, 150, 200, 300],
}

t0 = time.time()
grid_search_RF = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=param_grid_RF,
                              scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
grid_search_RF.fit(X_train_resampled, y_train_resampled)
y_pred_RF = grid_search_RF.predict(X_val)
print(f'[Random Forest] done in {time.time()-t0:.1f}s')
print('Best parameters: ', grid_search_RF.best_params_)
print('Accuracy: ', accuracy_score(y_val, y_pred_RF))
print(classification_report(y_val, y_pred_RF))

#%%
cm = confusion_matrix(y_val, y_pred_RF, labels=grid_search_RF.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_RF.classes_)
disp.plot()
plt.show()

#%%
# Decision Tree
# Fix #16: Added random_state
from sklearn.tree import DecisionTreeClassifier

param_grid_DT = {
    'max_depth': [5, 15, 25, 30, 40, 50],
}

t0 = time.time()
grid_search_DT = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid=param_grid_DT,
                              scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
grid_search_DT.fit(X_train_resampled, y_train_resampled)
y_pred_DT = grid_search_DT.predict(X_val)
print(f'[Decision Tree] done in {time.time()-t0:.1f}s')
print('Best parameters: ', grid_search_DT.best_params_)
print('Accuracy: ', accuracy_score(y_val, y_pred_DT))
print(classification_report(y_val, y_pred_DT))

#%%
cm = confusion_matrix(y_val, y_pred_DT, labels=grid_search_DT.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_DT.classes_)
disp.plot()
plt.show()

#%%
# Support Vector Machine
# Fix #16: Added random_state
from sklearn.svm import SVC

param_grid_SVC = {
    'gamma': [0.001, 0.01, 0.05, 0.1],
    'C': [1, 5, 10, 100],
}

t0 = time.time()
grid_search_SVC = GridSearchCV(SVC(random_state=42), param_grid=param_grid_SVC,
                               scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
grid_search_SVC.fit(X_train_resampled, y_train_resampled)
print(f'[SVM] done in {time.time()-t0:.1f}s')
print('Best parameters: ', grid_search_SVC.best_params_)
y_pred_SVC = grid_search_SVC.predict(X_val)
print('Accuracy: ', accuracy_score(y_val, y_pred_SVC))
print(classification_report(y_val, y_pred_SVC))

#%%
cm = confusion_matrix(y_val, y_pred_SVC, labels=grid_search_SVC.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_SVC.classes_)
disp.plot()
plt.show()

#%%
# Multi-layer Perceptron
# Fix #16: Added random_state
from sklearn.neural_network import MLPClassifier

param_grid_MLP = {
    'hidden_layer_sizes': [(100,), (300,), (500,)],
    'activation': ['relu'],
    'solver': ['adam'],
    'momentum': [0.1, 0.5, 0.9],
    'learning_rate': ['adaptive'],
}

t0 = time.time()
grid_search_MLP = GridSearchCV(MLPClassifier(random_state=42), param_grid=param_grid_MLP,
                               scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
grid_search_MLP.fit(X_train_resampled, y_train_resampled)
print(f'[MLP] done in {time.time()-t0:.1f}s')
print('Best parameters: ', grid_search_MLP.best_params_)
y_pred_MLP = grid_search_MLP.predict(X_val)
print('Accuracy: ', accuracy_score(y_val, y_pred_MLP))
print(classification_report(y_val, y_pred_MLP))

#%%
cm = confusion_matrix(y_val, y_pred_MLP, labels=grid_search_MLP.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_MLP.classes_)
disp.plot()
plt.show()

#%%
# k-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

param_grid_kNN = {
    'n_neighbors': np.arange(3, 16),
}

t0 = time.time()
grid_search_kNN = GridSearchCV(KNeighborsClassifier(), param_grid=param_grid_kNN,
                               scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
grid_search_kNN.fit(X_train_resampled, y_train_resampled)
print(f'[kNN] done in {time.time()-t0:.1f}s')
print('Best parameters: ', grid_search_kNN.best_params_)
y_pred_kNN = grid_search_kNN.predict(X_val)
print('Accuracy: ', accuracy_score(y_val, y_pred_kNN))
print(classification_report(y_val, y_pred_kNN))

#%%
cm = confusion_matrix(y_val, y_pred_kNN, labels=grid_search_kNN.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_kNN.classes_)
disp.plot()
plt.show()

#%%
# Logistic Regression
# Fix #16: Added random_state
from sklearn.linear_model import LogisticRegression

param_grid_LR = {
    'C': [0.01, 0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 0.8, 1, 10, 15],
}

t0 = time.time()
grid_search_LR = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid=param_grid_LR,
                               scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
grid_search_LR.fit(X_train_resampled, y_train_resampled)
print(f'[Logistic Regression] done in {time.time()-t0:.1f}s')
print('Best parameters: ', grid_search_LR.best_params_)
y_pred_LR = grid_search_LR.predict(X_val)
print('Accuracy: ', accuracy_score(y_val, y_pred_LR))
print(classification_report(y_val, y_pred_LR))

#%%
cm = confusion_matrix(y_val, y_pred_LR, labels=grid_search_LR.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid_search_LR.classes_)
disp.plot()
plt.show()

#%%
# XGBoost
# Fix #16: Added random_state
from xgboost import XGBClassifier

param_grid_XGB = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [3, 5, 7, 9, 15],
    'learning_rate': [0.3, 0.5, 0.7, 0.9],
}

t0 = time.time()
grid_search_XGB = GridSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
                               param_grid=param_grid_XGB,
                               scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
grid_search_XGB.fit(X_train_resampled, y_train_resampled)
print(f'[XGBoost] done in {time.time()-t0:.1f}s')
print('Best parameters: ', grid_search_XGB.best_params_)
y_pred_XGB = grid_search_XGB.predict(X_val)
print('Accuracy: ', accuracy_score(y_val, y_pred_XGB))
print(classification_report(y_val, y_pred_XGB))

#%%
cm = confusion_matrix(y_val, y_pred_XGB)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#%%
# AdaBoost
# Fix #16: Added random_state
from sklearn.ensemble import AdaBoostClassifier

param_grid_Ada = {
    'n_estimators': [300, 400, 500, 600, 700],
    'learning_rate': [0.5, 0.7, 0.9],
}

t0 = time.time()
grid_search_Ada = GridSearchCV(AdaBoostClassifier(random_state=42, algorithm='SAMME'),
                               param_grid=param_grid_Ada,
                               scoring='accuracy', cv=5, n_jobs=-1, verbose=2)
grid_search_Ada.fit(X_train_resampled, y_train_resampled)
print(f'[AdaBoost] done in {time.time()-t0:.1f}s')
y_pred_Ada = grid_search_Ada.predict(X_val)
print('Best parameters: ', grid_search_Ada.best_params_)
print('Accuracy: ', accuracy_score(y_val, y_pred_Ada))
print(classification_report(y_val, y_pred_Ada))

#%%
cm = confusion_matrix(y_val, y_pred_Ada)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()
