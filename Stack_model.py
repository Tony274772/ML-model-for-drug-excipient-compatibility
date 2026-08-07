#%% Import data
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('mol2vec_data.csv')
df.columns = ['Pub_CID' + str(i+1) if i < 100 else col for i, col in enumerate(df.columns)]
df.columns = ['Pub_Excipient' + str(i+1) if 99 < i < 200 else col for i, col in enumerate(df.columns)]
print(f"Loaded data shape: {df.shape}")
print(f"Outcome1 distribution:\n{df['Outcome1'].value_counts()}")

#%% Define X and y
# Fix #6: Select label by name (this file already did it correctly)
y = df['Outcome1'].values
X = df.drop(columns=["Outcome1", "API_CID", "Excipient_CID"], axis=1)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

#%% Training set, validation set and test set
# Fix #16: Added random_state for reproducibility
# Fix #17: Added stratify=y
from sklearn.model_selection import train_test_split
X_train, X_remain, y_train, y_remain = train_test_split(
    X, y, test_size=.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_remain, y_remain, test_size=.5, random_state=42, stratify=y_remain
)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

#%% Handle imbalanced data
# Fix #16: Added random_state to SVMSMOTE
from imblearn.over_sampling import SVMSMOTE
svmsmote = SVMSMOTE(random_state=42)
X_train_resampled, y_train_resampled = svmsmote.fit_resample(X_train, y_train)
print(f"Post-SMOTE: {X_train_resampled.shape}, class dist: {np.bincount(y_train_resampled)}")

#%% Random forest and XGBoost
# Fix #16: Added random_state to all estimators
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier

model1 = AdaBoostClassifier(learning_rate=0.7, n_estimators=700, random_state=42, algorithm='SAMME')
model2 = RandomForestClassifier(n_estimators=100, random_state=42)
model3 = XGBClassifier(max_depth=15, learning_rate=0.9, n_estimators=200,
                       random_state=42, use_label_encoder=False, eval_metric='logloss')

#%% Stack model
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=10, random_state=42, max_iter=1000)
stack_model = StackingClassifier(
    estimators=[('ada', model1), ('rf', model2), ('xgb', model3)],
    final_estimator=lr
)

#%% Stack model - Fit data
t0 = time.time()
class_weights = {0: 0.3, 1: 0.7}
stack_model.fit(X_train_resampled, y_train_resampled,
                sample_weight=[class_weights[i] for i in y_train_resampled])
print(f"Stacking model trained in {time.time()-t0:.1f}s")

#%% Stack model - Evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

y_pred = stack_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
print("Accuracy:", accuracy)
print("F1-score:", f1)
print("Precision:", precision)
print("Recall: ", recall)

#%% Stack model - Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_pred, labels=stack_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=stack_model.classes_)
disp.plot()
plt.show()
