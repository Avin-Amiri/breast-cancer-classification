import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.base import clone

RANDOM = 42
data = load_breast_cancer()
X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM)

candidates = {
    'logreg': (
        Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(max_iter=5000)),
        ]),
        {'clf__C': [0.01, 0.1, 1, 10]},
    ),
    'random_forest': (
        Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('clf', RandomForestClassifier(random_state=RANDOM)),
        ]),
        {'clf__n_estimators': [200, 400], 'clf__max_depth': [None, 5, 10]},
    ),
}

best_name, best_search, best_score = None, None, -np.inf
print('== CV (train only)')
for name, (pipe, grid) in candidates.items():
    search = GridSearchCV(pipe, grid, cv=cv, scoring='roc_auc', n_jobs=-1)
    search.fit(X_train, y_train)
    print(f'{name:14s} best CV AUC = {search.best_score_:.4f}  params={search.best_params_}')
    if search.best_score_ > best_score:
        best_name, best_search, best_score = name, search, search.best_score_

model = best_search.best_estimator_
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f'\n== Final test ({best_name})')
print(f'ROC AUC           : {roc_auc_score(y_test, y_prob):.4f}')
print(f'Balanced accuracy : {balanced_accuracy_score(y_test, y_pred):.4f}')
print('Confusion matrix (rows=true 0/1, cols=pred 0/1):')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=data.target_names))

joblib.dump(model, 'best_model.joblib')
print('Saved: best_model.joblib')

aucs = []
for i in range(200):
    r = np.random.default_rng(i)
    m = clone(model).fit(X_train, r.permutation(y_train))
    aucs.append(roc_auc_score(y_test, m.predict_proba(X_test)[:, 1]))
aucs = np.array(aucs)
print('\n== Shuffled-label check x200 (باید میانگین حدود 0.5 باشه)')
print(f'Shuffled AUC: mean={aucs.mean():.3f}, std={aucs.std():.3f}, max={aucs.max():.3f}')

train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
test_auc = roc_auc_score(y_test, y_prob)
print('\n== Train vs test')
print(f'Train AUC: {train_auc:.4f} | Test AUC: {test_auc:.4f}')

rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM)
scores = cross_val_score(clone(model), X, y, cv=rcv, scoring='roc_auc')
print('\n== Repeated CV (5 fold x 10) on all data')
print(f'AUC mean={scores.mean():.4f}, std={scores.std():.4f}')
