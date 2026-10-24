
<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/joblib-4B8BBE?style=for-the-badge" alt="joblib">
  <img src="https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white" alt="Git">
</p>

# Breast Cancer Classification

A small, leakage-free classification pipeline on the Wisconsin Diagnostic Breast Cancer dataset. The goal is not a record accuracy (this dataset is easy) but a clean evaluation: a single untouched test set, preprocessing inside a pipeline, and several checks that the reported score is not an artifact.

## Dataset

- Source: `sklearn.datasets.load_breast_cancer` (Wisconsin Diagnostic Breast Cancer, UCI).
- 569 samples, 30 numeric features computed from digitized images of fine-needle aspirates of breast masses (mean, standard error and worst value of 10 cell-nucleus measurements).
- Target: malignant (0, 212 samples) or benign (1, 357 samples).
- The data ships with scikit-learn, so no download is needed.

## Method

1. Stratified 80/20 split: 455 training and 114 test samples. The test set is used once, at the end.
2. Two candidate models, each wrapped in a `Pipeline` so that preprocessing is fit on training folds only:
   - Logistic regression: median imputation, standard scaling, `C` in {0.01, 0.1, 1, 10}.
   - Random forest: median imputation, `n_estimators` in {200, 400}, `max_depth` in {None, 5, 10}.
3. Hyperparameters are chosen with stratified 5-fold cross-validation on the training set, scored by ROC AUC.
4. The model with the best cross-validated AUC is evaluated on the test set.

The data has no missing values; the imputer is kept as a safeguard so the pipeline also works on data with gaps.

## Results

Cross-validation on the training set:

| Model | Best CV ROC AUC | Best parameters |
|---|---|---|
| Logistic regression | 0.9959 | `C = 1` |
| Random forest | 0.9906 | `n_estimators = 400`, `max_depth = None` |

Logistic regression was selected. Hold-out test results (114 samples):

| Metric | Value |
|---|---|
| ROC AUC | 0.9954 |
| Balanced accuracy | 0.9812 |
| Accuracy | 0.98 |

Confusion matrix (rows: true class, columns: predicted class):

|  | Pred. malignant | Pred. benign |
|---|---|---|
| **True malignant** | 41 | 1 |
| **True benign** | 1 | 71 |

Two of 114 test samples were misclassified.

## Validation checks

- **Label shuffling (200 permutations).** The model was refit on shuffled training labels and scored on the real test labels. Test AUC: mean 0.497, std 0.119, max 0.778. This is consistent with chance, so there is no sign of leakage. The spread is wide because the test set is small.
- **Train vs. test AUC.** 0.9975 vs. 0.9954. The gap is small, so there is no strong overfitting.
- **Repeated stratified 5-fold CV (10 repeats) on all data.** AUC 0.9948 ± 0.0053. This includes the test samples, so it only measures stability and is not the headline result.

## Limitations

- The test set has 114 samples, so each metric carries noticeable uncertainty. One more or fewer error changes the accuracy by almost one percentage point.
- The dataset is small, old and from a single source. The results say little about performance on other hospitals, scanners or patient populations.
- This is a research exercise, not a diagnostic tool.

## How to run

```bash
git clone https://github.com/Avin-Amiri/breast-cancer-classification.git
cd breast-cancer-classification
pip install -r requirements.txt
python breast_cancer_project.py
```

The script prints all results above and saves the fitted model to `best_model.joblib`, which is excluded from version control.

## Repository structure

```
breast-cancer-classification/
├── breast_cancer_project.py
├── requirements.txt
├── .gitignore
└── README.md
└── LICENSE

