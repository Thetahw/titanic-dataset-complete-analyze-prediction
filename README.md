# Titanic Survival Prediction

End-to-end analysis of the [Kaggle Titanic competition](https://www.kaggle.com/c/titanic)
dataset — exploring which passenger characteristics were most associated with
survival, and building a clean, leak-free feature engineering pipeline as a baseline
for model training.

## Status

- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature Engineering & Selection
- ⏳ **Model Training & Tuning — in progress, not yet in this notebook**
- ⏳ Model Interpretation (SHAP) — pending, depends on the above

This README currently documents everything through Feature Engineering & Selection.
It will be extended once the modeling section is added.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Key EDA Findings](#key-eda-findings)
- [Feature Engineering Summary](#feature-engineering-summary)
- [Tech Stack](#tech-stack)
- [How to Run](#how-to-run)
- [Next Steps](#next-steps)

## Overview

The goal of this project is to predict whether a passenger survived the Titanic
disaster based on attributes like class, sex, age, fare, and family relationships.
The notebook is organized as a full analytical pipeline:

1. **EDA** — understand distributions, missing data, and how individual features
   relate to survival.
2. **Feature Engineering & Selection** — create new features from the raw columns,
   encode categoricals, impute missing values, and select a final, non-redundant
   feature set using permutation importance, VIF (multicollinearity) analysis, and
   sequential feature selection.
3. **Modeling** *(pending)* — train and compare multiple classifiers (Logistic
   Regression, KNN, SVM, Random Forest, XGBoost), tune hyperparameters, and evaluate
   with cross-validation.
4. **Model Interpretation** *(pending)* — use SHAP to understand what drives
   individual predictions and validate that the model's reasoning aligns with the
   patterns found in EDA.

## Dataset

Data comes from Kaggle's Titanic competition: `train.csv`, containing demographic and
travel information for 891 passengers, with the goal of predicting the binary
`Survived` outcome. Key columns: `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`,
`Ticket`, `Fare`, `Cabin`, `Embarked`.

## Project Structure

The analysis lives in a single Jupyter notebook, organized into clearly-labeled
sections (Importing → Reading The Data → EDA → Feature Engineering & Selection →
*Modeling, coming soon*). Each step includes inline commentary explaining not just
what the code does, but why that choice was made and what trade-offs it involves.

## Key EDA Findings

- **Sex** was the strongest single predictor of survival (~74% survival for women vs.
  ~19% for men).
- **Class and wealth** mattered a lot: 1st-class passengers survived at far higher
  rates than 3rd-class. Port of embarkation showed a similar pattern, but only
  because Cherbourg boarded a higher proportion of 1st-class passengers — not because
  of the port itself.
- **Family size** had a U-shaped relationship with survival: traveling completely
  alone was worse than traveling with 1–2 companions, while very large families fared
  worse again (though with less reliable estimates due to small sample sizes).
- Outlier checks on `Age` and `Fare` found extreme-but-valid values (e.g. high fares
  for 1st-class cabins) rather than data errors, so no records were removed.

## Feature Engineering Summary

Starting from 11 raw features, the pipeline:

- Encodes `Sex` numerically and extracts `Title` from `Name` (grouping rare titles,
  including ones that only appear in Kaggle's official test set, into a `Rare`
  bucket).
- Builds `FamilySize` (`SibSp + Parch`) and `TicketGroupSize` (passengers sharing a
  ticket number) as more informative alternatives to the raw counts.
- Splits into train/test **before** any fitting-based step (encoding, imputation,
  scaling), so nothing is fit on data the model shouldn't see yet.
- Imputes missing `Age` values with KNN, then derives `AgeCategory` from the
  *imputed* age so the two stay consistent.
- Selects a final feature set using three complementary methods — permutation
  importance, VIF (to catch multicollinearity, e.g. `FamilySize` being a perfect
  linear combination of `SibSp` + `Parch`), and sequential feature selection — landing
  on: **`Pclass`, `Sex`, `Age`, `FamilySize`, `TicketGroupSize`, `AgeCategory`**.

## Tech Stack

- Python, pandas, NumPy
- scikit-learn (preprocessing, model selection, feature selection, RandomForest)
- statsmodels (VIF)
- Matplotlib, Seaborn (visualization)
- XGBoost, SHAP, Optuna *(imported, to be used in the upcoming modeling section)*

## How to Run

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt   # pandas, numpy, scikit-learn, statsmodels, seaborn, matplotlib, xgboost, shap, optuna
jupyter notebook titanic_analysis.ipynb
```

Place Kaggle's `train.csv` in a `../data/` folder relative to the notebook (or update
the path in the "Reading The Data" section).

## Next Steps

- [ ] Train and compare Logistic Regression, KNN, SVM, Random Forest, and XGBoost
- [ ] Hyperparameter tuning with Optuna
- [ ] Cross-validated model evaluation (accuracy, precision, recall)
- [ ] SHAP-based interpretation of the final model
- [ ] Generate predictions on Kaggle's official `test.csv` and submit

---

*This README will be updated once the modeling section is complete.*