# Ames House Price Prediction

A machine-learning project focused on predicting residential sale prices using the **Ames Housing** dataset. The notebooks explore the structure of the housing data, investigate missing values, build preprocessing workflows, compare regression models, and generate predictions for the held-out test set.

## Project overview

The target variable is `SalePrice`, the sale price of each home. The dataset describes residential properties in Ames, Iowa, through structural, quality, location, and sale-related characteristics.

The repository includes the standard Ames training and test files:

- `train.csv` — 1,460 labelled houses, including `SalePrice`.
- `test.csv` — 1,459 houses without the target.
- `sample_submission.csv` — the expected submission format.
- `submission.csv` — a generated prediction file.
- `data_description.txt` — descriptions of the dataset variables and category codes.

## Repository contents

| File | Description |
|---|---|
| `main.ipynb` | Main analysis notebook: data inspection, missing-value investigation, preprocessing, regression experiments, cross-validation, feature significance analysis, and feature selection. |
| `dumpi.ipynb` | Additional experimentation notebook comparing regression models and producing test-set predictions. |
| `numerical_missing_value_handler.ipynb.ipynb` | Focused experiments with model-based imputation of missing numerical features. |
| `train.csv` | Labelled Ames Housing training data. |
| `test.csv` | Unlabelled test data. |
| `sample_submission.csv` | Example output structure. |
| `submission.csv` | Saved house-price predictions. |
| `data_description.txt` | Detailed feature descriptions from the Ames dataset. |

## Workflow

### 1. Data understanding and exploratory analysis

The notebooks inspect the data types, descriptive statistics, numerical and categorical columns, and the distribution of missing values. They also explore how property characteristics relate to sale price.

The feature analysis includes variables describing:

- Overall material and finish quality.
- Above-ground living area and basement area.
- Lot size and frontage.
- Construction and remodelling year.
- Garage capacity and condition.
- Kitchen, basement, and fireplace quality.
- Neighbourhood and sale conditions.

### 2. Missing-value investigation and imputation

Missing values are treated as a modelling problem rather than applying one blanket replacement rule.

The experiments include:

- Identifying columns with substantial missingness.
- Separating numerical and categorical features.
- Using LightGBM regression to estimate missing numerical values from other available features.
- Exploring probabilistic approaches to categorical missing values.
- Comparing the effects of preprocessing choices on downstream modelling.

For Ames data, a missing value can sometimes mean that a feature is absent (for example, no basement or no garage), rather than that a measurement was simply not recorded. The variable descriptions should therefore be consulted before deciding how to impute a field.

### 3. Preprocessing and feature selection

The main notebook builds a scikit-learn preprocessing workflow with:

- Median imputation and standardisation for numerical predictors.
- Most-frequent imputation and one-hot encoding for categorical predictors.
- A `ColumnTransformer` and pipelines to apply transformations consistently.

It also explores feature relevance using correlation-based checks, statistical significance testing, and a custom stepwise AIC feature-selection routine.

### 4. Regression models

The notebooks experiment with several regression approaches, including:

- K-Nearest Neighbours Regressor
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor
- Gaussian Naive Bayes as an exploratory comparison
- A neural-network regression experiment

The main notebook includes five-fold cross-validation for Random Forest, KNN, and Decision Tree models. The saved cross-validation results are:

| Model | Mean RMSE | Mean MAE | Mean R² |
|---|---:|---:|---:|
| Random Forest | 29,938.55 | 17,713.27 | 0.8566 |
| K-Nearest Neighbours | 38,204.77 | 22,151.31 | 0.7665 |
| Decision Tree | 38,994.44 | 25,639.27 | 0.7499 |

These figures are the recorded results from the notebook's five-fold evaluation. They are useful as a baseline comparison, not a guarantee of performance on unseen data. The separate experimentation notebook reports results from a different split and setup, so its scores should not be compared directly with the cross-validation table.

### 5. Prediction and submission

The notebooks prepare the test features using the selected preprocessing workflow, generate `SalePrice` predictions, and save a two-column submission file containing:

- `Id`
- `SalePrice`

The repository's `submission.csv` is an example of this generated output.

## Technology stack

- Python
- pandas and NumPy
- Matplotlib and Seaborn
- scikit-learn
- LightGBM
- XGBoost
- statsmodels
- SciPy
- Jupyter Notebook

## Running the project

1. Clone the repository:

   ```bash
   git clone https://github.com/MJwd-Hasan/House-Price-prediction.git
   cd House-Price-prediction
   ```

2. Install the main dependencies:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost statsmodels scipy jupyter
   ```

3. Open the notebooks:

   ```bash
   jupyter notebook
   ```

4. Run `main.ipynb` for the primary workflow or `dumpi.ipynb` for the additional experiments.

The notebooks currently contain local Windows file paths in some cells. Update those paths to the location of `train.csv`, `test.csv`, and the other dataset files before running them in a new environment.

## Notes and limitations

- The notebooks represent an evolving learning and experimentation workflow; not every model experiment uses the same split or preprocessing setup.
- The five-fold cross-validation table is the most suitable like-for-like model comparison currently recorded in the main notebook.
- Some early experiments use very small model settings or limited training epochs and should be treated as demonstrations rather than fully tuned final models.
- Missing-value handling should respect the meaning of each Ames feature; absence of a facility is not necessarily the same as an unknown value.
- The generated predictions are not a deployed house-valuation service.

## Objective

The goal is to understand which property characteristics are informative for sale-price prediction and to build a reproducible regression workflow that handles mixed data types, missing values, feature selection, and model evaluation.
