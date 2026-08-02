# ML Recruitment Task: Predicting Serum Cholesterol from Clinical Data

## Overview

This project was completed as part of the ML Domain recruitment process for the AI/ML Club (ADG). The task was to take the UCI Heart Disease dataset (Cleveland subset) — a dataset built for classifying presence of heart disease — and repurpose it for a harder, less conventional regression problem: **predicting a patient's serum cholesterol (`chol`)** from their other clinical measurements.

The goal of this submission is not to show off a high R² score. It's to demonstrate:
- genuine curiosity during exploratory data analysis,
- clearly justified data-cleaning decisions,
- clinically grounded (not arbitrary) feature engineering, and
- **honest interpretation of results**, including when the model underperforms a trivial baseline.

## Dataset

- **Source:** UCI Heart Disease Dataset, Cleveland subset
- **Size:** 303 patients, 14 clinical attributes
- **Target variable:** `chol` (serum cholesterol, mg/dl) — used as a regression target, not the dataset's original classification label
- **Features:** age, sex, chest pain type (`cp`), resting blood pressure (`trestbps`), fasting blood sugar (`fbs`), resting ECG results (`restecg`), max heart rate achieved (`thalach`), exercise-induced angina (`exang`), ST depression (`oldpeak`), slope of peak exercise ST segment (`slope`), number of major vessels (`ca`), and thalassemia status (`thal`)
- **Known data quality issue:** missing values in `ca` and `thal` are encoded as the string `'?'` rather than a proper null

## Methodology

The notebook is organized into four phases:

1. **Understanding the Data** — full exploratory data analysis: shape/type inspection, missing-value detection (`'?'` entries in `ca` and `thal`), clinical plausibility checks (e.g. `chol == 0` and `trestbps == 0` flagged as impossible), distribution plots, correlation analysis, and a written summary of initial hypotheses.
2. **Data Cleaning** — every decision is justified in the notebook: median imputation for `ca` (few missing, small-range ordinal count), mode imputation for `thal` (nominal, few missing), correction of physiologically impossible zero-values in `chol`/`trestbps`, duplicate checks, and one-hot / binary encoding of categorical variables (`drop_first=True` to avoid the dummy-variable trap).
3. **Feature Engineering** — a clinically motivated **Cardiovascular Load Index / Heart Rate Reserve** feature (`(220 - age) - thalach`) and a bonus **Hypertension + ST Depression risk flag** (`trestbps > 140` and `oldpeak > 1.0`), each explained and evaluated against the target rather than added arbitrarily.
4. **Linear Regression & Honest Interpretation** — an 80/20 train-test split, OLS linear regression, coefficient analysis, residual diagnostics (residuals-vs-fitted, Q-Q plot), VIF multicollinearity check, a simple age+sex baseline comparison, RFE feature selection, Ridge/Lasso regularization, and 5-fold cross-validation.

## Key Findings

The consistent, honest finding across this project is that **cholesterol is only weakly related to the clinical variables available in this dataset**, and the fully engineered model does not outperform a much simpler one:

- **Full linear model R² ≈ 0.03** on the held-out test set — the model explains almost none of the variance in cholesterol.
- **Baseline model (age + sex only) achieved R² ≈ 0.11**, meaningfully *outperforming* the full 19-feature model. Adding the extra clinical/cardiac-test features did not help, and likely added noise.
- **5-fold cross-validation produced a negative mean R² (≈ -0.13)** — on average across folds, the full model predicts *worse* than simply guessing the mean cholesterol value. This is a stronger and more useful signal than the single train/test split alone.
- **Infinite VIF (perfect multicollinearity)** was detected between `age`, `thalach`, and the engineered `hr_reserve` feature. This is expected and mechanical: `hr_reserve = (220 - age) - thalach` is an exact linear combination of the other two, so their individual coefficients are not uniquely identifiable. This was flagged rather than hidden.
- The largest-magnitude raw coefficient (`restecg_1`) comes from a subgroup of only **4 patients out of 303** — almost certainly a small-sample artifact rather than a real clinical effect.
- **Conclusion:** serum cholesterol is driven substantially by factors this dataset does not measure — diet, genetics/family history, physical activity, and lipid-lowering medication (e.g. statins). No amount of feature engineering on the variables that *are* present in this dataset can substitute for that missing information. The honest takeaway is that this is a case where a low R² reflects a genuine limitation of the data, not a flaw in the modeling approach.

## Technologies Used

- Python 3
- pandas, numpy
- matplotlib, seaborn, missingno
- scikit-learn
- statsmodels
- scipy

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Soumya0204-star/<repo-name>.git
   cd <repo-name>
   ```
2. (Recommended) create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook Soumya_D_ML_task.ipynb
   ```
5. Run all cells top to bottom (`Kernel > Restart & Run All`) to reproduce every output.

## Conclusions

This project set out to predict serum cholesterol from routine cardiac clinical data, and the honest result is that it largely **cannot** be predicted well from these features alone. Every phase of the analysis — EDA, cleaning, feature engineering, and modeling — pointed toward the same conclusion: the available variables carry only a weak, unstable signal about cholesterol, weaker even than a two-variable demographic baseline. Rather than force an inflated narrative around a low R², this submission treats that outcome as the actual finding worth reporting.

## Limitations

- **Sample size:** 303 patients is small, especially once split into train/test and further subdivided by one-hot encoded categories (e.g. only 4 patients with `restecg == 1`).
- **Missing predictors:** diet, genetics/family history, physical activity, alcohol use, and medication (especially statins) are not recorded, despite being major known drivers of cholesterol.
- **Data quality:** a small number of missing (`'?'`) and physiologically impossible (`chol == 0`, `trestbps == 0`) values had to be imputed or corrected, adding minor noise.
- **Model assumptions:** residual diagnostics show mild departures from normality, and the engineered `hr_reserve` feature introduces exact multicollinearity with `age` and `thalach` that should be resolved (e.g. by dropping one of the three) before trusting any of their individual coefficients.
- **Generalizability:** cross-validation results (negative mean R²) suggest the full model does not generalize reliably beyond this specific train/test split.

## Acknowledgments

- Dataset: UCI Machine Learning Repository — Heart Disease Dataset (Cleveland subset)
- Built as part of the ML Domain Recruitment Task for the AI/ML Club

## Project Structure

```
.
├── Soumya_D_ML_task.ipynb   # Fully executed Jupyter notebook (all 4 phases)
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── .gitignore                # Files/directories excluded from version control
```
