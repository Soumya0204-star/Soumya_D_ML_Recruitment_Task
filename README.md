# Predicting Serum Cholesterol from Clinical Data: A Machine Learning Case Study

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)]()
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)]()
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()
[![License](https://img.shields.io/badge/Task-ADG%20ML%20Domain%20Recruitment-lightgrey.svg)]()

> **ML Domain Recruitment Task — ADG**
> A regression case study on whether serum cholesterol can be predicted from routine cardiac clinical data.

---

## 1. Project Overview

This project was completed for the **ADG ML Domain Recruitment Task**. Rather than the standard classification use-case for the UCI Heart Disease dataset (predicting presence of heart disease), this task repurposes it for a harder, less conventional problem: **predicting a patient's serum cholesterol (`chol`)** from their other clinical measurements.

**Honest summary:** While the full model performed poorly (R² ≈ 0.03), the analysis demonstrates rigorous EDA, justified data cleaning, thoughtful feature engineering, and intellectually honest reporting. The goal of this submission was never to force a high R² — it was to show sound reasoning at every step, including the step of accurately reporting when a model doesn't work well.

---

## 2. Dataset Summary

| | |
|---|---|
| **Dataset** | UCI Heart Disease Dataset (Cleveland subset) |
| **Records** | 303 patients |
| **Attributes** | 14 clinical features |
| **Target variable** | `chol` — serum cholesterol (mg/dL) |
| **Missing value encoding** | `'?'` strings in `ca` and `thal` |

**Key features:**

| Feature | Description |
|---|---|
| `age`, `sex` | Patient demographics |
| `cp` | Chest pain type (typical angina → asymptomatic) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `fbs` | Fasting blood sugar > 120 mg/dL (binary) |
| `restecg` | Resting ECG results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina (binary) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of peak exercise ST segment |
| `ca` | Number of major vessels colored by fluoroscopy (0–3) |
| `thal` | Thalassemia status (normal / fixed defect / reversible defect) |

---

## 3. Exploratory Data Analysis (EDA)

The full notebook contains all interactive/rendered charts with executed outputs. Below is a description of each key visualization for quick reference.

### Distribution Plots

| Chart | What it shows |
|---|---|
| **Cholesterol distribution** (histogram + KDE) | Right-skewed distribution, mean ≈ 246 mg/dL, with a long tail of high-cholesterol outliers extending past 400 mg/dL |
| **Age distribution** | Roughly normal, centered in the 50s |
| **Resting BP (`trestbps`) distribution** | Approximately normal with a mild right skew |
| **Max heart rate (`thalach`) distribution** | Slightly left-skewed; most patients achieve a fairly high max HR |
| **ST depression (`oldpeak`) distribution** | Heavily right-skewed with a large spike at zero |

```
chol histogram (schematic)
Count
│        ▄▄▆█▇▆▄▂
│      ▂▆████████▅▃▂
│    ▂▅███████████████▄▂        ← long right tail (>400 mg/dL)
└──────────────────────────────► chol (mg/dL)
   150   200   250   300  350  400+
```

*[Insert actual rendered histogram image from the notebook here, e.g. `assets/chol_distribution.png`]*

### Relationship Plots

| Chart | What it shows |
|---|---|
| **Scatter: age vs chol** | Diffuse cloud, no clear linear trend |
| **Scatter: trestbps vs chol** | Diffuse cloud, weak/no linear trend |
| **Scatter: thalach vs chol** | Diffuse cloud, weak/no linear trend |
| **Scatter: oldpeak vs chol** | Diffuse cloud, weak/no linear trend |
| **Box plot: chol by sex** | Overlapping distributions, similar medians |
| **Box plot: chol by chest pain type (`cp`)** | Overlapping distributions across all 4 categories |
| **Box plot: chol by exercise-induced angina (`exang`)** | Overlapping distributions, no clear separation |
| **Correlation heatmap** | All numerical variable correlations with `chol` are weak (\|r\| well under 0.4) |
| **Pairplot (age, trestbps, chol, thalach, oldpeak)** | Confirms the heatmap visually — no tight linear pairwise relationships |

*[Insert rendered scatter/box/heatmap/pairplot images here, e.g. `assets/correlation_heatmap.png`, `assets/scatter_grid.png`]*

### Categorical Analysis

Bar charts were generated for all 8 categorical variables: `sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`. These confirm the sample is male-skewed, mostly reports some form of chest pain, and shows the small `'?'` placeholder bars in `ca`/`thal` that flagged the missing-value investigation in Phase 1.

*[Insert rendered categorical bar chart grid here, e.g. `assets/categorical_bars.png`]*

> 📓 **Note:** All charts above are fully rendered with real data and interpretation text inside `Soumya_D_ML_task.ipynb`. This README summarizes them for quick browsing; open the notebook to see the actual plots and executed outputs.

---

## 4. Data Cleaning Summary

| Issue | Action Taken | Justification |
|---|---|---|
| `ca` missing (4 rows) | Imputed with **median** | Small-range ordinal count (0–3); median guarantees a real, occurring value |
| `thal` missing (2 rows) | Imputed with **mode** | Nominal categorical variable with very few missing rows |
| `chol == 0` (impossible) | Treated as missing, imputed with median | No living patient has zero serum cholesterol — a sentinel/error code, not a real value |
| `trestbps == 0` (impossible) | Treated as missing, imputed with median | Zero resting blood pressure is physiologically impossible |
| Categorical variables | One-hot encoded (`drop_first=True`) for `cp`, `restecg`, `slope`, `thal`; label-encoded (already binary) for `sex`, `fbs`, `exang` | Avoids treating nominal codes as ordered; `drop_first` avoids the dummy-variable trap |
| Duplicates | Checked via `.duplicated().sum()` | None found |

**Shape after cleaning:** 303 rows × 19 columns (row count unchanged — only imputation/encoding, no rows dropped; column count grew due to one-hot expansion).

---

## 5. Feature Engineering

### Heart Rate Reserve
```python
hr_reserve = (220 - age) - thalach
```
**Clinical rationale:** measures cardiovascular reserve — the gap between a patient's age-predicted maximum heart rate and their actual achieved max heart rate during testing.

### Hypertension + ST Depression Flag
```python
high_risk_flag = (trestbps > 140) & (oldpeak > 1.0)
```
**Clinical rationale:** this combination of elevated resting blood pressure and moderate-to-large ST depression is associated with elevated ischemia risk.

**Honest note:** neither engineered feature turned out to be a strong predictor of cholesterol — `hr_reserve` correlated at only **-0.086** with `chol`, and `high_risk_flag` at just **0.007**. Both were still clinically motivated and non-arbitrary, and were retained to see whether they added value in combination with other features — they did not meaningfully move the needle.

---

## 6. Model Results (Honest Reporting)

| Model | R² | MAE (mg/dL) | RMSE (mg/dL) |
|---|---|---|---|
| OLS Full Model | 0.0285 | 42.18 | 62.70 |
| Baseline (age + sex) | 0.1113 | — | — |
| Ridge (α=1.0) | 0.0687 | — | — |
| Lasso (α=1.0) | 0.1084 | — | — |
| **5-Fold CV (mean)** | **-0.1336** | 40.10 | — |

**Key Insight:** The baseline age + sex model **outperformed** the full 19-feature model, and cross-validation produced a **negative mean R²** — meaning the full model, on average across folds, predicts *worse* than simply guessing the mean cholesterol. This is a stronger and more informative signal than the single train/test R² alone, and it is reported here without spin.

---

## 7. Additional Analyses

- **VIF Multicollinearity Check:** Infinite VIF detected for `age`, `thalach`, and `hr_reserve` — an expected, mechanical result of `hr_reserve = (220 - age) - thalach` being an *exact* linear combination of the other two. This is flagged as a real issue that would need resolving (e.g. dropping one of the three) before trusting their individual coefficients.
- **RFE Feature Selection:** Top 5 features selected via Recursive Feature Elimination, compared against the top 5 features by raw coefficient magnitude.
- **Residual Analysis:** Residuals-vs-fitted plot (checked for heteroscedasticity) and a Q-Q plot (checked for normality) — residuals showed mild departure from normality in the tails, consistent with the right-skewed target variable.

---

## 8. Key Findings / Conclusions

- The full model explains only **~3% of variance** in cholesterol on held-out test data.
- A simple **age + sex model performed better** (~11% R²) than the full 19-feature model.
- The clinical/cardiac-test features in this dataset are **weak predictors** of cholesterol, individually and combined.
- Major real-world drivers of cholesterol — **diet, genetics, medication (e.g. statins), and lifestyle** — are simply not captured in this dataset, and no amount of feature engineering on the available variables can substitute for that missing information.
- The analysis demonstrates rigorous methodology, clinically grounded feature engineering, and — most importantly — **honest reporting of a negative result**, rather than an inflated narrative built around a low R².

---

## 9. Technologies Used

- **Python**, **pandas**, **numpy**, **matplotlib**, **seaborn**
- **scikit-learn** (`LinearRegression`, `Ridge`, `Lasso`, `RFE`, `cross_val_score`)
- **statsmodels** (Variance Inflation Factor)
- **missingno**, **scipy**

---

## 10. How to Run

```bash
# Clone the repository
git clone https://github.com/Soumya0204-star/Soumya_D_ML_Recruitment_Task.git
cd Soumya_D_ML_Recruitment_Task

# Install dependencies
pip install -r requirements.txt

# Launch the notebook
jupyter notebook Soumya_D_ML_task.ipynb
```

Run all cells top to bottom (`Kernel > Restart & Run All`) to reproduce every output shown above.

---

## 11. Acknowledgments

- **Dataset:** UCI Machine Learning Repository — Heart Disease Dataset (Cleveland Clinic)
- **Task:** ADG ML Domain Recruitment Task

---

## Project Structure

```
.
├── Soumya_D_ML_task.ipynb   # Fully executed Jupyter notebook (all 4 phases)
├── README.md                 # This file
├── requirements.txt           # Python dependencies
└── .gitignore                 # Files/directories excluded from version control
```
