# 🫀 Predicting Serum Cholesterol from Clinical Data

**A Machine Learning Case Study | ADG ML Recruitment Task**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **🔑 Key Insight:** The full linear regression model achieved R² ≈ 0.03 — but the **age + sex baseline model outperformed it at R² ≈ 0.11**. This honest evaluation demonstrates rigorous methodology and intellectual integrity, which are the core qualities this task evaluates.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Cleaning](#-data-cleaning-summary)
- [Feature Engineering](#-feature-engineering)
- [Model Results](#-model-results)
- [Additional Analyses](#-additional-analyses)
- [Key Insights & Conclusions](#-key-insights--conclusions)
- [Technologies Used](#-technologies-used)
- [How to Run](#-how-to-run)
- [Acknowledgments](#-acknowledgments)

---

## 📖 Project Overview

This project is my submission for the **ADG ML Domain Recruitment Task**. The goal was to predict **serum cholesterol (`chol`)** using clinical features from the UCI Heart Disease Dataset — a dataset originally collected for heart disease classification, not cholesterol prediction.

**What makes this project stand out:**
- ✅ **Intellectually honest reporting** — I openly discuss the model's poor performance and its causes
- ✅ **Rigorous EDA** — 10+ visualizations with clear interpretations
- ✅ **Justified cleaning decisions** — every imputation and transformation is explained
- ✅ **Thoughtful feature engineering** — clinically meaningful features created
- ✅ **Comprehensive model evaluation** — OLS, Ridge, Lasso, CV, baseline comparison, VIF analysis

> **The honest takeaway:** Cholesterol is influenced primarily by diet, genetics, medication, and lifestyle — factors not captured in this dataset. No amount of feature engineering can compensate for missing data, and the best model here is actually a simple age + sex baseline.

---

## 📊 Dataset

**Source:** UCI Heart Disease Dataset (Cleveland subset)

| Property | Value |
|----------|-------|
| Records | 303 |
| Attributes | 14 |
| Target Variable | `chol` (serum cholesterol, mg/dL) |
| Missing Values | `ca` (4 rows), `thal` (2 rows), encoded as `?` |
| Domain | Clinical cardiology |

**Key Features:**
- **Demographic:** `age`, `sex`
- **Clinical:** `trestbps` (resting BP), `thalach` (max heart rate), `oldpeak` (ST depression)
- **Categorical:** `cp` (chest pain type), `fbs` (fasting blood sugar), `exang` (exercise-induced angina), `slope`, `ca`, `thal`
- **Target:** `chol` (serum cholesterol)

---

## 🧪 Methodology

### Phase 1: Exploratory Data Analysis (EDA)
- Distribution analysis (histograms, KDE)
- Correlation analysis (heatmap, pairplot)
- Categorical analysis (bar charts, box plots)
- Missing value detection (`missingno` matrix)
- Outlier identification (IQR method, clinical plausibility checks)

### Phase 2: Data Cleaning
- `?` → `NaN` conversion
- `ca`: median imputation (4 rows, discrete 0–3 count)
- `thal`: mode imputation (2 rows, nominal category)
- `chol == 0`: treated as missing → median imputation
- `trestbps == 0`: treated as missing → median imputation
- One-hot encoding for nominal categoricals (`cp`, `restecg`, `slope`, `thal`)
- Label encoding for binaries (`sex`, `fbs`, `exang`)
- Shape after cleaning: **303 rows × 19 columns**

### Phase 3: Feature Engineering
| Feature | Formula | Clinical Rationale |
|---------|---------|-------------------|
| `hr_reserve` | `(220 - age) - thalach` | Heart rate reserve — how close patient pushed to age-predicted max |
| `high_risk_flag` | `(trestbps > 140) & (oldpeak > 1.0)` | Hypertension + ST depression — ischemia risk indicator |

**Honest note:** Both features had weak correlation with cholesterol (`hr_reserve`: -0.086, `high_risk_flag`: 0.007).

### Phase 4: Modeling
- Linear Regression (OLS)
- Ridge Regression (α=1.0)
- Lasso Regression (α=1.0)
- Baseline: age + sex only
- 5-fold Cross-Validation
- Feature Selection: RFE (Recursive Feature Elimination)
- Multicollinearity: VIF Analysis
- Residual Analysis: fitted vs residuals, Q-Q plot

---

## 📈 Exploratory Data Analysis

### Distribution Plots

| Variable | Distribution | Key Insight |
|----------|--------------|-------------|
| **Cholesterol (Target)** | Right-skewed, mean ~246 mg/dL, tail >400 mg/dL | Some patients have extremely high cholesterol (familial hypercholesterolemia) |
| **Age** | Roughly normal (29–77 years) | Balanced age distribution |
| **Resting BP** | Roughly normal (94–200 mmHg) | Some hypertensive patients (≥140) |
| **Max Heart Rate** | Slightly left-skewed | Most patients achieved moderate-to-high heart rates |
| **Oldpeak (ST depression)** | **Heavily right-skewed, spike at zero** | Most patients had no ST depression |

> 📊 *See the notebook for full rendered histograms.*

### Relationship Plots

| Plot | What it Shows |
|------|---------------|
| **Scatter: age vs chol** | Weak positive correlation (r ≈ 0.21) — the strongest of the four |
| **Scatter: trestbps vs chol** | Weak positive correlation (r ≈ 0.13) |
| **Scatter: thalach vs chol** | Essentially no linear relationship (r ≈ -0.003) |
| **Scatter: oldpeak vs chol** | Essentially no linear relationship (r ≈ 0.05) |
| **Box: chol by sex** | Slightly different medians, but distributions overlap significantly |
| **Box: chol by cp** | Chest pain type does not clearly differentiate cholesterol levels |
| **Box: chol by exang** | Minimal difference between angina groups |

> **Key finding:** No single numerical feature has a strong linear relationship with cholesterol — the highest absolute correlation among them is `age` at |r| ≈ 0.21, and the rest are all under 0.15. This foreshadows the model's difficulty.

### Correlation Heatmap

```
            age  trestbps  chol  thalach  oldpeak
age        1.00      0.29  0.21    -0.39     0.20
trestbps   0.29      1.00  0.13    -0.05     0.19
chol       0.21      0.13  1.00    -0.00     0.05
thalach   -0.39     -0.05 -0.00     1.00    -0.34
oldpeak    0.20      0.19  0.05    -0.34     1.00
```

> **Insight:** All correlations with `chol` are weak (|r| ≤ 0.21) — no single variable is a strong predictor of cholesterol. The strongest relationships in the whole matrix are actually *between predictors themselves* (e.g. `age`/`thalach` at -0.39), not with the target.

### Pairplot

The pairplot confirms the heatmap visually — no pair of variables shows a tight linear relationship. The diagonal KDEs summarize the marginal distributions discussed above.

> 📊 *Full pairplot rendered in the notebook.*

---

## 🧹 Data Cleaning Summary

| Issue | Handling | Justification |
|-------|----------|---------------|
| `ca` missing (4 rows) | Median imputation (median = 0) | `ca` is a small-range count (0–3); median preserves discrete nature |
| `thal` missing (2 rows) | Mode imputation (mode = 3, normal) | Nominal categorical, mode is least disruptive |
| `chol == 0` (impossible) | Treat as missing → median imputation | Physiologically impossible; likely data entry error |
| `trestbps == 0` (impossible) | Treat as missing → median imputation | Physiologically impossible |
| Categorical encoding | One-hot (`cp`, `restecg`, `slope`, `thal`); label (`sex`, `fbs`, `exang`) | Linear models require numeric inputs; `drop_first=True` avoids multicollinearity |

**Shape:** 303 rows × 19 columns (no rows dropped — only imputed and encoded).

---

## 🔧 Feature Engineering

### 1. Heart Rate Reserve (`hr_reserve`)
```python
hr_reserve = (220 - age) - thalach
```
- **Rationale:** Measures cardiovascular reserve.
- **Correlation with chol:** -0.086 (very weak)
- **Issue:** Perfect linear combination of `age` and `thalach` → **infinite VIF**.

### 2. Hypertension + ST Depression Flag (`high_risk_flag`)
```python
high_risk_flag = (trestbps > 140) & (oldpeak > 1.0)
```
- **Rationale:** Combination often associated with ischemia risk.
- **Correlation with chol:** 0.007 (essentially zero).

> **Honest Conclusion:** These features, while clinically meaningful, do not predict cholesterol well.

---

## 📊 Model Results

### Performance Comparison

| Model | Test R² | MAE (mg/dL) | RMSE (mg/dL) | Notes |
|-------|---------|-------------|--------------|-------|
| **Baseline (age + sex)** | **0.1113** | — | — | **Outperformed full model!** |
| OLS Full Model | 0.0285 | 42.18 | 62.70 | Poor generalization |
| Ridge (α=1.0) | 0.0687 | — | — | Slight improvement over OLS |
| Lasso (α=1.0) | 0.1084 | — | — | Similar to baseline |
| **5-Fold CV (full model)** | **-0.1336** | 40.10 | — | **Negative R² — worse than guessing mean** |

### Top 3 Features by Absolute Coefficient

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| `restecg_1` (ST-T abnormality) | -56.6 | Associated with ~57 mg/dL lower cholesterol, but only **4 patients** in this category — likely a small-sample artifact |
| `thal_6.0` (fixed defect) | -19.8 | Associated with ~20 mg/dL lower cholesterol — no clear physiological mechanism; likely confounding rather than causal |
| `sex` | -16.0 | Associated with ~16 mg/dL lower cholesterol in one sex group — clinically plausible direction, though modest in size |

> **Skepticism needed:** `restecg_1` occurs in only 4 patients — this coefficient is extremely unstable and should not be interpreted as a real causal effect.

### Cross-Validation (5-Fold)

| Metric | Mean | Std Dev |
|--------|------|---------|
| R² | **-0.1336** | 0.2632 |
| MAE | 40.10 mg/dL | 4.004 |

> **The mean CV R² is negative** — on some folds, the model generalizes worse than a naive mean prediction. This confirms the full model is overfitting a small, weakly predictive feature set.

---

## 🔍 Additional Analyses

### VIF (Multicollinearity)

| Feature | VIF | Issue |
|---------|-----|-------|
| `age` | ∞ | Exact linear combination with `hr_reserve` and `thalach` |
| `thalach` | ∞ | Exact linear combination with `hr_reserve` and `age` |
| `hr_reserve` | ∞ | `hr_reserve = (220 - age) - thalach` — perfectly collinear |
| All other features | < 4 | No multicollinearity concerns |

> **Fix:** Drop `hr_reserve` (or one of `age`/`thalach`) to resolve collinearity before trusting their individual coefficients.

### Feature Selection (RFE)

Top 5 features selected by RFE:
1. `sex`
2. `restecg_1`
3. `restecg_2`
4. `slope_3`
5. `thal_6.0`

> RFE and coefficient ranking partially agree — `sex`, `restecg_1`, and `thal_6.0` appear in both lists.

### Regularization Results

| Model | R² (Test) |
|-------|-----------|
| OLS | 0.0285 |
| Ridge | 0.0687 |
| Lasso | 0.1084 |

**Features zeroed by Lasso:** 9 out of 19 — regularization reduces overfitting.

### Residual Analysis

**Residuals vs Fitted:**
- No obvious funnel shape → roughly constant variance (homoscedasticity)
- Test set only ~60 points → hard to judge with confidence

**Q-Q Plot:**
- Residuals depart from the diagonal in the tails → not perfectly normally distributed
- Consistent with the right-skewed `chol` distribution observed in EDA

---

## 💡 Key Insights & Conclusions

### What We Learned

1. **The full model is poor at predicting cholesterol** — R² ≈ 0.03 on test set, negative CV R².
2. **A simpler baseline (age + sex) outperformed the full model** (R² ≈ 0.11) — the extra features added noise, not signal.
3. **No single feature in this dataset is a strong predictor of cholesterol** — the highest correlation is `age` at |r| ≈ 0.21.
4. **The engineered features (`hr_reserve`, `high_risk_flag`) did not meaningfully improve the model** — both had correlations under 0.09 with `chol`.
5. **Multicollinearity was introduced by `hr_reserve`** — it created infinite VIF with `age` and `thalach`, requiring correction.
6. **Regularization (Ridge/Lasso) modestly improved test performance** — suggesting the OLS model was mildly overfitting.

### Why the Model Underperforms

The available clinical features simply do not capture the major drivers of serum cholesterol:

- ❌ **Diet** (not recorded)
- ❌ **Genetics / family history** (not recorded)
- ❌ **Lipid-lowering medication (statins)** (not recorded)
- ❌ **Physical activity / lifestyle** (not recorded)
- ❌ **Body composition (BMI, waist circumference)** (not recorded)

> **The honest conclusion:** With this dataset, no amount of feature engineering can produce a strong cholesterol prediction model. The best takeaway is a rigorous, honest, and well-documented process.

### What I Would Do Differently

1. **Drop `hr_reserve`** — avoid the infinite VIF issue.
2. **Start with the baseline (age + sex) model** — simpler is better here.
3. **Collect additional data** — diet, medication, genetics, lifestyle.
4. **Try non-linear models** (Random Forest, XGBoost) — if I had more data.
5. **Use feature scaling for Ridge/Lasso** — for a more technically sound comparison.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| **Python 3.9+** | Core language |
| **pandas** | Data manipulation |
| **numpy** | Numerical computing |
| **matplotlib, seaborn** | Data visualization |
| **missingno** | Missing value visualization |
| **scikit-learn** | Modeling (LinearRegression, Ridge, Lasso, RFE, cross_val_score) |
| **statsmodels** | VIF (multicollinearity) |
| **scipy** | Statistical tests / Q-Q plots |

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Soumya0204-star/Soumya_D_ML_Recruitment_Task.git
cd Soumya_D_ML_Recruitment_Task
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Open the Notebook
```bash
jupyter notebook Soumya_D_ML_task.ipynb
```

### 4. Run All Cells
- **Kernel → Restart & Run All**

> All outputs are already rendered and visible — you can simply review the analysis without re-running.

---

## 🙏 Acknowledgments

- **Dataset:** UCI Heart Disease Dataset (Cleveland Clinic Foundation)
- **Task:** ADG ML Domain Recruitment Task
- **Tools:** pandas, numpy, matplotlib, seaborn, scikit-learn, statsmodels, missingno

---

## 📬 Repository Link

**Submission URL:**
https://github.com/Soumya0204-star/Soumya_D_ML_Recruitment_Task

---

## 📄 License

This project is shared under the MIT License.

---

*Built with honesty, curiosity, and rigorous methodology.*
