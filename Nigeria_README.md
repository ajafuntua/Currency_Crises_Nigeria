# 🇳🇬 Currency Crisis Early Warning Model — Nigerian Economy
### Random Forest Ensemble Classifier  |  1990–2023

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)](https://scikit-learn.org/)
[![Platform](https://img.shields.io/badge/Platform-Google%20Colab-orange)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Overview

This project builds a **machine learning early warning system (EWS)** to identify and predict currency crisis episodes in Nigeria using annual macroeconomic data from 1990–2023. The model frames the problem as a **binary classification task** — Currency Crisis (1) vs. No Crisis (0) — with Random Forest as the primary algorithm, complemented by SHAP explainability to ensure that predictions are transparent and actionable for policymakers.

The EWS is grounded in the **Exchange Market Pressure (EMP) index** methodology, which combines exchange rate depreciation and reserve loss into a single crisis signal. This provides a theoretically defensible, data-driven crisis label rather than relying on subjective episode dating.

---

## 🎯 Research Objective

> *Can macroeconomic, external sector, and oil sector indicators reliably predict currency crisis episodes in Nigeria, and what are the dominant early warning signals?*

The project addresses this across four dimensions:

1. **Crisis measurement** — Constructing the EMP index and deriving binary crisis labels
2. **Prediction** — Binary classification of annual currency crisis status
3. **Benchmarking** — Comparing Random Forest against four alternative classifiers
4. **Interpretation** — SHAP values to quantify each feature's contribution to crisis predictions

---

## 🔴 Crisis Definition

A year is labelled a **currency crisis** if any one of the following conditions holds:

| Condition | Rule |
|---|---|
| **EMP threshold** | EMP index > μ + 1.5σ (where μ and σ are the sample mean and standard deviation) |
| **Severe depreciation** | Annual exchange rate depreciation \|ΔE/E\| > 15% |
| **Reserve collapse** | Annual reserve change ΔR/R < −15% |

The EMP index is constructed as:

```
EMP_t  =  w₁ · (ΔE/E)  −  w₂ · (ΔR/R)
```

where weights w₁ and w₂ are the inverse standard deviations of each component (precision weighting), following Kaminsky, Lizondo & Reinhart (1998).

---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| **Coverage** | 1990–2023 (annual) |
| **Observations** | 34 years (32 usable after lag engineering) |
| **Crisis episodes** | ~14 years classified as currency crisis |
| **Non-crisis episodes** | ~18 years |
| **Primary sources** | CBN Statistical Bulletin, IMF IFS, World Bank WDI, OPEC |

### Raw Features

| Feature | Description |
|---|---|
| `ExRate_NGBUSD` | Official NGN/USD exchange rate (end of year) |
| `ExRate_Depr_pct` | Annual % depreciation of the Naira |
| `Parallel_Premium` | Parallel market premium over official rate (%) |
| `FX_Reserves_USDbn` | Gross foreign exchange reserves (USD billions) |
| `Reserves_Change_pct` | Annual % change in FX reserves |
| `Oil_Price_USD` | Average annual Brent crude price (USD/barrel) |
| `OilRev_pct_TotalRev` | Oil revenue as % of total government revenue |
| `Inflation_pct` | Annual consumer price inflation (%) |
| `M2_Growth_pct` | Broad money supply growth (%) |
| `Trade_Bal_USDbn` | Trade balance (USD billions) |
| `EMP_Index` | Constructed Exchange Market Pressure index |
| `IMF_Program` | Binary flag — active IMF programme (1 = yes) |

### Engineered Features

| Feature | Construction |
|---|---|
| `*_lag1`, `*_lag2` | One- and two-year lags of 8 key variables (leading indicators) |
| `Oil_Vulnerability` | `(OilRev_pct_TotalRev / 100) × (1 / Oil_Price_USD)` — combined oil dependence signal |
| `Import_Cover_Proxy` | `FX_Reserves_USDbn / |Trade_Bal_USDbn|` — reserve adequacy relative to trade exposure |
| `Currency_Overvaluation` | Deviation of official rate from estimated equilibrium path |

---

## 🏗️ Pipeline Architecture

```
Raw Data (1990–2023)
        │
        ▼
 1. Exploratory Data Analysis
    ├── Exchange rate & EMP timeline
    ├── Correlation heatmap
    └── KDE plots: crisis vs. no-crisis distributions
        │
        ▼
 2. EMP Index Construction & Crisis Label Verification
    ├── Precision-weighted EMP formula
    ├── Threshold = μ + 1.5σ
    └── Reconcile EMP rule with composite crisis label
        │
        ▼
 3. Feature Engineering
    ├── Lag features (t-1, t-2) for 8 variables
    ├── Oil Vulnerability score
    ├── Import Cover Proxy
    └── Currency Overvaluation indicator
        │
        ▼
 4. Preprocessing
    ├── StandardScaler (normalisation)
    └── SMOTE (k=3, class imbalance correction)
        │
        ▼
 5. Model Training
    ├── 6.1 Baseline Random Forest (n=300, OOB scoring)
    ├── 6.2 Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)
    └── 6.3 Final model fit on best params
        │
        ▼
 6. Model Evaluation & Comparison
    ├── Walk-forward CV: 5 models × 4 metrics
    ├── Confusion matrix, ROC curve, Precision-Recall curve
    └── Optimal decision threshold analysis
        │
        ▼
 7. SHAP Explainability
    ├── Beeswarm summary plot (top 18 features)
    ├── Feature importance bar chart (top 15)
    ├── Waterfall plot — 2023 case study
    └── MDI vs. SHAP importance comparison
        │
        ▼
 8. Early Warning Score Output
    ├── Crisis probability per year (1992–2023)
    ├── Alert levels: 🟢 Low | 🟡 Moderate | 🟠 High | 🔴 Critical
    └── Optimal threshold analysis (Precision / Recall / F1 trade-off)
```

---

## 🤖 Models Compared

| Model | Notes |
|---|---|
| **Random Forest (Tuned)** | Primary model; 300–600 trees, balanced class weight, OOB scoring |
| **Gradient Boosting** | 200 trees, learning rate 0.05, max depth 3 |
| **Logistic Regression** | Balanced class weight, max_iter=500 |
| **Decision Tree** | Balanced class weight, max depth 5 |
| **K-Nearest Neighbours** | k=5 |

All models are evaluated using **walk-forward (time-series) cross-validation** (`TimeSeriesSplit`, n=5) to preserve temporal ordering and prevent future data leakage.

---

## ⚙️ Hyperparameter Tuning

Tuning is performed in **Section 6.2** using scikit-learn's search utilities with `TimeSeriesSplit(n_splits=3)` and `scoring='roc_auc'`.

**Option A — GridSearchCV (default)**

```python
param_grid = {
    'n_estimators'      : [200, 300, 500],
    'max_depth'         : [3, 5, 10, None],
    'max_features'      : ['sqrt', 'log2', 0.5],
    'min_samples_leaf'  : [1, 2, 4],
    'min_samples_split' : [2, 5, 10],
}
```

**Option B — RandomizedSearchCV (faster)**  
Samples 80 random combinations from continuous and discrete distributions using `scipy.stats` priors. Uncomment Option B in Section 6.2 to switch.

---

## 📈 Key Outputs

- **EWS probability table** — Crisis probability and alert level for every year 1992–2023
- **Optimal threshold chart** — Precision / Recall / F1 trade-off to support policy calibration
- **SHAP beeswarm plot** — Direction and magnitude of each feature's influence on crisis predictions
- **SHAP waterfall plot (2023)** — Case study of the FX unification crisis episode
- **MDI vs. SHAP comparison** — Validates that the built-in RF importance and model-agnostic SHAP rankings agree
- **ROC & Precision-Recall curves** — Full discrimination and calibration assessment
- **Model comparison chart** — AUC-ROC, F1, Recall, Precision across all five classifiers

---

## 🚀 Getting Started

### Run on Google Colab (recommended)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### Run locally

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/nigeria-currency-crisis-ews.git
cd nigeria-currency-crisis-ews

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the notebook
jupyter notebook Nigeria_Currency_Crisis_EWS_RandomForest.ipynb
```

---

## 📦 Requirements

```
scikit-learn>=1.3
imbalanced-learn>=0.11
shap>=0.44
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scipy>=1.11
```

Install all at once:

```bash
pip install scikit-learn imbalanced-learn shap pandas numpy matplotlib seaborn scipy
```

---

## 📁 Repository Structure

```
nigeria-currency-crisis-ews/
│
├── Nigeria_Currency_Crisis_EWS_RandomForest.ipynb   # Main notebook (Google Colab)
├── README.md                                         # This file
├── requirements.txt                                  # Python dependencies
│
├── outputs/
│   ├── nigeria_corr_heatmap.png                     # Correlation heatmap
│   ├── nigeria_shap_summary.png                     # SHAP beeswarm plot
│   ├── nigeria_shap_waterfall_2023.png              # SHAP waterfall — 2023
│   ├── nigeria_ews_timeline.png                     # EWS probability chart
│   └── nigeria_roc_curve.png                        # ROC curve
│
└── data/
    └── Nigeria_Currency_Crisis_Dataset.xlsx          # Raw dataset (optional upload)
```

---

## 📐 Methodological Notes

- **EMP index** follows the precision-weighting approach of Kaminsky et al. (1998), using inverse standard deviations as component weights to normalise for different variances.
- **Class imbalance** is addressed with SMOTE (`k_neighbors=3`) and `class_weight='balanced'` inside Random Forest, ensuring the model does not default to predicting the majority class.
- **Temporal integrity** is preserved throughout using `TimeSeriesSplit` for all cross-validation — no shuffling, no future data leakage.
- **OOB score** (Out-of-Bag) is used as an additional unbiased performance estimate, a unique advantage of bootstrap-based ensemble methods.
- **Dual importance analysis** — MDI (Mean Decrease in Impurity, RF built-in) and SHAP values are computed side-by-side to cross-validate feature rankings and detect any MDI bias toward high-cardinality features.
- **Optimal threshold** is derived from the Precision-Recall curve by maximising F1-score, allowing policymakers to tune the EWS sensitivity to their preference for false alarm vs. missed crisis trade-offs.

---

## 📚 Data Sources

| Source | Variables |
|---|---|
| **Central Bank of Nigeria (CBN) Statistical Bulletin** | Exchange rates, FX reserves, M2, credit |
| **IMF International Financial Statistics (IFS)** | BOP, EMP components, external debt |
| **World Bank World Development Indicators (WDI)** | GDP growth, inflation, trade balance |
| **OPEC Annual Statistical Bulletin** | Oil prices, Nigerian crude production |
| **IMF World Economic Outlook (WEO)** | Fiscal aggregates, current account |

---

## 📖 References

- Kaminsky, G., Lizondo, S. & Reinhart, C.M. (1998). *Leading Indicators of Currency Crises*. IMF Staff Papers, 45(1), 1–48.
- Rose, A.K. & Spiegel, M.M. (2012). *Cross-country causes and consequences of the crisis: An update*. IMF Economic Review, 60(2), 281–312.
- Peltonen, T.A. (2006). *Are Emerging Market Currency Crises Predictable? A Test*. ECB Working Paper Series, No. 571.

---

## 📜 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
