# Predicting Measles Outbreaks with Machine Learning

This project asks one question: **will a country have a measles outbreak next year?**

It uses WHO measles and rubella surveillance data (2012–2025). Each country's yearly history becomes a supervised learning problem built from lagged features. Random Forest, Logistic Regression and SVM classifiers are compared on that problem. The best model, a tuned Random Forest, reaches **0.84 accuracy and 0.86 ROC-AUC** on years it never saw during training (2023–2025).

## Repository contents

| File | Description |
|---|---|
| `ml_assignment.ipynb` | The full analysis: EDA, outbreak labelling, feature engineering, model training and evaluation |
| `cases_month.csv` | Monthly measles/rubella case counts by country (22,780 rows) |
| `cases_year.csv` | Yearly case counts, population and incidence rates by country (2,382 rows) |

## Data

The data comes from the [TidyTuesday 2025-06-24 dataset](https://github.com/rfordatascience/tidytuesday/tree/main/data/2025/2025-06-24), which is drawn from WHO provisional measles and rubella surveillance data. It covers six WHO regions (AFRO, AMRO, EMRO, EURO, SEARO, WPRO) from 2012 to 2025.

Key columns:

- **Yearly:** `total_population`, `measles_total`, `measles_lab_confirmed`, `measles_epi_linked`, `measles_clinical`, `measles_incidence_rate_per_1000000_total_population`, `discarded_cases`
- **Monthly:** `month`, `measles_suspect`, `measles_clinical`, `measles_epi_linked`, `measles_lab_confirmed`, `measles_total`

> The notebook reads the CSVs directly from the TidyTuesday GitHub URLs. The local copies in this repo are a snapshot of the same files. To run offline, change the `pd.read_csv(...)` calls to `pd.read_csv('cases_month.csv')` and `pd.read_csv('cases_year.csv')`.

## Method

### 1. Exploratory data analysis
- Global measles cases by year and by month
- Monthly distribution per year, with a close look at 2020–2025
- Regional burden: total and average cases per region, the top countries, and the EUR region in 2024
- Seasonality by region

Main finding: cases peak in the first three months of the year and begin rising in October. Cases rose sharply after 2022, and early 2024 had the highest counts.

### 2. Defining an outbreak
An outbreak is labelled **per country**, using measles incidence per million population. Three thresholds were compared:

| Method | Rule |
|---|---|
| Mean + SD (**used as the target**) | incidence > country mean + 1 standard deviation |
| IQR | incidence > country median + 2 × IQR |
| 90th percentile | incidence > country p90 |

The three methods agree on 94–96% of country-years. Mean + SD was chosen as the target (`outbreak_mean_std`). Outbreak years are visualised as per-region heatmaps of country × year. The notebook also computes the coefficient of variation per country, to show which countries have the most volatile incidence.

### 3. Feature engineering
All features use only information from **previous years**, so the model cannot see the year it is predicting.

| Feature set | Features |
|---|---|
| **Baseline** | `incidence_lag1`, `cases_lag1`, `growth_lag1` (year-on-year % change), `roll3_lag1` (3-year rolling mean of incidence), one-hot region |
| **Enhanced** | Baseline + `lab_confirm_rate`, `discard_rate`, `clinical_ratio`, `epi_ratio` |
| **Advanced** | Enhanced + monthly-derived features: `peak_cases`, `variability`, `outbreak_duration_lag1` (months with >10 cases), `peak_season_lag1` (cases in Oct–Mar), `was_outbreak_lag1` |

### 4. Modelling and evaluation
- **Temporal split:** train on years ≤ 2022, test on 2023–2025
- **Class imbalance:** about 20% of country-years are outbreaks, so the models use `class_weight='balanced'`
- **Nested cross-validation:** 5-fold outer and 3-fold inner, with `GridSearchCV` for the Random Forest, used to compare model families
- **Hyperparameter tuning:** 5-fold `GridSearchCV` over `n_estimators`, `max_depth` and `min_samples_leaf`

## Results

**Model comparison (nested CV, ROC-AUC):**

| Model | Mean ROC-AUC |
|---|---|
| Random Forest | **0.890** |
| Logistic Regression | 0.692 |
| SVM | 0.645 |
| SVM + StandardScaler | 0.634 |

**Feature sets (Random Forest, held-out test set 2023–2025, 342 rows):**

| Feature set | ROC-AUC | Outbreak recall | Outbreak F1 |
|---|---|---|---|
| Baseline | **0.863** | 0.70 | 0.60 |
| Enhanced | 0.832 | 0.54 | 0.50 |
| Advanced | 0.847 | 0.62 | 0.55 |

**Tuned Random Forest** (`n_estimators=200`, `max_depth=15`, `min_samples_leaf=1`, `class_weight='balanced'`):

- Training accuracy: 0.953
- Test accuracy: **0.839**
- Outbreak class: precision 0.59, recall 0.64, F1 0.62

The most useful signals are lagged incidence and case counts. The extra surveillance-quality and seasonal features did not beat the simpler baseline. The advanced model correctly flagged recent outbreaks in countries such as Ethiopia (2024), Malawi (2025), Cameroon (2023), Canada (2025) and Kyrgyzstan (2025).

## Getting started

### Requirements
- Python 3.9+
- Jupyter Notebook or JupyterLab

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Run
```bash
jupyter notebook ml_assignment.ipynb
```

Run the cells from top to bottom. The notebook needs an internet connection to fetch the data, unless you switch it to the local CSVs as described above.

## Notes and limitations

- **Labels use the full time range.** Each country's outbreak threshold (mean + SD) is computed from all years, including the test years. Some information about the test period therefore reaches the training labels. Computing thresholds from training years only would give a stricter evaluation.
- Small countries with very low incidence can be flagged as having an "outbreak" from just a few cases, because thresholds are relative to each country.

## Acknowledgements

- Data: World Health Organization, via [TidyTuesday](https://github.com/rfordatascience/tidytuesday) (R for Data Science community)
- Built with pandas, scikit-learn, matplotlib and seaborn
