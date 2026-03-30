# Final Report — Tech Job Market Salary Analysis

**Course:** DSA210 Introduction to Data Science — Spring 2026
**Author:** cansatir
**Repository:** https://github.com/cansatir/dsa210-spring2026-project

---

## 1. Motivation

The data job market is large, fast-moving, and opaque. Advertised salaries vary by orders
of magnitude depending on role, country, employer, and the specific skills listed in the
job posting — yet these factors are rarely quantified in a principled way.

This project asks: **can we predict a data professional's advertised salary from publicly
available job metadata?** To answer this, I combined two complementary datasets — a
large corpus of real job postings and official U.S. government wage benchmarks — and
applied statistical hypothesis testing and machine-learning regression and classification
to uncover which factors matter most.

The secondary goal is to validate four concrete hypotheses about salary determinants
(remote work, Python fluency, seniority, and gap relative to BLS benchmarks) before
building predictive models.

---

## 2. Data Sources

### 2.1 `lukebarousse/data_jobs` (HuggingFace)

- **Description:** ~786,000 tech job postings scraped from job boards worldwide in 2023.
- **Key fields:** `job_title_short`, `job_country`, `salary_year_avg`, `job_work_from_home`,
  `job_skills` (free-text list of required skills), `job_posted_date`.
- **License:** CC-BY 4.0.
- **Limitations:** Salary is self-reported by employers and only present for ~10 % of rows;
  postings are not deduplicated across job boards.

### 2.2 OEWS National Occupational Employment and Wage Statistics (BLS, May 2023)

- **Description:** Official U.S. government annual wage estimates for ~900 occupational
  codes, published by the Bureau of Labor Statistics.
- **Key fields:** `OCC_CODE`, `OCC_TITLE`, `A_MEAN` (annual mean wage).
- **License:** Public domain.
- **Use in this project:** Provides a ground-truth benchmark to compare advertised salaries
  against; used to compute a `salary_gap` feature (advertised − BLS mean).

---

## 3. Data Analysis

### 3.1 Cleaning and Preprocessing

The raw jobs dataset required several cleaning steps before analysis:

1. **Salary filter:** Dropped all rows where `salary_year_avg` was null (~90 % of the
   dataset). The retained subset (~15,000 rows) skews toward US-based, English-language
   postings with explicit salary disclosure.
2. **Outlier removal:** Applied a 1.5 × IQR fence to `salary_year_avg`, removing extreme
   entries (e.g., $1 or $10,000,000 typos).
3. **Date parsing:** Converted `job_posted_date` to `datetime` and extracted month for
   time-series visualisation.
4. **Seniority extraction:** Parsed seniority level (Junior / Mid / Senior) from
   `job_title` using keyword matching (`senior`, `sr`, `lead`, `junior`, `jr`, `entry`).
5. **Skills parsing:** `job_skills` was stored as a string representation of a Python list;
   parsed with `ast.literal_eval` and a regex fallback.

### 3.2 EDA Findings

**Salary distribution** (`fig1_salary_distribution.png`): Right-skewed; median ≈ $105 k,
with a long tail above $180 k.

**Salary by role** (`fig2_salary_by_role.png`): Machine Learning Engineers and Senior Data
Scientists command the highest medians; Data Analysts sit at the bottom.

**Remote vs on-site** (`fig3_remote_vs_onsite.png`): Remote postings show a modestly
higher median salary and a heavier upper tail.

**Top skills** (`fig4_top_skills.png`): SQL, Python, and Excel appear in the most postings;
cloud skills (AWS, Azure, Spark) cluster in higher-salary roles.

**Country distribution** (`fig5_salary_by_country.png`): US-based postings dominate and
pay substantially more than all other countries; the next tier includes Australia and
the UK.

**Postings over time** (`fig6_postings_over_time.png`): Volume peaks in Q1 2023, with a
gradual decline through the rest of the year.

**Salary gap** (`fig7_salary_gap_distribution.png`): Most advertised salaries exceed BLS
benchmarks by $20–60 k; a minority fall below the government average.

### 3.3 Hypothesis Test Results

All four hypotheses were tested at the α = 0.05 significance level.

| # | Hypothesis | Test | Result |
|---|---|---|---|
| H1 | Remote postings pay more than on-site | Mann-Whitney U | **Supported** (p < 0.05) |
| H2 | Python skill commands a salary premium | Mann-Whitney U | **Supported** (p < 0.05) |
| H3 | Advertised salaries exceed BLS averages | One-sample t-test on salary_gap | **Supported** (p < 0.05) |
| H4 | Salary differs significantly across seniority levels | Kruskal-Wallis | **Supported** (p < 0.05) |

Non-parametric tests (Mann-Whitney U, Kruskal-Wallis) were chosen for H1, H2, and H4
because salary distributions are skewed and non-normal. The one-sample t-test for H3 was
applied to the `salary_gap` variable, which is approximately normal after outlier removal.

---

## 4. ML Results

### 4.1 Feature Engineering

Features were constructed from the enriched dataset:

- **Binary skill indicators** for the top 20 most-mentioned skills (sql, python, aws,
  azure, tableau, spark, excel, power bi, r, sas, hadoop, java, snowflake, airflow,
  kafka, redshift, oracle, databricks, nosql, scala).
- **One-hot encoding** of `job_title_short` (top 10 titles) and `job_country_grp`
  (top 15 countries + "Other").
- **Binary flags:** `job_work_from_home`, `job_no_degree_mention`.
- **Seniority encoding:** One-hot `seniority_Junior / Mid / Senior`.
- **BLS enrichment:** `salary_gap` and `bls_mean_salary` carried forward.

Training / test split: 80 / 20, stratified by seniority. Final feature matrix: 57 columns.

### 4.2 Model Comparison

Three regression models were trained with default hyperparameters on the training split
and evaluated on the held-out test split.

| Model | RMSE (USD) | MAE (USD) | R² |
|---|---|---|---|
| Linear Regression | 34,415 | 27,426 | 0.306 |
| Random Forest | 34,097 | 26,454 | 0.319 |
| **XGBoost** | **33,522** | **26,256** | **0.342** |

XGBoost is the best model across all three metrics. The overall R² of 0.34 is moderate —
reasonable for salary prediction from job-posting metadata alone, without employer, years
of experience, or candidate data.

### 4.3 Key Features

Both Random Forest and XGBoost agree on the top drivers (see `fig9_feature_importance.png`
and `fig10_shap_summary.png`):

1. `job_title_short_Data Analyst` — strongest single predictor (RF importance: 0.197)
2. `seniority_Senior` — second most important (XGB importance: 0.195)
3. `job_title_short_Senior Data Analyst`
4. `job_country_grp_United States`
5. `skill_sql`, `skill_aws`, `skill_python`

SHAP analysis confirms these rankings and shows that being in the US adds ~$15–20 k to the
predicted salary, while holding the Python skill flag set to 1 adds ~$5–10 k.

### 4.4 Seniority Classification

A Random Forest classifier was trained to predict seniority level (Junior / Mid / Senior)
from salary and skill features. The confusion matrix (`fig11_confusion_matrix.png`) shows
strong performance on Senior and Junior classes; Mid-level predictions have the most
misclassifications, which is expected given the ambiguous boundary in job titles.

---

## 5. Key Findings

1. **Job title is the dominant salary predictor** — more so than any individual skill.
   Being a "Data Analyst" vs. a "Machine Learning Engineer" creates a larger salary gap
   than any combination of technical skills.

2. **Seniority is the second-strongest predictor.** Senior roles in every title group
   earn substantially more. The Kruskal-Wallis test confirmed this difference is
   statistically significant across all three seniority levels.

3. **US location drives salary above all other geographic factors.** The US premium
   persists even after controlling for job title and seniority.

4. **Remote work is associated with higher pay**, but the effect size is modest compared
   to title and seniority. This may reflect selection bias: companies that advertise
   salaries tend to be larger, more established firms that also offer remote flexibility.

5. **Python and SQL skills are widespread but still predictive.** Their prevalence (>50 %
   of postings) means they contribute moderate marginal value; rarer cloud skills (Spark,
   Kafka, Databricks) show higher per-skill premiums.

6. **Advertised salaries exceed BLS benchmarks by a median of ~$25 k.** This gap is
   largest for ML Engineers and Senior Data Scientists.

7. **~34 % of salary variance is explainable from job-posting metadata.** The remaining
   variance likely reflects employer size, industry, candidate negotiation, and
   unreported compensation components (equity, bonus).

---

## 6. Limitations and Future Work

### Limitations

- **Salary missingness (~90 %):** Only postings with explicit salary disclosure were
  analysed. This introduces selection bias toward US-based, larger employers and likely
  overestimates global salary norms.
- **No hyperparameter tuning:** Models were trained with defaults. Grid search or Bayesian
  optimisation could improve R² by several points.
- **Single year of data (2023):** Salary trends in tech are volatile; findings may not
  generalise to other years.
- **Self-reported skills:** `job_skills` reflects what employers advertise, not what is
  actually required or used on the job.
- **BLS mapping is approximate:** OEWS occupations do not map one-to-one to the informal
  job title taxonomy used in job postings.

### Future Work

- **Include employer-level features** (company size, funding stage) to reduce unexplained
  variance.
- **NLP on job descriptions** to extract more granular skill and responsibility signals
  beyond the structured skills list.
- **Time-series modelling** of salary trends to capture the 2022–2023 tech market
  correction.
- **Salary imputation** for the 90 % missing rows using semi-supervised or multiple
  imputation techniques.
- **Causal inference** to disentangle the independent effect of individual skills from
  confounders like job title.

---

## 7. AI Assistance

The following is a transparent account of all AI-assisted portions of this project.

### Prompts and Generated Artifacts

| Step | Prompt Summary | What Was Generated |
|---|---|---|
| Project setup | "Create a zero-assumption Docker + Jupyter setup for a Python data science project. No local Python needed." | `Dockerfile`, `docker-compose.yml`, `Makefile`, initial `requirements.txt` |
| Data download script | "Write a script that downloads `lukebarousse/data_jobs` from HuggingFace and the BLS OEWS May 2023 Excel file, saves them to `data/raw/`." | `scripts/download_data.py` |
| Notebook skeleton | "Generate a Jupyter notebook skeleton with numbered sections: load, clean, enrich, EDA, hypothesis tests, feature engineering, regression models, feature importance, classification." | Section headers and import boilerplate in `notebooks/analysis.ipynb` |
| Data cleaning cells | "Write pandas code to drop null salaries, remove outliers via IQR, parse dates, extract seniority from job title, and parse skill lists." | Cells in Section 2 of the notebook |
| BLS enrichment | "Write code to map job_title_short to BLS OCC_CODE, merge with OEWS data, compute salary_gap, and save enriched parquet." | Section 3 cells |
| EDA figures | "Generate matplotlib/seaborn figures for: salary distribution, salary by role, remote vs on-site, top skills, salary by country, postings over time, salary gap." | Figures 1–7 (plotting code in Section 4) |
| Hypothesis tests | "Write scipy hypothesis tests: Mann-Whitney U for remote vs on-site and Python premium, one-sample t-test for salary gap, Kruskal-Wallis for seniority." | Section 5 cells |
| Feature engineering | "One-hot encode job title and country, create binary skill flags for top 20 skills, add seniority dummies, train/test split 80/20." | Section 6 cells |
| Regression models | "Train Linear Regression, Random Forest, and XGBoost regressors; evaluate with RMSE, MAE, R²; plot predicted vs actual for the best model." | Section 7 cells and fig8 |
| Feature importance + SHAP | "Plot RF and XGBoost feature importances side-by-side; generate a SHAP summary plot for XGBoost." | Section 8 cells, figs 9–10 |
| Seniority classifier | "Train a Random Forest classifier to predict seniority; plot the confusion matrix." | Section 9 cells, fig11 |
| README | "Write a complete README with motivation, dataset table, repo tree, two-command reproduction steps, and key findings summary." | `README.md` |
| Final report | "Write a final report in markdown with sections: motivation, data sources, analysis, ML results, findings, limitations, AI assistance." | `report/final_report.md` (this file) |

### Extent of AI Use

All analysis logic, statistical test choices, and interpretation of results were
authored or reviewed by the student. AI assistance was used to accelerate boilerplate
coding (Docker setup, pandas pipelines, matplotlib figures) and to draft documentation.
No AI tool was used to fabricate results, manipulate data, or bypass academic integrity
requirements.
