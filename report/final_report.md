# Final Report — Tech Job Market Salary Analysis

**Course:** DSA210 Introduction to Data Science — Spring 2026
**Author:** cansatir
**Repository:** https://github.com/cansatir/dsa210-spring2026-project

---

## 1. Motivation

I am a computer science student who will be entering the job market in a few years, and I kept noticing that tech salaries are everywhere — some job postings list $80k, others list $180k for what looks like the same role. I wanted to understand what actually drives that difference. Is it the job title? The skills you have? Whether the job is remote? Or just where you happen to live?

That question became this project. I took a large dataset of real tech job postings and combined it with official U.S. government wage data to see if I could find patterns — and eventually build a model that predicts salary from things you can actually read in a job ad.

---

## 2. Data Sources

### 2.1 `lukebarousse/data_jobs` (HuggingFace)

- **Description:** ~786,000 tech job postings scraped from job boards worldwide in 2023.
- **Key fields:** `job_title_short`, `job_country`, `salary_year_avg`, `job_work_from_home`, `job_skills`, `job_posted_date`.
- **License:** CC-BY 4.0.
- **Limitation:** Only about 10% of postings include salary data, which means the analysis is based on employers who voluntarily disclose compensation — likely skewed toward larger US-based companies.

### 2.2 OEWS National Occupational Employment and Wage Statistics (BLS, May 2024)

- **Description:** Official U.S. government annual wage estimates for ~900 occupational codes, published by the Bureau of Labor Statistics.
- **Key fields:** `OCC_CODE`, `OCC_TITLE`, `A_MEAN` (annual mean wage).
- **License:** Public domain.
- **Why I added this:** I wanted a ground-truth benchmark — something to compare advertised salaries against rather than just looking at the job posting data in isolation. By computing `salary_gap = advertised salary − BLS average`, I could see whether companies were paying above or below the official norm for each role.

---

## 3. Data Analysis

### 3.1 Cleaning and Preprocessing

The raw data needed quite a bit of work before I could analyze it:

1. **Dropped rows without salary** — about 90% of the dataset. This was the most painful step, but there is nothing you can do with a salary analysis when the salary is missing.
2. **Removed outliers** — used 1.5× IQR to filter out entries that looked like data entry errors (e.g., $1 or $10,000,000).
3. **Parsed dates** — converted `job_posted_date` to datetime so I could look at trends over time.
4. **Extracted seniority** — I wrote a keyword matcher to label each job as Junior, Mid, or Senior based on words in the job title (`senior`, `sr`, `lead`, `junior`, `jr`, `entry`). It is not perfect, but it worked reasonably well.
5. **Parsed skills** — the skills column was stored as a string that looked like a Python list, so I used `ast.literal_eval` with a regex fallback to convert it into an actual list.

### 3.2 EDA Findings

The first thing I noticed was that salary is bimodal — there is a cluster around $80–100k and another around $140–160k, which roughly corresponds to mid-level and senior roles. The overall median was $115k.

When I broke it down by role (`fig2`), the hierarchy made sense: Senior Data Scientists and Senior Data Engineers were at the top, while Data Analysts and Business Analysts were at the bottom. What surprised me was how wide the ranges were — a "Data Engineer" posting could pay anywhere from $30k to $220k depending on the employer.

For skills (`fig4`), SQL and Python came out on top by a large margin, which was expected. What was more interesting was that cloud skills like AWS, Spark, and Azure clustered in higher-paying roles.

The country chart (`fig5`) initially showed strange results — Bahamas and Dominican Republic at the top — because those countries had only a handful of postings that happened to be high-paying. After I filtered to countries with at least 50 postings, the picture made more sense: US, Australia, Ireland, and Canada led the rankings.

### 3.3 Hypothesis Tests

I tested four hypotheses at the α = 0.05 significance level. All four were supported.

| # | Hypothesis | Test | Result |
|---|---|---|---|
| H1 | Remote jobs pay more than on-site | Mann-Whitney U | **Supported** — remote median $127.5k vs on-site $115k (p < 0.001) |
| H2 | Python skill commands a salary premium | Mann-Whitney U | **Supported** — +$24.7k median premium (p ≈ 0) |
| H3 | Advertised salaries differ from BLS averages | One-sample t-test | **Supported** — mean gap −$2,086 (p < 0.001) |
| H4 | Salary differs across seniority levels | Kruskal-Wallis | **Supported** — Junior $70k / Mid $109k / Senior $140k (p ≈ 0) |

I used non-parametric tests for H1, H2, and H4 because salary distributions are heavily skewed — a standard t-test would not have been appropriate here.

---

## 4. ML Results

### 4.1 Feature Engineering

I built features from everything available in a job posting: binary flags for the top 20 skills, one-hot encoded job titles and countries, remote/degree flags, and seniority dummies. After encoding, I had 57 features. I split the data 80/20 for training and testing.

### 4.2 Model Comparison

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | $34,415 | $27,426 | 0.306 |
| Random Forest | $34,097 | $26,454 | 0.319 |
| **XGBoost** | **$33,522** | **$26,256** | **0.342** |

XGBoost performed best across all metrics. An R² of 0.34 means the model explains about a third of salary variance — which sounds low, but makes sense given that the biggest salary drivers (years of experience, company size, equity) are simply not in a job posting.

### 4.3 Key Features

Both the feature importance plot (`fig9`) and SHAP analysis (`fig10`) pointed to the same top drivers:

1. **Job title** — being a "Data Analyst" vs. a "Senior Data Scientist" matters more than any skill.
2. **Seniority** — the Senior flag was consistently the second most important feature.
3. **US location** — working in the US adds roughly $15–20k according to SHAP values.
4. **Python and SQL** — widespread but still predictive; +$5–10k from Python alone.

### 4.4 Seniority Classification

I also trained a Random Forest classifier to predict Junior / Mid / Senior from the available features. It reached **84% accuracy**. Looking at the confusion matrix (`fig11`), Junior was the hardest class to predict — there are very few Junior postings in the dataset, so the model has limited examples to learn from. Senior and Mid performed well.

---

## 5. Key Findings

1. **Job title matters most.** The gap between a Data Analyst and a Machine Learning Engineer is larger than any combination of individual skills.
2. **Seniority is the second-strongest predictor.** This was not surprising, but the scale confirmed it — Senior roles earn roughly twice as much as Junior roles.
3. **Remote work is associated with higher pay**, but the effect is modest. I think this partly reflects that companies willing to advertise salaries and offer remote work tend to be larger, better-paying employers.
4. **Python and SQL are table stakes**, not differentiators. Their marginal salary impact is positive but smaller than job title or location.
5. **Advertised salaries are slightly below BLS benchmarks** on average (−$2,086), which was surprising — I expected companies to advertise higher numbers to attract candidates.
6. **Only 34% of salary variance is explainable from a job posting.** The rest comes from things that are not written in the ad.

---

## 6. Limitations and Future Work

### Limitations

- **90% of salary data is missing.** The analysis is based on a self-selected subset of employers who disclose compensation. This likely overestimates global salary norms.
- **BLS mapping is approximate.** I manually mapped job titles to SOC codes, which is imperfect — some roles do not have a clean equivalent in the BLS taxonomy.
- **Single year (2023).** Tech salaries were shifting significantly during this period; the findings may not generalize to other years.
- **No hyperparameter tuning.** I used default settings for all models. GridSearch or Bayesian optimization could improve R² further.
- **Incomplete data toward end of collection period.** Job posting counts decline sharply after August 2023 (`fig6`), which appears to be a data collection artifact rather than a real market trend. Time-series conclusions should be treated with caution.

### Future Work

- Add employer-level features (company size, funding) to reduce unexplained variance.
- Use NLP on job descriptions to extract richer skill and responsibility signals.
- Apply salary imputation to recover the 90% of missing salary rows.
- Expand to multiple years to track salary trends over time.

---

## 7. AI Assistance

This project used AI tools (Claude) to accelerate boilerplate coding and documentation. All analysis logic, hypothesis choices, and result interpretation were authored and reviewed by me.

| Step | What AI Generated |
|---|---|
| Docker + Makefile setup | `Dockerfile`, `docker-compose.yml`, `Makefile`, `requirements.txt` |
| Data download script | `scripts/download_data.py` |
| Notebook skeleton | Section headers and import boilerplate |
| Cleaning, enrichment, EDA, hypothesis test cells | Pandas/scipy/matplotlib code |
| Feature engineering + ML models | Sklearn/XGBoost/SHAP code |
| README and final report drafts | Markdown documents (reviewed and edited by student) |

The specific prompts used to generate each section are documented in the conversation history with the AI assistant used throughout this project.
