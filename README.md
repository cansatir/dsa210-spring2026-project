# DSA210 Spring 2026 — Tech Job Market Salary Analysis

**Author:** cansatir | **Course:** DSA210 Introduction to Data Science | **Term:** Spring 2026

---

## Motivation

What determines a data professional's salary? This project investigates how job title,
seniority, location, remote-work availability, and required technical skills predict
advertised salaries in the tech job market. It combines ~786 k real-world job postings
with official U.S. Bureau of Labor Statistics wage data to answer that question with
statistical tests and machine-learning models.

---

## Dataset

| Dataset | Source | Rows | License |
|---|---|---|---|
| `lukebarousse/data_jobs` | HuggingFace | ~786,000 | CC-BY 4.0 |
| OEWS National (May 2023) | BLS / data.bls.gov | ~900 occupations | Public Domain |

Raw data is downloaded automatically at runtime and is **not tracked in git**.

---

## Repository Structure

```
dsa210-spring2026-project/
├── data/
│   ├── raw/               # Downloaded at runtime (git-ignored, .gitkeep tracked)
│   └── processed/         # Generated at runtime (git-ignored, .gitkeep tracked)
├── figures/               # All output plots (11 PNGs)
├── notebooks/
│   └── analysis.ipynb     # Single end-to-end notebook
├── report/
│   └── final_report.md
├── results/
│   ├── feature_importance.csv
│   ├── hypothesis_results.json
│   └── model_comparison.csv
├── scripts/
│   └── download_data.py   # Fetches raw datasets inside container
├── Dockerfile
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## Reproducing the Results

**Requirements:** Docker (no other dependencies needed)

```bash
git clone https://github.com/cansatir/dsa210-spring2026-project
cd dsa210-spring2026-project
make build
```

`make build` starts Jupyter at **http://localhost:8888**.
Open `notebooks/analysis.ipynb` and run all cells top-to-bottom.
The notebook downloads data, runs all analysis, and writes figures and results automatically.

To stop the container:
```bash
make stop
```

---

## Key Findings

### Salary Prediction (Regression)

| Model | RMSE (USD) | MAE (USD) | R² |
|---|---|---|---|
| Linear Regression | 34,415 | 27,426 | 0.306 |
| Random Forest | 34,097 | 26,454 | 0.319 |
| **XGBoost** | **33,522** | **26,256** | **0.342** |

XGBoost achieved the best performance (R² = 0.34), explaining roughly a third of salary
variance from job metadata and skill flags alone.

### Top Salary Drivers

1. **Job title** — "Data Analyst" and "Senior Data Analyst" are the strongest predictors
2. **Seniority** — Senior roles command a substantial premium over junior/mid positions
3. **US location** — US-based roles pay significantly more than the global average
4. **Skills** — SQL, AWS, Python, Tableau, and Azure all contribute positively

### Hypothesis Tests

| Hypothesis | Result |
|---|---|
| H1: Remote roles pay more than on-site | Supported (Mann-Whitney U, p < 0.05) |
| H2: Python skill commands a salary premium | Supported (Mann-Whitney U, p < 0.05) |
| H3: Advertised salaries exceed BLS averages | Supported (one-sample t-test, p < 0.05) |
| H4: Salary differs across seniority levels | Supported (Kruskal-Wallis, p < 0.05) |

### Seniority Classification

A Random Forest classifier trained on salary, skills, and job metadata correctly
identifies seniority level (Junior / Mid / Senior) — see `figures/fig11_confusion_matrix.png`.

---

## License

Academic project — not for commercial use.
