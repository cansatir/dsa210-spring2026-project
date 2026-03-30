#!/usr/bin/env python3
"""
Generate notebooks/analysis.ipynb up to a given stage.
Usage:  python scripts/build_notebook.py <stage>
  stage 1 → Section 1 (Load & Inspect)
  stage 2 → Section 2 (Cleaning)
  stage 3 → Section 3 (OEWS Enrichment)
  stage 4 → Section 4 (EDA)
  stage 5 → Section 5 (Hypothesis Tests)
"""
import json, sys
from pathlib import Path

stage = int(sys.argv[1]) if len(sys.argv) > 1 else 5

def md(cell_id, source):
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source}

def code(cell_id, source):
    return {
        "cell_type": "code", "execution_count": None,
        "id": cell_id, "metadata": {}, "outputs": [], "source": source,
    }

# ── Section 0: Setup ─────────────────────────────────────────────────────────
S0 = [
    md("md-title",
       "# DSA210 — Tech Job Market Salary Analysis\n"
       "**Author:** cansatir  \n"
       "**Datasets:** lukebarousse/data_jobs · BLS OEWS"),
    md("md-s0", "## 0. Setup & Imports"),
    code("code-s0",
"""from pathlib import Path
import ast, json, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Pin CWD to project root (works whether run interactively or via nbconvert)
_cwd = Path.cwd()
PROJECT_ROOT = _cwd.parent if _cwd.name == 'notebooks' else _cwd
os.chdir(PROJECT_ROOT)

plt.rcParams['figure.figsize'] = (12, 5)
sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns', None)

DATA_RAW       = PROJECT_ROOT / 'data' / 'raw'
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
FIGURES        = PROJECT_ROOT / 'figures'
RESULTS        = PROJECT_ROOT / 'results'

for d in (DATA_PROCESSED, FIGURES, RESULTS):
    d.mkdir(parents=True, exist_ok=True)

print(f'Project root: {PROJECT_ROOT}')"""),
]

# ── Section 1: Load & Inspect ─────────────────────────────────────────────────
S1 = [
    md("md-s1", "## 1. Load & Inspect Raw Datasets"),
    code("code-s1a",
"""# ── Jobs dataset ──────────────────────────────────────────────
df_jobs = pd.read_parquet(DATA_RAW / 'jobs_raw.parquet')
print(f'Shape: {df_jobs.shape}')
print(f'\\nDtypes:\\n{df_jobs.dtypes}')
print(f'\\nMissing values (top 10):\\n{df_jobs.isnull().sum().sort_values(ascending=False).head(10)}')
df_jobs.head()"""),
    code("code-s1b",
"""# ── OEWS enrichment dataset ───────────────────────────────────
OEWS_COLS = ['OCC_CODE', 'OCC_TITLE', 'A_MEAN', 'A_MEDIAN',
             'A_PCT10', 'A_PCT25', 'A_PCT75', 'A_PCT90']
oews_raw = pd.read_excel(DATA_RAW / 'oews_national.xlsx',
                         header=0, dtype={'OCC_CODE': str})
print(f'Shape: {oews_raw.shape}')
print(f'\\nDtypes:\\n{oews_raw.dtypes}')
print(f'\\nMissing values (key cols):\\n{oews_raw[OEWS_COLS].isnull().sum()}')
oews_raw[OEWS_COLS].head()"""),
    code("code-s1c",
"""print(f'Section 1 complete — jobs: {df_jobs.shape}, oews: {oews_raw.shape}')"""),
]

# ── Section 2: Cleaning ───────────────────────────────────────────────────────
S2 = [
    md("md-s2", "## 2. Data Cleaning & Preprocessing"),
    code("code-s2a",
"""# ── Drop rows with null salary_year_avg ───────────────────────
n_before = len(df_jobs)
df_clean = df_jobs.dropna(subset=['salary_year_avg']).copy()
n_dropped = n_before - len(df_clean)
print(f'Dropped {n_dropped:,} rows ({100 * n_dropped / n_before:.1f}%) with null salary_year_avg')"""),
    code("code-s2b",
"""# ── Remove salary outliers via 1.5× IQR ───────────────────────
q1, q3 = df_clean['salary_year_avg'].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_before_iqr = len(df_clean)
df_clean = df_clean[df_clean['salary_year_avg'].between(lo, hi)].copy()
n_removed = n_before_iqr - len(df_clean)
print(f'IQR bounds: [{lo:,.0f}, {hi:,.0f}]')
print(f'Removed {n_removed:,} outliers ({100 * n_removed / n_before_iqr:.1f}%)')
print(f'Remaining rows: {len(df_clean):,}')"""),
    code("code-s2c",
"""# ── Parse job_posted_date to datetime ─────────────────────────
df_clean['job_posted_date'] = pd.to_datetime(df_clean['job_posted_date'])
print('Date range:', df_clean['job_posted_date'].min(),
      '→', df_clean['job_posted_date'].max())"""),
    code("code-s2d",
"""# ── Extract seniority from job_title ──────────────────────────
def extract_seniority(title: str) -> str:
    t = str(title).lower()
    if any(w in t for w in ('senior', 'lead', 'principal', 'sr.')):
        return 'Senior'
    if any(w in t for w in ('junior', 'jr.', 'jr ', 'entry')):
        return 'Junior'
    return 'Mid'

df_clean['seniority'] = df_clean['job_title'].apply(extract_seniority)
print('Seniority distribution:')
print(df_clean['seniority'].value_counts())"""),
    code("code-s2e",
"""# ── Parse job_skills string into Python list ──────────────────
def parse_skills(s) -> list:
    if pd.isna(s):
        return []
    try:
        return ast.literal_eval(s)
    except Exception:
        return []

df_clean['skills_list'] = df_clean['job_skills'].apply(parse_skills)
n_with_skills = (df_clean['skills_list'].apply(len) > 0).sum()
print(f'Rows with ≥1 skill: {n_with_skills:,}')"""),
    code("code-s2f",
"""# ── Save cleaned data ─────────────────────────────────────────
df_clean.to_parquet(DATA_PROCESSED / 'jobs_clean.parquet', index=False)
print(f'Section 2 complete — {len(df_clean):,} rows saved to jobs_clean.parquet')"""),
]

# ── Section 3: OEWS Enrichment ────────────────────────────────────────────────
S3 = [
    md("md-s3", "## 3. Enrich with BLS OEWS"),
    code("code-s3a",
"""# ── Reload cleaned data and OEWS ──────────────────────────────
df_clean = pd.read_parquet(DATA_PROCESSED / 'jobs_clean.parquet')
OEWS_COLS = ['OCC_CODE', 'OCC_TITLE', 'A_MEAN', 'A_MEDIAN',
             'A_PCT10', 'A_PCT25', 'A_PCT75', 'A_PCT90']
oews = pd.read_excel(DATA_RAW / 'oews_national.xlsx',
                     header=0, dtype={'OCC_CODE': str})
oews = oews[OEWS_COLS].drop_duplicates('OCC_CODE').copy()
for col in ['A_MEAN', 'A_MEDIAN', 'A_PCT10', 'A_PCT25', 'A_PCT75', 'A_PCT90']:
    oews[col] = pd.to_numeric(oews[col], errors='coerce')
print(f'OEWS rows after dedup: {len(oews)}')"""),
    code("code-s3b",
"""# ── Map job_title_short → BLS OCC_CODE ───────────────────────
OCC_MAP = {
    'Data Scientist':            '15-2051',
    'Data Engineer':             '15-1243',
    'Data Analyst':              '15-2031',
    'Software Engineer':         '15-1252',
    'Machine Learning Engineer': '15-2051',
    'Cloud Engineer':            '15-1299',
    'Business Analyst':          '13-1111',
    'Senior Data Scientist':     '15-2051',
    'Senior Data Engineer':      '15-1243',
    'Senior Data Analyst':       '15-2031',
}
df_clean = df_clean.copy()
df_clean['OCC_CODE'] = df_clean['job_title_short'].map(OCC_MAP)
mapped_pct = df_clean['OCC_CODE'].notna().mean() * 100
print(f'Title mapping coverage: {mapped_pct:.1f}% of rows')"""),
    code("code-s3c",
"""# ── Merge and compute salary_gap ─────────────────────────────
df_enriched = df_clean.merge(oews, on='OCC_CODE', how='left')
merge_rate = df_enriched['A_MEAN'].notna().mean() * 100
df_enriched['salary_gap'] = df_enriched['salary_year_avg'] - df_enriched['A_MEAN']
gap_mean   = df_enriched['salary_gap'].mean()
gap_median = df_enriched['salary_gap'].median()
print(f'Merge success rate:                  {merge_rate:.1f}%')
print(f'Mean salary gap (advertised − BLS):   ${gap_mean:,.0f}')
print(f'Median salary gap (advertised − BLS): ${gap_median:,.0f}')"""),
    code("code-s3d",
"""# ── Save enriched data ────────────────────────────────────────
df_enriched.to_parquet(DATA_PROCESSED / 'jobs_enriched.parquet', index=False)
print(f'Section 3 complete — {len(df_enriched):,} rows saved to jobs_enriched.parquet')"""),
]

# ── Section 4: EDA ────────────────────────────────────────────────────────────
S4 = [
    md("md-s4", "## 4. Exploratory Data Analysis"),
    code("code-s4-load",
"""df = pd.read_parquet(DATA_PROCESSED / 'jobs_enriched.parquet')
print(f'Loaded {len(df):,} rows for EDA')"""),

    # fig1
    code("code-s4-fig1",
"""# fig1 — Salary distribution ───────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
sns.histplot(df['salary_year_avg'], bins=60, kde=True, ax=ax)
ax.axvline(df['salary_year_avg'].mean(),
           color='red', linestyle='--',
           label=f"Mean ${df['salary_year_avg'].mean():,.0f}")
ax.axvline(df['salary_year_avg'].median(),
           color='orange', linestyle='--',
           label=f"Median ${df['salary_year_avg'].median():,.0f}")
ax.set_title('Salary Distribution')
ax.set_xlabel('Annual Salary (USD)')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / 'fig1_salary_distribution.png', dpi=120)
plt.show()"""),

    # fig2
    code("code-s4-fig2",
"""# fig2 — Salary by role (top 8) ────────────────────────────────
top_roles = df['job_title_short'].value_counts().head(8).index
df_top = df[df['job_title_short'].isin(top_roles)]
order = (df_top.groupby('job_title_short')['salary_year_avg']
               .median().sort_values(ascending=False).index)

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df_top, x='job_title_short', y='salary_year_avg',
            order=order, ax=ax)
ax.set_title('Salary by Role — Top 8 Titles')
ax.set_xlabel('Job Title')
ax.set_ylabel('Annual Salary (USD)')
ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig(FIGURES / 'fig2_salary_by_role.png', dpi=120)
plt.show()"""),

    # fig3
    code("code-s4-fig3",
"""# fig3 — Remote vs on-site ─────────────────────────────────────
df_r = df.copy()
df_r['work_type'] = df_r['job_work_from_home'].map({True: 'Remote', False: 'On-site'})

fig, ax = plt.subplots(figsize=(12, 5))
sns.boxplot(data=df_r, x='work_type', y='salary_year_avg', ax=ax)
ax.set_title('Remote vs On-site Salary')
ax.set_xlabel('Work Type')
ax.set_ylabel('Annual Salary (USD)')
plt.tight_layout()
plt.savefig(FIGURES / 'fig3_remote_vs_onsite.png', dpi=120)
plt.show()"""),

    # fig4
    code("code-s4-fig4",
"""# fig4 — Top 20 skills ─────────────────────────────────────────
from collections import Counter
all_skills = [s for row in df['skills_list'] for s in row]
top20 = pd.DataFrame(Counter(all_skills).most_common(20),
                     columns=['skill', 'count'])

fig, ax = plt.subplots(figsize=(10, 8))
sns.barplot(data=top20, x='count', y='skill', ax=ax)
ax.set_title('Top 20 Job Skills')
ax.set_xlabel('Count')
ax.set_ylabel('Skill')
plt.tight_layout()
plt.savefig(FIGURES / 'fig4_top_skills.png', dpi=120)
plt.show()"""),

    # fig5
    code("code-s4-fig5",
"""# fig5 — Top 10 countries by median salary ─────────────────────
top_countries = (df.groupby('job_country')['salary_year_avg']
                   .median().nlargest(10)
                   .reset_index(name='median_salary'))

fig, ax = plt.subplots(figsize=(12, 5))
sns.barplot(data=top_countries, x='median_salary', y='job_country', ax=ax)
ax.set_title('Top 10 Countries by Median Salary')
ax.set_xlabel('Median Annual Salary (USD)')
ax.set_ylabel('Country')
plt.tight_layout()
plt.savefig(FIGURES / 'fig5_salary_by_country.png', dpi=120)
plt.show()"""),

    # fig6
    code("code-s4-fig6",
"""# fig6 — Monthly job postings over time ────────────────────────
monthly = (df.assign(ym=df['job_posted_date'].dt.to_period('M'))
             .groupby('ym').size()
             .reset_index(name='postings'))
monthly['ym_str'] = monthly['ym'].astype(str)

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(monthly['ym_str'], monthly['postings'], marker='o', markersize=3)
ax.set_title('Monthly Job Postings Over Time')
ax.set_xlabel('Month')
ax.set_ylabel('Number of Postings')
step = max(1, len(monthly) // 12)
ax.set_xticks(range(0, len(monthly), step))
ax.set_xticklabels(monthly['ym_str'].iloc[::step], rotation=45, ha='right')
plt.tight_layout()
plt.savefig(FIGURES / 'fig6_postings_over_time.png', dpi=120)
plt.show()"""),

    # fig7
    code("code-s4-fig7",
"""# fig7 — Salary gap distribution ───────────────────────────────
gap = df['salary_gap'].dropna()

fig, ax = plt.subplots(figsize=(12, 5))
sns.histplot(gap, bins=50, kde=True, ax=ax)
ax.axvline(gap.mean(), color='red', linestyle='--',
           label=f"Mean gap ${gap.mean():,.0f}")
ax.axvline(0, color='black', linewidth=1,
           label='BLS average (reference)')
ax.set_title('Salary Gap Distribution (Advertised − BLS Average)')
ax.set_xlabel('Salary Gap (USD)')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / 'fig7_salary_gap_distribution.png', dpi=120)
plt.show()"""),

    # summary stats
    code("code-s4-summary",
"""# Summary stats by role ────────────────────────────────────────
summary = (df.groupby('job_title_short')['salary_year_avg']
             .agg(['mean', 'median', 'std', 'min', 'max'])
             .round(0)
             .sort_values('median', ascending=False))
print('Summary statistics by role:')
display(summary)
n_figs = len(list(FIGURES.glob('*.png')))
print(f'Section 4 complete — {n_figs} figures saved to {FIGURES}/')"""),
]

# ── Section 5: Hypothesis Tests ───────────────────────────────────────────────
S5 = [
    md("md-s5", "## 5. Hypothesis Tests"),
    code("code-s5-load",
"""df = pd.read_parquet(DATA_PROCESSED / 'jobs_enriched.parquet')
hyp_results = {}
print(f'Loaded {len(df):,} rows for hypothesis testing')"""),

    code("code-s5-h1",
"""# H1 — Remote vs on-site salary (Mann-Whitney U) ───────────────
remote = df[df['job_work_from_home'] == True]['salary_year_avg']
onsite = df[df['job_work_from_home'] == False]['salary_year_avg']
u1, p1 = stats.mannwhitneyu(remote, onsite, alternative='two-sided')

print('H1 — Remote vs On-site salary')
print(f'  Remote median:  ${remote.median():,.0f}  (n={len(remote):,})')
print(f'  On-site median: ${onsite.median():,.0f}  (n={len(onsite):,})')
print(f'  U={u1:.0f},  p={p1:.4f}')
conclusion1 = 'Remote jobs pay significantly more (p<0.05).' if p1 < 0.05 else 'No significant salary difference between remote and on-site (p≥0.05).'
print(f'  Conclusion: {conclusion1}')

hyp_results['H1_remote_vs_onsite'] = {
    'remote_median': float(remote.median()),
    'onsite_median': float(onsite.median()),
    'u_statistic': float(u1), 'p_value': float(p1),
    'significant': bool(p1 < 0.05), 'conclusion': conclusion1,
}"""),

    code("code-s5-h2",
"""# H2 — Python skill premium (Mann-Whitney U) ──────────────────
has_py = df[df['skills_list'].apply(lambda x: 'python' in x)]['salary_year_avg']
no_py  = df[df['skills_list'].apply(lambda x: 'python' not in x)]['salary_year_avg']
u2, p2 = stats.mannwhitneyu(has_py, no_py, alternative='two-sided')

print('H2 — Python skill premium')
print(f'  With Python:    ${has_py.median():,.0f}  (n={len(has_py):,})')
print(f'  Without Python: ${no_py.median():,.0f}  (n={len(no_py):,})')
print(f'  U={u2:.0f},  p={p2:.4f}')
conclusion2 = 'Python skill is linked to significantly higher salaries (p<0.05).' if p2 < 0.05 else 'No significant Python salary premium (p≥0.05).'
print(f'  Conclusion: {conclusion2}')

hyp_results['H2_python_premium'] = {
    'with_python_median': float(has_py.median()),
    'without_python_median': float(no_py.median()),
    'u_statistic': float(u2), 'p_value': float(p2),
    'significant': bool(p2 < 0.05), 'conclusion': conclusion2,
}"""),

    code("code-s5-h3",
"""# H3 — Advertised salary vs BLS average (one-sample t-test) ───
gap = df['salary_gap'].dropna()
t3, p3 = stats.ttest_1samp(gap, popmean=0)

print('H3 — Advertised salary vs BLS average  (H0: mean gap = 0)')
print(f'  Mean gap:  ${gap.mean():,.0f}')
print(f'  t={t3:.4f},  p={p3:.4f}')
conclusion3 = 'Advertised salaries differ significantly from BLS averages (p<0.05).' if p3 < 0.05 else 'No significant difference from BLS averages (p≥0.05).'
print(f'  Conclusion: {conclusion3}')

hyp_results['H3_advertised_vs_bls'] = {
    'mean_salary_gap': float(gap.mean()),
    't_statistic': float(t3), 'p_value': float(p3),
    'significant': bool(p3 < 0.05), 'conclusion': conclusion3,
}"""),

    code("code-s5-h4",
"""# H4 — Salary across seniority levels (Kruskal-Wallis) ────────
junior = df[df['seniority'] == 'Junior']['salary_year_avg']
mid    = df[df['seniority'] == 'Mid']['salary_year_avg']
senior = df[df['seniority'] == 'Senior']['salary_year_avg']
h4, p4 = stats.kruskal(junior, mid, senior)

print('H4 — Salary across seniority levels')
print(f'  Junior median:  ${junior.median():,.0f}  (n={len(junior):,})')
print(f'  Mid median:     ${mid.median():,.0f}  (n={len(mid):,})')
print(f'  Senior median:  ${senior.median():,.0f}  (n={len(senior):,})')
print(f'  H={h4:.4f},  p={p4:.4f}')
conclusion4 = 'Salary differs significantly across seniority levels (p<0.05).' if p4 < 0.05 else 'No significant salary difference across seniority levels (p≥0.05).'
print(f'  Conclusion: {conclusion4}')

hyp_results['H4_seniority_levels'] = {
    'junior_median': float(junior.median()),
    'mid_median': float(mid.median()),
    'senior_median': float(senior.median()),
    'h_statistic': float(h4), 'p_value': float(p4),
    'significant': bool(p4 < 0.05), 'conclusion': conclusion4,
}"""),

    code("code-s5-save",
"""# Save results ─────────────────────────────────────────────────
out_path = RESULTS / 'hypothesis_results.json'
with open(out_path, 'w') as f:
    json.dump(hyp_results, f, indent=2)
print(f'Section 5 complete — results saved to {out_path}')"""),
]

# ── Assemble and write ─────────────────────────────────────────────────────────
section_cells = [S1, S2, S3, S4, S5]
cells = S0[:]
for i in range(stage):
    cells.extend(section_cells[i])

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
        "path": "/app",
    },
    "cells": cells,
}

out = Path("notebooks/analysis.ipynb")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {out}  (stage={stage}, cells={len(cells)})")
