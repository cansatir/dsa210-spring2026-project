"""
Programmatically creates all 4 development notebooks using nbformat.
Run: python scripts/build_notebooks.py
"""
import pathlib
import nbformat

OUT = pathlib.Path("notebooks/02_development")
OUT.mkdir(parents=True, exist_ok=True)


def nb(cells):
    n = nbformat.v4.new_notebook()
    n.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    n.cells = cells
    return n


def md(src):
    return nbformat.v4.new_markdown_cell(src)


def code(src):
    return nbformat.v4.new_code_cell(src)


# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 01 — Data Exploration
# ══════════════════════════════════════════════════════════════════════════════

NB01 = nb([
    md("# 01 — Data Exploration\n\nLoad and explore SO 2025, SO 2023, and aijobs.net datasets. "
       "Filter to full-time employed, convert salaries to USD, save processed versions."),

    code("""\
import os, warnings
warnings.filterwarnings("ignore")
# ensure kernel runs from repo root regardless of notebook location
os.chdir("/app")
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_RAW = pathlib.Path("data/raw")
DATA_PROC = pathlib.Path("data/processed")
DATA_PROC.mkdir(parents=True, exist_ok=True)
"""),

    md("## Stack Overflow Developer Survey 2025"),

    code("""\
so25 = pd.read_csv(DATA_RAW / "stackoverflow_2025.csv", low_memory=False)
print(f"Shape: {so25.shape}")
print(f"\\nKey columns present: {[c for c in ['DevType','LanguageHaveWorkedWith','DatabaseHaveWorkedWith','WebframeHaveWorkedWith','WorkExp','EdLevel','Country','Employment','RemoteWork','CompTotal','Currency','JobSat','AISelect'] if c in so25.columns]}")
print(f"\\nMissing values (key cols):")
key_cols = ['DevType','LanguageHaveWorkedWith','WorkExp','EdLevel','Country','Employment','RemoteWork','CompTotal','Currency','JobSat','AISelect']
for c in key_cols:
    if c in so25.columns:
        pct = 100 * so25[c].isna().mean()
        print(f"  {c}: {so25[c].isna().sum():,} ({pct:.1f}%)")
"""),

    code("""\
print("Employment values:")
print(so25["Employment"].value_counts().to_string())
print("\\nCurrency top 10:")
print(so25["Currency"].value_counts().head(10).to_string())
print("\\nJobSat distribution:")
print(so25["JobSat"].value_counts().sort_index().to_string())
"""),

    code("""\
# Filter to full-time employed only
so25_emp = so25[so25["Employment"] == "Employed"].copy()
print(f"Full-time employed: {len(so25_emp):,} rows (of {len(so25):,})")

# Currency → USD conversion (approximate 2025 exchange rates)
RATES = {
    "USD United States dollar": 1.000,
    "EUR European Euro": 1.080,
    "GBP Pound sterling": 1.270,
    "INR Indian rupee": 0.012,
    "CAD Canadian dollar": 0.740,
    "AUD Australian dollar": 0.650,
    "BRL Brazilian real": 0.190,
    "CHF Swiss franc": 1.130,
    "SEK Swedish krona": 0.095,
    "NOK Norwegian krone": 0.094,
    "DKK Danish krone": 0.145,
    "PLN Polish zloty": 0.250,
    "CZK Czech koruna": 0.044,
    "HUF Hungarian forint": 0.0028,
    "TRY Turkish lira": 0.030,
    "MXN Mexican peso": 0.058,
    "SGD Singapore dollar": 0.740,
    "ILS Israeli new shekel": 0.270,
    "UAH Ukrainian hryvnia": 0.024,
    "JPY Japanese yen": 0.0067,
    "KRW South Korean won": 0.00072,
    "ZAR South African rand": 0.055,
    "RON Romanian leu": 0.220,
    "BGN Bulgarian lev": 0.550,
}
so25_emp["rate"] = so25_emp["Currency"].map(RATES)
so25_emp["CompUSD"] = so25_emp["CompTotal"] * so25_emp["rate"]

# Remove missing salary, no recognized currency, and outliers
so25_clean = so25_emp[
    so25_emp["CompUSD"].notna() &
    so25_emp["WorkExp"].notna() &
    so25_emp["DevType"].notna() &
    (so25_emp["CompUSD"] >= 10_000) &
    (so25_emp["CompUSD"] <= 400_000)
].copy()

print(f"After cleaning: {len(so25_clean):,} rows with valid salary + experience + role")
print(f"  Median salary: ${so25_clean['CompUSD'].median():,.0f}")
print(f"  Mean salary:   ${so25_clean['CompUSD'].mean():,.0f}")
print(f"  Salary range:  ${so25_clean['CompUSD'].min():,.0f} – ${so25_clean['CompUSD'].max():,.0f}")
"""),

    md("## Stack Overflow Developer Survey 2023"),

    code("""\
so23 = pd.read_csv(DATA_RAW / "stackoverflow_2023.csv", low_memory=False)
print(f"Shape: {so23.shape}")
print(f"\\nKey columns:")
key23 = ["DevType","LanguageHaveWorkedWith","WorkExp","EdLevel","Country","Employment","RemoteWork","ConvertedCompYearly","AIBen","AISelect"]
for c in key23:
    if c in so23.columns:
        pct = 100 * so23[c].isna().mean()
        print(f"  {c}: {so23[c].isna().sum():,} missing ({pct:.1f}%)")
    else:
        print(f"  {c}: NOT PRESENT")
"""),

    code("""\
so23_emp = so23[so23["Employment"].str.contains("Employed", na=False)].copy()
print(f"Full-time + contractor employed: {len(so23_emp):,} rows")

# SO 2023 already has ConvertedCompYearly (USD)
so23_emp = so23_emp.rename(columns={"ConvertedCompYearly": "CompUSD"})

so23_clean = so23_emp[
    so23_emp["CompUSD"].notna() &
    so23_emp["WorkExp"].notna() &
    so23_emp["DevType"].notna() &
    (so23_emp["CompUSD"] >= 10_000) &
    (so23_emp["CompUSD"] <= 400_000)
].copy()

print(f"After cleaning: {len(so23_clean):,} rows")
print(f"  Median salary: ${so23_clean['CompUSD'].median():,.0f}")
print(f"AIBen values:")
if "AIBen" in so23_clean.columns:
    print(so23_clean["AIBen"].value_counts().to_string())
"""),

    md("## aijobs.net Salaries (2020–2025)"),

    code("""\
aijobs = pd.read_csv(DATA_RAW / "aijobs_salaries.csv")
print(f"Shape: {aijobs.shape}")
print(f"\\nDtypes:\\n{aijobs.dtypes.to_string()}")
print(f"\\nMissing values:\\n{aijobs.isna().sum().to_string()}")
print(f"\\nWork years: {sorted(aijobs['work_year'].unique())}")
print(f"\\nExperience levels: {aijobs['experience_level'].value_counts().to_string()}")
print(f"\\nTop 15 job titles:")
print(aijobs["job_title"].value_counts().head(15).to_string())
print(f"\\nMedian salary by year:")
print(aijobs.groupby("work_year")["salary_in_usd"].median().to_string())
"""),

    md("## Salary Distribution Visualizations"),

    code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(so25_clean["CompUSD"] / 1_000, bins=40, color="#4C72B0", edgecolor="white", linewidth=0.4)
axes[0].set_title("SO 2025 Salaries (USD, $K)")
axes[0].set_xlabel("Annual Salary ($K)")
axes[0].set_ylabel("Respondents")

axes[1].hist(so23_clean["CompUSD"] / 1_000, bins=40, color="#55A868", edgecolor="white", linewidth=0.4)
axes[1].set_title("SO 2023 Salaries (USD, $K)")
axes[1].set_xlabel("Annual Salary ($K)")

# aijobs: full-time only
ai_ft = aijobs[aijobs["employment_type"] == "FT"]
axes[2].hist(ai_ft["salary_in_usd"] / 1_000, bins=40, color="#DD8452", edgecolor="white", linewidth=0.4)
axes[2].set_title("aijobs.net Salaries (USD, $K)")
axes[2].set_xlabel("Annual Salary ($K)")

plt.tight_layout()
plt.savefig("figures/fig_data_exploration_salaries.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved salary distribution figure")
"""),

    code("""\
# Top DevTypes with median salary (SO 2025)
so25_clean["DevType_primary"] = so25_clean["DevType"].str.split(";").str[0].str.strip()
role_salary = (
    so25_clean.groupby("DevType_primary")["CompUSD"]
    .agg(median="median", count="count")
    .query("count >= 20")
    .sort_values("median", ascending=True)
    .tail(15)
)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(role_salary.index, role_salary["median"] / 1_000, color="#4C72B0")
ax.set_xlabel("Median Salary ($K)")
ax.set_title("Median Salary by Role — SO 2025 (top 15 roles, ≥20 respondents)")
plt.tight_layout()
plt.savefig("figures/fig_salary_by_role_so25.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved role salary figure")
"""),

    md("## Save Processed Datasets"),

    code("""\
so25_clean.to_parquet(DATA_PROC / "so25_processed.parquet", index=False)
so23_clean.to_parquet(DATA_PROC / "so23_processed.parquet", index=False)
aijobs.to_parquet(DATA_PROC / "aijobs_processed.parquet", index=False)

print(f"Saved so25_processed.parquet  — {len(so25_clean):,} rows")
print(f"Saved so23_processed.parquet  — {len(so23_clean):,} rows")
print(f"Saved aijobs_processed.parquet — {len(aijobs):,} rows")
"""),
])

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 02 — SO Salary Model
# ══════════════════════════════════════════════════════════════════════════════

NB02 = nb([
    md("# 02 — Developer-Profile Salary Model\n\nTrain XGBoost on SO 2025 survey data. "
       "Compute actual salary ranges (p25/median/p75) per profile bucket."),

    code("""\
import os, warnings
warnings.filterwarnings("ignore")
os.chdir("/app")
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import joblib
from xgboost import XGBRegressor

DATA_PROC = pathlib.Path("data/processed")
MODELS = pathlib.Path("models")
MODELS.mkdir(exist_ok=True)
"""),

    md("## Feature Engineering"),

    code("""\
so25 = pd.read_parquet(DATA_PROC / "so25_processed.parquet")
print(f"Loaded SO 2025: {so25.shape}")

# Primary role (first DevType)
so25["DevType_primary"] = so25["DevType"].str.split(";").str[0].str.strip()

# Top 15 roles (enough samples per role)
top_roles = (
    so25.groupby("DevType_primary")["CompUSD"]
    .count()
    .nlargest(15)
    .index.tolist()
)
so25 = so25[so25["DevType_primary"].isin(top_roles)].copy()
print(f"After role filter: {len(so25):,} rows, {len(top_roles)} roles")
print("Roles:", top_roles)
"""),

    code("""\
# Education level → ordinal
ED_ORDER = {
    "Primary/elementary school": 0,
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 1,
    "Some college/university study without earning a degree": 2,
    "Associate degree (A.A., A.S., etc.)": 2,
    "Bachelor's degree (B.A., B.S., B.Eng., etc.)": 3,
    "Master's degree (M.A., M.S., M.Eng., MBA, etc.)": 4,
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 5,
    "Other (please specify):": 2,
}
so25["ed_ord"] = so25["EdLevel"].map(ED_ORDER).fillna(2).astype(int)

def simplify_ed(ed):
    if pd.isna(ed): return "Other"
    ed = str(ed)
    if "Bachelor" in ed: return "Bachelor's"
    if "Master" in ed or "Ph.D" in ed or "Professional" in ed: return "Graduate+"
    return "No Degree"

so25["ed_level"] = so25["EdLevel"].apply(simplify_ed)

# Experience band
def exp_band(y):
    if y <= 2:   return "0-2"
    if y <= 5:   return "3-5"
    if y <= 10:  return "6-10"
    if y <= 20:  return "11-20"
    return "20+"

so25["exp_band"] = so25["WorkExp"].apply(exp_band)

# Country grouping — top 20 countries, rest → "Other"
top_countries = so25["Country"].value_counts().head(20).index.tolist()
so25["country_grp"] = so25["Country"].where(so25["Country"].isin(top_countries), "Other")

print("Experience band distribution:")
print(so25["exp_band"].value_counts().to_string())
print("\\nEducation distribution:")
print(so25["ed_level"].value_counts().to_string())
print("\\nTop country groups:")
print(so25["country_grp"].value_counts().head(10).to_string())
"""),

    code("""\
# Top 30 languages as binary features
from collections import Counter

all_langs = []
for s in so25["LanguageHaveWorkedWith"].dropna():
    all_langs.extend(s.split(";"))
top_langs = [l.strip() for l, _ in Counter(all_langs).most_common(30)]
print("Top 30 languages:", top_langs)

def col_name(lang):
    return ("lang_" + lang.lower()
            .replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
            .replace("#", "sharp").replace("+", "plus").replace(".", "dot")
            .replace("-", "_").replace(",", "").replace("__", "_"))

lang_cols = []
for lang in top_langs:
    c = col_name(lang)
    so25[c] = so25["LanguageHaveWorkedWith"].str.contains(lang, na=False, regex=False).astype(np.int8)
    lang_cols.append(c)

print(f"\\nCreated {len(lang_cols)} language feature columns")
"""),

    code("""\
# Label encode categoricals for XGBoost
le_role    = LabelEncoder().fit(so25["DevType_primary"])
le_country = LabelEncoder().fit(so25["country_grp"])

so25["role_enc"]    = le_role.transform(so25["DevType_primary"])
so25["country_enc"] = le_country.transform(so25["country_grp"])

feature_cols = ["WorkExp", "ed_ord", "role_enc", "country_enc"] + lang_cols
X = so25[feature_cols].values
y = so25["CompUSD"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}  Test: {X_test.shape}")
"""),

    md("## Train XGBoost"),

    code("""\
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
ss_res = np.sum((y_test - y_pred) ** 2)
ss_tot = np.sum((y_test - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print(f"RMSE: ${rmse:,.0f}")
print(f"MAE:  ${mae:,.0f}")
print(f"R²:   {r2:.4f}")
"""),

    code("""\
# Feature importance plot
importances = pd.Series(model.feature_importances_, index=feature_cols)
top_imp = importances.nlargest(20)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top_imp.index[::-1], top_imp.values[::-1], color="#4C72B0")
ax.set_xlabel("Feature Importance (XGBoost gain)")
ax.set_title("Top 20 Features — SO 2025 Salary Model")
plt.tight_layout()
plt.savefig("figures/fig_so_model_importance.png", dpi=120, bbox_inches="tight")
plt.show()
print("Saved feature importance figure")
"""),

    code("""\
# Actual vs predicted scatter
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(y_test / 1_000, y_pred / 1_000, alpha=0.15, s=10, color="#4C72B0")
lim = max(y_test.max(), y_pred.max()) / 1_000
ax.plot([0, lim], [0, lim], "r--", linewidth=1.5)
ax.set_xlabel("Actual Salary ($K)")
ax.set_ylabel("Predicted Salary ($K)")
ax.set_title(f"SO 2025 Salary Model — R²={r2:.3f}, RMSE=${rmse/1000:.1f}K")
plt.tight_layout()
plt.savefig("figures/fig_so_model_pred.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("## Salary Ranges Lookup Table"),

    code("""\
# Compute median + p25 + p75 per profile bucket
# Bucket: DevType_primary × country_grp × exp_band × ed_level
agg = (
    so25.groupby(["DevType_primary", "country_grp", "exp_band", "ed_level"])["CompUSD"]
    .agg(
        median_salary=lambda x: round(x.median(), 0),
        p25_salary=lambda x: round(x.quantile(0.25), 0),
        p75_salary=lambda x: round(x.quantile(0.75), 0),
        count="count",
    )
    .reset_index()
)
agg = agg.rename(columns={"DevType_primary": "role", "country_grp": "country"})
print(f"Salary ranges table: {len(agg):,} buckets")
print(f"Buckets with count < 30: {(agg['count'] < 30).sum():,}")
print(f"Buckets with count >= 30: {(agg['count'] >= 30).sum():,}")
print("\\nSample rows:")
print(agg.sort_values("count", ascending=False).head(8).to_string())
"""),

    code("""\
# Also compute role-level summary (for fallback)
role_summary = (
    so25.groupby("DevType_primary")["CompUSD"]
    .agg(
        median_salary=lambda x: round(x.median(), 0),
        p25_salary=lambda x: round(x.quantile(0.25), 0),
        p75_salary=lambda x: round(x.quantile(0.75), 0),
        count="count",
    )
    .reset_index()
    .rename(columns={"DevType_primary": "role"})
)
role_summary["country"] = "All"
role_summary["exp_band"] = "All"
role_summary["ed_level"] = "All"

# Save both
agg.to_parquet(DATA_PROC / "salary_ranges.parquet", index=False)
role_summary.to_parquet(DATA_PROC / "salary_ranges_by_role.parquet", index=False)
joblib.dump(model, MODELS / "xgboost_so_salary.pkl")

print("Saved salary_ranges.parquet")
print("Saved salary_ranges_by_role.parquet")
print("Saved xgboost_so_salary.pkl")
"""),
])

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 03 — Job Satisfaction
# ══════════════════════════════════════════════════════════════════════════════

NB03 = nb([
    md("# 03 — Job Satisfaction Analysis\n\n"
       "Compare job satisfaction across roles, countries, work arrangements, "
       "pay levels, and AI tool usage using SO 2025 (JobSat) and SO 2023 (AIBen)."),

    code("""\
import os, warnings
warnings.filterwarnings("ignore")
os.chdir("/app")
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PROC = pathlib.Path("data/processed")
"""),

    code("""\
so25 = pd.read_parquet(DATA_PROC / "so25_processed.parquet")
so23 = pd.read_parquet(DATA_PROC / "so23_processed.parquet")

# Derive primary role
so25["DevType_primary"] = so25["DevType"].str.split(";").str[0].str.strip()
so23["DevType_primary"] = so23["DevType"].str.split(";").str[0].str.strip()

print(f"SO 2025: {so25.shape} — JobSat non-null: {so25['JobSat'].notna().sum():,}")
print(f"SO 2023: {so23.shape} — AIBen non-null: {so23['AIBen'].notna().sum():,}")
print("\\nJobSat scale (SO 2025):", sorted(so25["JobSat"].dropna().unique()))
"""),

    md("## Q1: Which roles have highest job satisfaction?"),

    code("""\
# SO 2025 JobSat by role (top roles with ≥30 respondents)
role_sat = (
    so25[so25["JobSat"].notna()]
    .groupby("DevType_primary")["JobSat"]
    .agg(mean="mean", median="median", count="count")
    .query("count >= 30")
    .sort_values("median", ascending=True)
    .round(2)
)
print("Job satisfaction by role (SO 2025, JobSat 1-10):")
print(role_sat.to_string())

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(role_sat.index, role_sat["median"], color="#4C72B0")
ax.axvline(so25["JobSat"].median(), color="#DD4444", linestyle="--",
           label=f"Overall median: {so25['JobSat'].median():.1f}")
ax.set_xlabel("Median Job Satisfaction (1-10)")
ax.set_title("Median Job Satisfaction by Role — SO 2025")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig_jobsat_by_role.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("## Q2: Do higher paid developers report more satisfaction?"),

    code("""\
so25_sat = so25[so25["JobSat"].notna() & so25["CompUSD"].notna()].copy()

# Salary bands
so25_sat["comp_band"] = pd.cut(
    so25_sat["CompUSD"],
    bins=[0, 50_000, 100_000, 150_000, 200_000, 400_001],
    labels=["<$50K", "$50-100K", "$100-150K", "$150-200K", ">$200K"],
)
band_sat = (
    so25_sat.groupby("comp_band", observed=True)["JobSat"]
    .agg(mean="mean", median="median", count="count")
    .round(2)
)
print("Job satisfaction by salary band:")
print(band_sat.to_string())

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(band_sat.index, band_sat["median"], color=["#6baed6","#3182bd","#2171b5","#08519c","#08306b"])
ax.set_xlabel("Salary Band")
ax.set_ylabel("Median Job Satisfaction (1-10)")
ax.set_title("Job Satisfaction vs Salary Band — SO 2025")
plt.tight_layout()
plt.savefig("figures/fig_jobsat_by_salary.png", dpi=120, bbox_inches="tight")
plt.show()

corr = so25_sat[["CompUSD","JobSat"]].corr().iloc[0,1]
print(f"\\nPearson correlation CompUSD ~ JobSat: {corr:.3f}")
"""),

    md("## Q3: Do AI tool users report higher satisfaction?"),

    code("""\
# SO 2025 AISelect categories
print("AISelect values (SO 2025):")
print(so25["AISelect"].value_counts().to_string())

ai_sat = (
    so25[so25["AISelect"].notna() & so25["JobSat"].notna()]
    .groupby("AISelect")["JobSat"]
    .agg(mean="mean", median="median", count="count")
    .round(2)
)
print("\\nJob satisfaction by AI tool usage:")
print(ai_sat.to_string())

fig, ax = plt.subplots(figsize=(9, 4))
labels = [l[:40] for l in ai_sat.index]
ax.bar(range(len(labels)), ai_sat["median"], color="#55A868")
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Median Job Satisfaction (1-10)")
ax.set_title("Job Satisfaction by AI Tool Usage — SO 2025")
plt.tight_layout()
plt.savefig("figures/fig_jobsat_ai_usage.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    code("""\
# SO 2023: AIBen (trust in AI)
if "AIBen" in so23.columns:
    print("AIBen values (SO 2023):")
    print(so23["AIBen"].value_counts().to_string())
else:
    print("AIBen not in SO 2023 processed data")
"""),

    md("## Satisfaction by Country and Remote Work"),

    code("""\
# By country (top 10 by response count)
top_countries_sat = so25["Country"].value_counts().head(10).index.tolist()
country_sat = (
    so25[so25["JobSat"].notna() & so25["Country"].isin(top_countries_sat)]
    .groupby("Country")["JobSat"]
    .agg(mean="mean", median="median", count="count")
    .sort_values("median", ascending=False)
    .round(2)
)
print("Satisfaction by country (top 10):")
print(country_sat.to_string())
"""),

    code("""\
# By RemoteWork
remote_sat = (
    so25[so25["JobSat"].notna() & so25["RemoteWork"].notna()]
    .groupby("RemoteWork")["JobSat"]
    .agg(mean="mean", median="median", count="count")
    .sort_values("median", ascending=False)
    .round(2)
)
print("Satisfaction by work arrangement:")
print(remote_sat.to_string())

fig, ax = plt.subplots(figsize=(9, 4))
remote_sat_plot = remote_sat.sort_values("median")
ax.barh(range(len(remote_sat_plot)), remote_sat_plot["median"], color="#4C72B0")
ax.set_yticks(range(len(remote_sat_plot)))
ax.set_yticklabels([l[:45] for l in remote_sat_plot.index], fontsize=8)
ax.set_xlabel("Median Job Satisfaction (1-10)")
ax.set_title("Satisfaction by Work Arrangement — SO 2025")
plt.tight_layout()
plt.savefig("figures/fig_jobsat_remote.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("## Save Summary"),

    code("""\
# Combine into one summary table (role-level stats)
summary_role = (
    so25[so25["JobSat"].notna()]
    .groupby("DevType_primary")
    .agg(
        mean_jobsat=("JobSat", "mean"),
        median_jobsat=("JobSat", "median"),
        count_jobsat=("JobSat", "count"),
        median_salary=("CompUSD", lambda x: x.median() if x.notna().any() else np.nan),
    )
    .reset_index()
    .rename(columns={"DevType_primary": "role"})
    .round(2)
)

# AI usage summary
ai_usage_summary = (
    so25[so25["AISelect"].notna() & so25["JobSat"].notna()]
    .groupby("AISelect")
    .agg(
        mean_jobsat=("JobSat", "mean"),
        median_jobsat=("JobSat", "median"),
        count=("JobSat", "count"),
    )
    .reset_index()
    .rename(columns={"AISelect": "ai_usage"})
    .round(2)
)

# Remote work summary
remote_summary = (
    so25[so25["JobSat"].notna() & so25["RemoteWork"].notna()]
    .groupby("RemoteWork")
    .agg(
        mean_jobsat=("JobSat", "mean"),
        median_jobsat=("JobSat", "median"),
        count=("JobSat", "count"),
    )
    .reset_index()
    .rename(columns={"RemoteWork": "remote_work"})
    .round(2)
)

# SO 2023 AIBen
if "AIBen" in so23.columns:
    aiben_summary = (
        so23[so23["AIBen"].notna()]
        .groupby("AIBen")
        .agg(
            count=("AIBen", "count"),
            median_salary=("CompUSD", "median"),
        )
        .reset_index()
        .rename(columns={"AIBen": "ai_ben"})
    )
else:
    aiben_summary = pd.DataFrame(columns=["ai_ben","count","median_salary"])

# Save
summary_role.to_parquet(DATA_PROC / "job_satisfaction_summary.parquet", index=False)
ai_usage_summary.to_parquet(DATA_PROC / "ai_usage_satisfaction.parquet", index=False)
remote_summary.to_parquet(DATA_PROC / "remote_satisfaction.parquet", index=False)
aiben_summary.to_parquet(DATA_PROC / "aiben_summary.parquet", index=False)

print(f"Saved job_satisfaction_summary.parquet — {len(summary_role)} roles")
print(f"Saved ai_usage_satisfaction.parquet — {len(ai_usage_summary)} AI categories")
print(f"Saved remote_satisfaction.parquet — {len(remote_summary)} work types")
print(f"Saved aiben_summary.parquet — {len(aiben_summary)} AIBen categories")
"""),
])

# ══════════════════════════════════════════════════════════════════════════════
# NOTEBOOK 04 — Salary Trends
# ══════════════════════════════════════════════════════════════════════════════

NB04 = nb([
    md("# 04 — Salary Trends 2020–2025\n\n"
       "Use aijobs.net to show salary trends, remote ratio evolution, "
       "and company size effects across years."),

    code("""\
import os, warnings
warnings.filterwarnings("ignore")
os.chdir("/app")
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PROC = pathlib.Path("data/processed")
"""),

    code("""\
aijobs = pd.read_parquet(DATA_PROC / "aijobs_processed.parquet")
print(f"Shape: {aijobs.shape}")
print(f"Years: {sorted(aijobs['work_year'].unique())}")
print(f"Rows per year:\\n{aijobs['work_year'].value_counts().sort_index().to_string()}")
print(f"\\nTop 15 job titles:")
print(aijobs["job_title"].value_counts().head(15).to_string())
print(f"\\nExperience levels: {aijobs['experience_level'].unique()}")
print(f"Remote ratio values: {sorted(aijobs['remote_ratio'].unique())}")
print(f"Company size values: {aijobs['company_size'].unique()}")
"""),

    md("## Median Salary by Year for Top 10 Roles"),

    code("""\
# Focus on full-time and sufficient history
ai_ft = aijobs[aijobs["employment_type"] == "FT"].copy()

# Top 10 roles by total count (across all years)
top10_roles = ai_ft["job_title"].value_counts().head(10).index.tolist()
print("Top 10 roles:", top10_roles)

trend_data = (
    ai_ft[ai_ft["job_title"].isin(top10_roles)]
    .groupby(["work_year", "job_title"])["salary_in_usd"]
    .median()
    .reset_index()
    .rename(columns={"salary_in_usd": "median_salary"})
)

fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.tab10(np.linspace(0, 1, len(top10_roles)))
for i, role in enumerate(top10_roles):
    d = trend_data[trend_data["job_title"] == role].sort_values("work_year")
    if len(d) >= 3:
        ax.plot(d["work_year"], d["median_salary"] / 1_000, marker="o", label=role[:30], color=colors[i])
ax.set_xlabel("Year")
ax.set_ylabel("Median Annual Salary ($K)")
ax.set_title("Salary Trend by Role — aijobs.net 2020-2025")
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig_salary_trends.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("## Remote Work Ratio Trend"),

    code("""\
remote_trend = (
    ai_ft.groupby("work_year")["remote_ratio"]
    .mean()
    .reset_index()
    .rename(columns={"remote_ratio": "avg_remote_ratio"})
)
print("Remote ratio trend:")
print(remote_trend.to_string())

# Also show breakdown by remote_ratio category
remote_cat_trend = (
    ai_ft.groupby(["work_year", "remote_ratio"])
    .size()
    .reset_index(name="count")
)
total_per_year = ai_ft.groupby("work_year").size().reset_index(name="total")
remote_cat_trend = remote_cat_trend.merge(total_per_year, on="work_year")
remote_cat_trend["pct"] = 100 * remote_cat_trend["count"] / remote_cat_trend["total"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(remote_trend["work_year"], remote_trend["avg_remote_ratio"], marker="o", color="#4C72B0")
ax1.set_title("Average Remote Ratio by Year")
ax1.set_xlabel("Year")
ax1.set_ylabel("Average Remote Ratio (0=onsite, 100=fully remote)")

remote_labels = {0: "Onsite (0%)", 50: "Hybrid (50%)", 100: "Fully Remote (100%)"}
cat_colors = {0: "#DD4444", 50: "#DDAA00", 100: "#44AA44"}
for ratio_val, color in cat_colors.items():
    d = remote_cat_trend[remote_cat_trend["remote_ratio"] == ratio_val].sort_values("work_year")
    ax2.plot(d["work_year"], d["pct"], marker="o", label=remote_labels[ratio_val], color=color)
ax2.set_title("Work Arrangement Mix by Year (%)")
ax2.set_xlabel("Year")
ax2.set_ylabel("% of Job Postings")
ax2.legend()
plt.tight_layout()
plt.savefig("figures/fig_remote_trends.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("## Company Size Effect on Salary"),

    code("""\
size_trend = (
    ai_ft.groupby(["work_year", "company_size"])["salary_in_usd"]
    .median()
    .reset_index()
    .rename(columns={"salary_in_usd": "median_salary"})
)
size_labels = {"S": "Small (<50)", "M": "Medium (50-250)", "L": "Large (>250)"}
size_colors = {"S": "#4C72B0", "M": "#55A868", "L": "#DD8452"}

fig, ax = plt.subplots(figsize=(9, 5))
for size, color in size_colors.items():
    d = size_trend[size_trend["company_size"] == size].sort_values("work_year")
    ax.plot(d["work_year"], d["median_salary"] / 1_000, marker="o",
            label=size_labels.get(size, size), color=color, linewidth=2)
ax.set_xlabel("Year")
ax.set_ylabel("Median Salary ($K)")
ax.set_title("Median Salary by Company Size — aijobs.net")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig_company_size_trend.png", dpi=120, bbox_inches="tight")
plt.show()

print("Company size stats by year:")
print(size_trend.pivot(index="work_year", columns="company_size", values="median_salary")
      .div(1000).round(1).to_string())
"""),

    md("## Overall Salary Trend (All Roles)"),

    code("""\
overall_trend = (
    ai_ft.groupby("work_year")["salary_in_usd"]
    .agg(median="median", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75), count="count")
    .reset_index()
)
print("Overall salary trend:")
print(overall_trend.to_string())

fig, ax = plt.subplots(figsize=(9, 5))
ax.fill_between(overall_trend["work_year"],
                overall_trend["p25"] / 1_000,
                overall_trend["p75"] / 1_000,
                alpha=0.25, color="#4C72B0", label="P25–P75 band")
ax.plot(overall_trend["work_year"], overall_trend["median"] / 1_000,
        marker="o", color="#4C72B0", linewidth=2, label="Median")
ax.set_xlabel("Year")
ax.set_ylabel("Salary ($K)")
ax.set_title("Overall Salary Trend (All Roles) — aijobs.net")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig_overall_trend.png", dpi=120, bbox_inches="tight")
plt.show()
"""),

    md("## Save Results"),

    code("""\
# Save salary_trends.parquet
salary_trends = trend_data.copy()
salary_trends.to_parquet(DATA_PROC / "salary_trends.parquet", index=False)

# Save remote trend
remote_trend.to_parquet(DATA_PROC / "remote_trend.parquet", index=False)

# Save company size trend
size_trend.to_parquet(DATA_PROC / "company_size_trend.parquet", index=False)

# Save overall trend
overall_trend.to_parquet(DATA_PROC / "overall_salary_trend.parquet", index=False)

print(f"Saved salary_trends.parquet — {len(salary_trends)} rows ({len(top10_roles)} roles)")
print(f"Saved remote_trend.parquet")
print(f"Saved company_size_trend.parquet")
print(f"Saved overall_salary_trend.parquet")
"""),
])

# ── Write notebooks to disk ───────────────────────────────────────────────────

for path, notebook in [
    (OUT / "01_data_exploration.ipynb", NB01),
    (OUT / "02_so_salary_model.ipynb",  NB02),
    (OUT / "03_job_satisfaction.ipynb", NB03),
    (OUT / "04_salary_trends.ipynb",    NB04),
]:
    with open(path, "w") as fh:
        nbformat.write(notebook, fh)
    print(f"Created {path}")

print("\nDone — 4 notebooks created in notebooks/02_development/")
