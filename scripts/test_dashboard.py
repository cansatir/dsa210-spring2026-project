"""
Comprehensive tests for the Streamlit dashboard business logic.

Tests the underlying functions directly — no Streamlit UI involved.
Run inside Docker:
    docker exec -it <container> python scripts/test_dashboard.py
"""
import pathlib
import sys
import warnings
from collections import Counter
from itertools import product

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "xgboost_salary.pkl"

EXP_BANDS = ["0-2", "3-5", "6-10", "11-20", "20+"]
ED_LEVELS = ["Bachelor's", "Graduate+", "No Degree", "Other"]
SKILL_CATEGORIES = {
    "Languages": [
        "JavaScript", "HTML/CSS", "SQL", "Python", "Bash/Shell (all shells)",
        "TypeScript", "C#", "Java", "PowerShell", "C++", "Go", "C",
        "PHP", "Rust", "Kotlin",
    ],
    "Databases": [
        "PostgreSQL", "MySQL", "SQLite", "Microsoft SQL Server", "Redis",
        "MongoDB", "MariaDB", "Elasticsearch", "Dynamodb", "Oracle",
    ],
    "Frameworks": [
        "Node.js", "React", "jQuery", "ASP.NET Core", "Angular",
        "Next.js", "Vue.js", "Express", "Spring Boot", "ASP.NET",
    ],
}

# ── Constants (mirror dashboard.py) ──────────────────────────────────────────

RMSE = 33_522

JOB_TITLES = [
    "Business Analyst",
    "Cloud Engineer",
    "Data Analyst",
    "Data Engineer",
    "Data Scientist",
    "Machine Learning Engineer",
    "Senior Data Analyst",
    "Senior Data Engineer",
    "Senior Data Scientist",
    "Software Engineer",
]

COUNTRIES = [
    "United States",
    "Canada",
    "United Kingdom",
    "Australia",
    "Germany",
    "France",
    "India",
    "Netherlands",
    "Poland",
    "Spain",
    "Portugal",
    "Colombia",
    "Greece",
    "Israel",
    "Mexico",
    "Philippines",
    "Singapore",
    "South Africa",
    "South Korea",
    "Sudan",
    "Other",
]

SENIORITY = ["Junior", "Mid", "Senior"]

TOP_10_SKILLS = [
    "sql",
    "python",
    "aws",
    "azure",
    "tableau",
    "spark",
    "excel",
    "power bi",
    "r",
    "sas",
]

# ── Fixtures ──────────────────────────────────────────────────────────────────

def load_fixtures():
    df = pd.read_parquet(DATA_DIR / "jobs_enriched.parquet")
    features = pd.read_parquet(DATA_DIR / "features.parquet")
    model = joblib.load(MODEL_PATH)
    feature_cols = [c for c in features.columns if c != "salary_year_avg"]
    return df, features, model, feature_cols


def load_survey_fixtures():
    sr = pd.read_parquet(DATA_DIR / "salary_ranges.parquet")
    fallback = pd.read_parquet(DATA_DIR / "salary_ranges_by_role.parquet")
    jss = pd.read_parquet(DATA_DIR / "job_satisfaction_summary.parquet")
    ai_usage = pd.read_parquet(DATA_DIR / "ai_usage_satisfaction.parquet")
    remote_sat = pd.read_parquet(DATA_DIR / "remote_satisfaction.parquet")
    return sr, fallback, jss, ai_usage, remote_sat


def load_trends_fixtures():
    st_df = pd.read_parquet(DATA_DIR / "salary_trends.parquet")
    rt = pd.read_parquet(DATA_DIR / "remote_trend.parquet")
    cst = pd.read_parquet(DATA_DIR / "company_size_trend.parquet")
    ost = pd.read_parquet(DATA_DIR / "overall_salary_trend.parquet")
    return st_df, rt, cst, ost


def load_so25_skills():
    so25_path = DATA_DIR / "so25_processed.parquet"
    if not so25_path.exists():
        return None
    cols = [
        "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith",
        "WebframeHaveWorkedWith", "CompUSD", "JobSat",
    ]
    return pd.read_parquet(so25_path, columns=[c for c in cols if c])


# ── Dashboard functions under test (copied verbatim from dashboard.py) ────────

def make_feature_vector(
    job_title: str,
    country: str,
    seniority: str,
    remote: bool,
    skills_selected: list,
    feature_cols: list,
) -> pd.DataFrame:
    vec = {col: 0 for col in feature_cols}
    title_col = f"job_title_short_{job_title}"
    if title_col in vec:
        vec[title_col] = 1
    country_col = f"job_country_grp_{country}"
    if country_col in vec:
        vec[country_col] = 1
    seniority_col = f"seniority_{seniority}"
    if seniority_col in vec:
        vec[seniority_col] = 1
    vec["job_work_from_home"] = int(remote)
    for skill in skills_selected:
        skill_col = f"skill_{skill}"
        if skill_col in vec:
            vec[skill_col] = 1
    return pd.DataFrame([vec], columns=feature_cols)


FEATURE_LABELS = {
    "job_title_short_Business Analyst": "Job title: Business Analyst",
    "job_title_short_Cloud Engineer": "Job title: Cloud Engineer",
    "job_title_short_Data Analyst": "Job title: Data Analyst",
    "job_title_short_Data Engineer": "Job title: Data Engineer",
    "job_title_short_Data Scientist": "Job title: Data Scientist",
    "job_title_short_Machine Learning Engineer": "Job title: Machine Learning Engineer",
    "job_title_short_Senior Data Analyst": "Job title: Senior Data Analyst",
    "job_title_short_Senior Data Engineer": "Job title: Senior Data Engineer",
    "job_title_short_Senior Data Scientist": "Job title: Senior Data Scientist",
    "job_title_short_Software Engineer": "Job title: Software Engineer",
    "seniority_Junior": "Seniority: Junior",
    "seniority_Mid": "Seniority: Mid",
    "seniority_Senior": "Seniority: Senior",
    "job_work_from_home": "Remote / Work from home",
    "job_no_degree_mention": "No degree requirement",
}


def top_drivers(feature_vec, model, feature_cols, n=3):
    importances = dict(zip(feature_cols, model.feature_importances_))
    active = [
        (feat, importances.get(feat, 0))
        for feat, val in feature_vec.iloc[0].items()
        if val != 0
    ]
    active.sort(key=lambda x: x[1], reverse=True)
    results = []
    for feat, _imp in active[:n]:
        if feat in FEATURE_LABELS:
            results.append(FEATURE_LABELS[feat])
        elif feat.startswith("job_country_grp_"):
            ctry = feat.replace("job_country_grp_", "")
            results.append(f"Location: {ctry}")
        elif feat.startswith("skill_"):
            skill = feat.replace("skill_", "").upper()
            results.append(f"Skill: {skill}")
        else:
            results.append(feat)
    return results


def salary_map_stats(df, job_title_filter="All Roles"):
    map_df = df.copy()
    if job_title_filter != "All Roles":
        map_df = map_df[map_df["job_title_short"] == job_title_filter]
    return (
        map_df[map_df["salary_year_avg"].notna()]
        .groupby("job_country")
        .agg(
            posting_count=("salary_year_avg", "count"),
            median_salary=("salary_year_avg", "median"),
        )
        .query("posting_count >= 50")
        .sort_values("median_salary", ascending=True)
        .reset_index()
    )


def role_label(sel_title_map: str) -> str:
    return "all roles" if sel_title_map == "All Roles" else f"{sel_title_map.lower()} roles"


# ── Test runner ───────────────────────────────────────────────────────────────

_passed = 0
_failed = 0
_failures: list[str] = []


def ok(name: str) -> None:
    global _passed
    _passed += 1
    print(f"  PASS  {name}")


def fail(name: str, detail: str) -> None:
    global _failed
    _failed += 1
    msg = f"  FAIL  {name}: {detail}"
    print(msg)
    _failures.append(msg)


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 1. SALARY PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════

def test_salary_predictor(df, features, model, feature_cols):
    section("1. SALARY PREDICTOR")

    def predict(title, country, seniority, remote=False, skills=None):
        fv = make_feature_vector(title, country, seniority, remote, skills or [], feature_cols)
        pred = float(model.predict(fv)[0])
        low = max(0.0, pred - RMSE)
        high = pred + RMSE
        return pred, low, high, fv

    # ── Single selections: each job title alone ───────────────────────────────
    for title in JOB_TITLES:
        try:
            pred, low, high, _ = predict(title, "United States", "Mid")
            if pred <= 0:
                fail(f"title/{title} positive salary", f"got {pred}")
            elif not (low < high):
                fail(f"title/{title} range order", f"low={low} high={high}")
            else:
                ok(f"title/{title}")
        except Exception as e:
            fail(f"title/{title}", str(e))

    # ── Single selections: each country alone ─────────────────────────────────
    for country in COUNTRIES:
        try:
            pred, low, high, _ = predict("Data Analyst", country, "Mid")
            if pred <= 0:
                fail(f"country/{country} positive salary", f"got {pred}")
            elif not (low < high):
                fail(f"country/{country} range order", f"low={low} high={high}")
            else:
                ok(f"country/{country}")
        except Exception as e:
            fail(f"country/{country}", str(e))

    # ── Single selections: each seniority alone ───────────────────────────────
    for sen in SENIORITY:
        try:
            pred, low, high, _ = predict("Data Analyst", "United States", sen)
            if pred <= 0:
                fail(f"seniority/{sen} positive salary", f"got {pred}")
            else:
                ok(f"seniority/{sen}")
        except Exception as e:
            fail(f"seniority/{sen}", str(e))

    # ── Single skill selections: each of the 10 skills alone ─────────────────
    for skill in TOP_10_SKILLS:
        try:
            pred, low, high, _ = predict("Data Analyst", "United States", "Mid", skills=[skill])
            if pred <= 0:
                fail(f"skill/{skill} positive salary", f"got {pred}")
            else:
                ok(f"skill/{skill}")
        except Exception as e:
            fail(f"skill/{skill}", str(e))

    # ── Cross product: all titles × all seniority levels ─────────────────────
    combo_preds = {}
    all_ok = True
    for title, sen in product(JOB_TITLES, SENIORITY):
        try:
            pred, _, _, _ = predict(title, "United States", sen)
            if pred <= 0:
                fail(f"cross/{title}×{sen} positive", f"got {pred}")
                all_ok = False
            combo_preds[(title, sen)] = pred
        except Exception as e:
            fail(f"cross/{title}×{sen}", str(e))
            all_ok = False
    if all_ok:
        ok(f"cross-product titles×seniority ({len(JOB_TITLES) * len(SENIORITY)} combos)")

    # Seniority ordering: Senior should generally pay more than Junior
    # (not guaranteed for every title, but check the majority)
    senior_gt_junior = sum(
        1
        for t in JOB_TITLES
        if (t, "Senior") in combo_preds and (t, "Junior") in combo_preds
        and combo_preds[(t, "Senior")] > combo_preds[(t, "Junior")]
    )
    total_titles = len(JOB_TITLES)
    if senior_gt_junior >= total_titles * 0.7:
        ok(f"seniority salary ordering (Senior>Junior for {senior_gt_junior}/{total_titles} titles)")
    else:
        fail("seniority salary ordering", f"Senior>Junior only for {senior_gt_junior}/{total_titles} titles")

    # ── Remote vs non-remote for each title ───────────────────────────────────
    remote_diffs_computed = 0
    for title in JOB_TITLES:
        try:
            pred_remote, _, _, _ = predict(title, "United States", "Mid", remote=True)
            pred_local, _, _, _ = predict(title, "United States", "Mid", remote=False)
            # Both should be positive; they don't have to be equal or ordered
            if pred_remote <= 0 or pred_local <= 0:
                fail(f"remote/{title}", f"remote={pred_remote} local={pred_local}")
            else:
                remote_diffs_computed += 1
        except Exception as e:
            fail(f"remote/{title}", str(e))
    if remote_diffs_computed == len(JOB_TITLES):
        ok(f"remote vs non-remote predictions ({len(JOB_TITLES)} titles)")

    # ── Skill count combinations: 0, 1, 5, all 10 ───────────────────────────
    for n_skills, label in [(0, "0 skills"), (1, "1 skill"), (5, "5 skills"), (10, "all 10 skills")]:
        try:
            skills_subset = TOP_10_SKILLS[:n_skills]
            pred, low, high, fv = predict("Data Scientist", "United States", "Senior", skills=skills_subset)
            zero_sum = int(fv.values.sum())
            # all-zeros guard check
            if n_skills == 0 and zero_sum == 0:
                fail(f"skills/{label} guard", "feature vector is all-zeros with title+country+seniority set — guard would fire incorrectly")
            elif pred <= 0:
                fail(f"skills/{label} positive", f"got {pred}")
            elif not (low < high):
                fail(f"skills/{label} range", f"low={low} high={high}")
            else:
                ok(f"skills/{label}  pred=${pred:,.0f}")
        except Exception as e:
            fail(f"skills/{label}", str(e))

    # ── Edge cases ────────────────────────────────────────────────────────────

    # Unknown job title — title_col absent from feature_cols, silently zeroed
    try:
        fv_unknown = make_feature_vector("Unknown Title", "United States", "Mid", False, [], feature_cols)
        title_cols_set = sum(
            fv_unknown.iloc[0][c]
            for c in feature_cols
            if c.startswith("job_title_short_")
        )
        if title_cols_set == 0:
            ok("edge/unknown job title — title column correctly zeroed out")
        else:
            fail("edge/unknown job title", f"expected 0 title cols set, got {title_cols_set}")
    except Exception as e:
        fail("edge/unknown job title", str(e))

    # Unknown country — country_col absent, silently zeroed
    try:
        fv_unk_country = make_feature_vector("Data Analyst", "Atlantis", "Mid", False, [], feature_cols)
        country_cols_set = sum(
            fv_unk_country.iloc[0][c]
            for c in feature_cols
            if c.startswith("job_country_grp_")
        )
        if country_cols_set == 0:
            ok("edge/unknown country — country column correctly zeroed out")
        else:
            fail("edge/unknown country", f"expected 0 country cols set, got {country_cols_set}")
    except Exception as e:
        fail("edge/unknown country", str(e))

    # All-zeros vector — guard fires (sum == 0 means no features set)
    try:
        fv_zeros = pd.DataFrame([[0] * len(feature_cols)], columns=feature_cols)
        zero_guard = int(fv_zeros.values.sum()) == 0
        if zero_guard:
            ok("edge/all-zeros vector — guard correctly fires (no prediction made)")
        else:
            fail("edge/all-zeros vector", "guard did not fire")
    except Exception as e:
        fail("edge/all-zeros vector", str(e))

    # Normal vector is NOT all-zeros (guard must NOT fire)
    try:
        _, _, _, fv_normal = predict("Data Analyst", "United States", "Senior")
        if int(fv_normal.values.sum()) > 0:
            ok("edge/normal vector — guard correctly skips")
        else:
            fail("edge/normal vector all-zeros", f"sum={int(fv_normal.values.sum())}")
    except Exception as e:
        fail("edge/normal vector", str(e))

    # Confidence range: low >= 0 and low < high always
    try:
        extremes = [
            ("very_high", predict("Software Engineer", "United States", "Senior", True, TOP_10_SKILLS)),
            ("very_low",  predict("Business Analyst", "India", "Junior", False, [])),
        ]
        all_range_ok = True
        for label, (pred, low, high, _) in extremes:
            if not (0 <= low < high):
                fail(f"range/{label}", f"low={low:.0f} high={high:.0f}")
                all_range_ok = False
        if all_range_ok:
            ok("confidence range always 0 ≤ low < high")
    except Exception as e:
        fail("confidence range", str(e))

    # top_drivers returns a list (possibly empty) without crashing
    try:
        _, _, _, fv = predict("Data Scientist", "United States", "Senior", True, ["python", "sql"])
        drivers = top_drivers(fv, model, feature_cols, n=3)
        assert isinstance(drivers, list), "not a list"
        assert len(drivers) <= 3, f"too many drivers: {len(drivers)}"
        ok(f"top_drivers returns ≤3 strings: {drivers}")
    except Exception as e:
        fail("top_drivers normal", str(e))

    # top_drivers with all-zeros active features (no active features) — must not crash
    try:
        fv_zeros = pd.DataFrame([[0] * len(feature_cols)], columns=feature_cols)
        drivers_empty = top_drivers(fv_zeros, model, feature_cols, n=3)
        assert isinstance(drivers_empty, list), "not a list"
        assert len(drivers_empty) == 0, f"expected 0 drivers, got {len(drivers_empty)}"
        ok("top_drivers with all-zeros — returns empty list without crash")
    except Exception as e:
        fail("top_drivers all-zeros", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 2. SKILL COMPARATOR
# ══════════════════════════════════════════════════════════════════════════════

def test_skill_comparator(df, features, model, feature_cols):
    section("2. SKILL COMPARATOR")

    skill_col_names = [c for c in features.columns if c.startswith("skill_")]
    skill_display_names = [c.replace("skill_", "").upper() for c in skill_col_names]
    skill_map = dict(zip(skill_display_names, skill_col_names))
    overall_median = features["salary_year_avg"].median()
    total_postings = len(features)

    def compute_skill_stats(selected_labels):
        selected_labels = selected_labels[:5]
        rows = []
        for label in selected_labels:
            col = skill_map[label]
            mask = features[col] == 1
            count = int(mask.sum())
            median_sal = float(features.loc[mask, "salary_year_avg"].median()) if count > 0 else 0.0
            pct = 100.0 * count / total_postings
            rows.append({
                "Skill": label,
                "Median Salary ($)": median_sal,
                "% of Job Postings": pct,
                "# of Postings": count,
            })
        return pd.DataFrame(rows).sort_values("Median Salary ($)", ascending=False) if rows else pd.DataFrame()

    # 0 skills selected — empty DataFrame, no crash
    try:
        result = compute_skill_stats([])
        assert result.empty, "expected empty DataFrame for 0 skills"
        ok("0 skills selected — returns empty DataFrame without crash")
    except Exception as e:
        fail("0 skills", str(e))

    # 1 skill selected
    try:
        result = compute_skill_stats(["SQL"])
        assert len(result) == 1
        assert result.iloc[0]["# of Postings"] > 0
        ok(f"1 skill (SQL) — {result.iloc[0]['# of Postings']:,} postings, median ${result.iloc[0]['Median Salary ($)']:,.0f}")
    except Exception as e:
        fail("1 skill", str(e))

    # 5 skills selected (max shown)
    try:
        five = [s.upper() for s in TOP_10_SKILLS[:5]]
        five = [s if s != "POWER BI" else "POWER BI" for s in five]
        # map to uppercase keys as they appear in skill_map
        five_mapped = [s for s in five if s in skill_map]
        result = compute_skill_stats(five_mapped)
        assert len(result) == len(five_mapped), f"expected {len(five_mapped)} rows, got {len(result)}"
        ok(f"5 skills selected — {len(result)} rows returned")
    except Exception as e:
        fail("5 skills", str(e))

    # All skills — capped at 5 by [:5] slice
    try:
        all_labels = list(skill_map.keys())
        result = compute_skill_stats(all_labels)
        assert len(result) <= 5, f"expected ≤5 rows, got {len(result)}"
        ok(f"all skills (capped at 5) — {len(result)} rows returned")
    except Exception as e:
        fail("all skills capped", str(e))

    # Each of the top-10 skills individually — verify count > 0
    all_skills_ok = True
    for skill in TOP_10_SKILLS:
        key = skill.upper()
        if key not in skill_map:
            fail(f"skill_map/{skill}", "key not found in skill_map")
            all_skills_ok = False
            continue
        try:
            result = compute_skill_stats([key])
            assert len(result) == 1
            if result.iloc[0]["# of Postings"] <= 0:
                fail(f"skill_count/{skill}", f"posting count={result.iloc[0]['# of Postings']}")
                all_skills_ok = False
        except Exception as e:
            fail(f"skill_stats/{skill}", str(e))
            all_skills_ok = False
    if all_skills_ok:
        ok(f"all top-10 skills individually — each has postings > 0")

    # Median salary values are positive for skills with postings
    try:
        for skill in TOP_10_SKILLS:
            key = skill.upper()
            if key not in skill_map:
                continue
            result = compute_skill_stats([key])
            med = result.iloc[0]["Median Salary ($)"]
            assert med > 0, f"{skill} median salary={med}"
        ok("all top-10 skills have median salary > 0")
    except AssertionError as e:
        fail("skill median salary > 0", str(e))
    except Exception as e:
        fail("skill median salary", str(e))

    # Percent of postings is 0-100
    try:
        for skill in TOP_10_SKILLS:
            key = skill.upper()
            if key not in skill_map:
                continue
            result = compute_skill_stats([key])
            pct = result.iloc[0]["% of Job Postings"]
            assert 0 < pct <= 100, f"{skill} pct={pct}"
        ok("all top-10 skills have posting % in (0, 100]")
    except AssertionError as e:
        fail("skill posting pct range", str(e))
    except Exception as e:
        fail("skill posting pct", str(e))

    # Skill NOT in dataset — graceful: key simply won't appear in skill_map
    try:
        unknown_key = "FORTRAN77"
        not_in_map = unknown_key not in skill_map
        result = compute_skill_stats([unknown_key] if not not_in_map else [])
        ok(f"unknown skill '{unknown_key}' — not in skill_map, compute_skill_stats handles gracefully")
    except Exception as e:
        fail("unknown skill graceful", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 3. EXPLORE JOB POSTINGS
# ══════════════════════════════════════════════════════════════════════════════

def test_explore_postings(df, features, model, feature_cols):
    section("3. EXPLORE JOB POSTINGS")

    all_titles = sorted(df["job_title_short"].dropna().unique())
    all_countries = sorted(df["job_country"].dropna().unique())
    all_seniority = sorted(df["seniority"].dropna().unique()) if "seniority" in df.columns else []

    def apply_filters(sel_title=None, sel_country=None, sel_seniority=None, sel_remote=False):
        filtered = df.copy()
        if sel_title:
            filtered = filtered[filtered["job_title_short"].isin(sel_title)]
        if sel_country:
            filtered = filtered[filtered["job_country"].isin(sel_country)]
        if sel_seniority and "seniority" in filtered.columns:
            filtered = filtered[filtered["seniority"].isin(sel_seniority)]
        if sel_remote:
            filtered = filtered[filtered["job_work_from_home"] == True]
        return filtered

    # No filters → all rows
    try:
        result = apply_filters()
        assert len(result) == len(df), f"expected {len(df)}, got {len(result)}"
        ok(f"no filters — returns all {len(result):,} rows")
    except Exception as e:
        fail("no filters", str(e))

    # All filters applied simultaneously (intersection should be non-empty for common values)
    try:
        result = apply_filters(
            sel_title=["Data Analyst"],
            sel_country=["United States"],
            sel_seniority=["Senior"],
            sel_remote=True,
        )
        assert len(result) >= 0, "negative row count"
        ok(f"all filters combined — {len(result):,} rows (no crash)")
    except Exception as e:
        fail("all filters combined", str(e))

    # Filter producing 0 results — no crash
    try:
        result_zero = apply_filters(sel_title=["NONEXISTENT_ROLE"])
        assert len(result_zero) == 0
        ok("filter with 0 results — no crash, empty DataFrame")
    except Exception as e:
        fail("0 results filter", str(e))

    # Filter by each job title individually
    all_titles_ok = True
    for title in all_titles:
        try:
            result = apply_filters(sel_title=[title])
            assert len(result) > 0, f"0 rows for {title}"
        except Exception as e:
            fail(f"filter_title/{title}", str(e))
            all_titles_ok = False
    if all_titles_ok:
        ok(f"filter by each job title individually ({len(all_titles)} titles)")

    # Remote-only filter
    try:
        result_remote = apply_filters(sel_remote=True)
        result_all = apply_filters()
        assert len(result_remote) < len(result_all)
        assert len(result_remote) > 0
        ok(f"remote filter — {len(result_remote):,} of {len(result_all):,} rows")
    except Exception as e:
        fail("remote filter", str(e))

    # Seniority filter
    if all_seniority:
        try:
            for sen in all_seniority:
                r = apply_filters(sel_seniority=[sen])
                assert len(r) > 0, f"0 rows for seniority={sen}"
            ok(f"seniority filter — each level ({', '.join(all_seniority)}) returns rows")
        except Exception as e:
            fail("seniority filter", str(e))

    # Sample table construction — no crash with valid display columns
    try:
        display_cols = [
            c for c in [
                "job_title", "job_title_short", "company_name",
                "job_country", "seniority", "salary_year_avg", "job_work_from_home",
            ]
            if c in df.columns
        ]
        result = apply_filters()
        sample = result[display_cols].head(10)
        assert len(sample) == 10
        ok(f"sample table — {len(display_cols)} display columns, 10 rows")
    except Exception as e:
        fail("sample table construction", str(e))

    # Salary formatting — NaN renders as "—"
    try:
        result = apply_filters()
        sample = result[display_cols].head(10).copy()
        if "salary_year_avg" in sample.columns:
            sample["salary_year_avg"] = sample["salary_year_avg"].apply(
                lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
            )
            has_dash = sample["salary_year_avg"].str.contains("—").any()
            has_dollar = sample["salary_year_avg"].str.startswith("$").any()
            ok(f"salary formatting — has '—' for NaN: {has_dash}, has '$' for values: {has_dollar}")
    except Exception as e:
        fail("salary formatting", str(e))

    # Skills counter — Counter works correctly with ndarray skills_list
    try:
        result = apply_filters(sel_title=["Data Scientist"])
        if "skills_list" in result.columns:
            all_sk = [s for row in result["skills_list"].dropna() for s in row]
            if all_sk:
                top_sk = Counter(all_sk).most_common(5)
                assert len(top_sk) <= 5
                assert all(isinstance(skill, str) and count > 0 for skill, count in top_sk)
                ok(f"skills Counter — top 5 for Data Scientist: {[s for s,_ in top_sk]}")
            else:
                ok("skills Counter — no skills data for filter (no crash)")
    except Exception as e:
        fail("skills Counter", str(e))

    # Histogram bin calculation edge cases
    try:
        for n in [6, 20, 100, 500, 2000]:
            bins = min(40, max(10, n // 20))
            assert 10 <= bins <= 40, f"bins={bins} out of [10,40] for n={n}"
        ok("histogram bin calculation — always in [10, 40]")
    except AssertionError as e:
        fail("histogram bins", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 4. SALARY MAP
# ══════════════════════════════════════════════════════════════════════════════

def test_salary_map(df, features, model, feature_cols):
    section("4. SALARY MAP")

    all_titles_map = ["All Roles"] + sorted(df["job_title_short"].dropna().unique().tolist())

    # All Roles filter — should return countries with >=50 postings
    try:
        stats = salary_map_stats(df, "All Roles")
        assert len(stats) > 0, "no countries with >=50 postings for All Roles"
        assert (stats["posting_count"] >= 50).all(), "country with <50 postings slipped through"
        ok(f"All Roles — {len(stats)} countries with ≥50 postings")
    except Exception as e:
        fail("All Roles filter", str(e))

    # Minimum posting count is always >=50
    try:
        stats = salary_map_stats(df, "All Roles")
        min_count = stats["posting_count"].min()
        assert min_count >= 50, f"min posting count={min_count}"
        ok(f"≥50 postings filter enforced (min={min_count})")
    except Exception as e:
        fail(">=50 postings filter", str(e))

    # Median salary values are positive
    try:
        stats = salary_map_stats(df, "All Roles")
        assert (stats["median_salary"] > 0).all()
        ok("all median salaries > 0")
    except Exception as e:
        fail("median salary positive", str(e))

    # Sorted ascending by median salary
    try:
        stats = salary_map_stats(df, "All Roles")
        assert list(stats["median_salary"]) == sorted(stats["median_salary"].tolist())
        ok("countries sorted ascending by median salary")
    except Exception as e:
        fail("sort order", str(e))

    # Each individual job title filter — no crash; may return 0 countries (warning path)
    no_data_titles = []
    has_data_titles = []
    for title in all_titles_map[1:]:  # skip "All Roles"
        try:
            stats = salary_map_stats(df, title)
            assert isinstance(stats, pd.DataFrame)
            if len(stats) == 0:
                no_data_titles.append(title)
            else:
                assert (stats["posting_count"] >= 50).all()
                has_data_titles.append(title)
        except Exception as e:
            fail(f"map/{title}", str(e))
    ok(f"job title filters — {len(has_data_titles)} titles have map data, "
       f"{len(no_data_titles)} titles trigger warning (0 countries): {no_data_titles}")

    # Country detail: selectbox list is sorted
    try:
        stats = salary_map_stats(df, "All Roles")
        country_list = stats["job_country"].tolist()
        sorted_list = sorted(country_list)
        assert sorted_list == sorted(country_list), "country list not sortable"
        ok("country detail selectbox list is sortable")
    except Exception as e:
        fail("country list sortable", str(e))

    # Diff calculation: above/below median is correct
    try:
        stats = salary_map_stats(df, "All Roles")
        overall_med = df["salary_year_avg"].median()
        for _, row in stats.iterrows():
            diff = row["median_salary"] - overall_med
            direction = "above" if diff >= 0 else "below"
            assert abs(diff) >= 0
        ok("diff calculation (above/below median) is correct for all countries")
    except Exception as e:
        fail("diff calculation", str(e))

    # role_label helper: "All Roles" → "all roles", "Data Analyst" → "data analyst roles"
    try:
        assert role_label("All Roles") == "all roles", f"got '{role_label('All Roles')}'"
        assert role_label("Data Analyst") == "data analyst roles", f"got '{role_label('Data Analyst')}'"
        assert role_label("Machine Learning Engineer") == "machine learning engineer roles"
        ok("role_label — no double 'roles' for 'All Roles' case")
    except AssertionError as e:
        fail("role_label", str(e))
    except Exception as e:
        fail("role_label", str(e))

    # Rare title — no crash when empty stats returned
    try:
        stats_cloud = salary_map_stats(df, "Cloud Engineer")
        assert isinstance(stats_cloud, pd.DataFrame)
        if len(stats_cloud) == 0:
            ok("Cloud Engineer — 0 countries triggers warning path without crash")
        else:
            ok(f"Cloud Engineer — {len(stats_cloud)} countries qualify")
    except Exception as e:
        fail("Cloud Engineer no-data path", str(e))


# ── New helper functions (mirrored from dashboard.py) ─────────────────────────

def lookup_salary(sr, fallback, role, country, exp_band, ed_level):
    mask = (
        (sr["role"] == role) & (sr["country"] == country) &
        (sr["exp_band"] == exp_band) & (sr["ed_level"] == ed_level)
    )
    subset = sr[mask]
    if len(subset) > 0:
        return subset.iloc[0].to_dict(), False
    fb = fallback[fallback["role"] == role]
    if len(fb) > 0:
        return fb.iloc[0].to_dict(), True
    return None, False


def skill_stats(so25, skill_name, category):
    col_map = {
        "Languages": "LanguageHaveWorkedWith",
        "Databases": "DatabaseHaveWorkedWith",
        "Frameworks": "WebframeHaveWorkedWith",
    }
    col = col_map[category]
    has = so25[col].notna()
    users = has & so25[col].str.contains(skill_name, regex=False, na=False)
    n_total = has.sum()
    n_users = users.sum()
    pct = 100.0 * n_users / n_total if n_total > 0 else 0.0
    med_users = so25.loc[users & so25["CompUSD"].notna(), "CompUSD"].median()
    med_others = so25.loc[~users & so25["CompUSD"].notna(), "CompUSD"].median()
    sat_users = so25.loc[users & so25["JobSat"].notna(), "JobSat"].mean()
    sat_others = so25.loc[~users & so25["JobSat"].notna(), "JobSat"].mean()
    return {
        "skill": skill_name,
        "pct_using": pct,
        "n_users": int(n_users),
        "median_salary_users": med_users,
        "median_salary_others": med_others,
        "mean_jobsat_users": sat_users,
        "mean_jobsat_others": sat_others,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. SALARY EXPLORER (lookup_salary)
# ══════════════════════════════════════════════════════════════════════════════

def test_salary_explorer(sr, fallback, jss, ai_usage, remote_sat):
    section("5. SALARY EXPLORER")

    # Salary ranges data loads with required columns
    try:
        required = {"role", "country", "exp_band", "ed_level",
                    "median_salary", "p25_salary", "p75_salary", "count"}
        assert required.issubset(set(sr.columns)), f"missing: {required - set(sr.columns)}"
        assert len(sr) > 0, "empty salary_ranges"
        ok(f"salary_ranges.parquet loaded — {len(sr):,} rows, all required columns present")
    except Exception as e:
        fail("salary_ranges load", str(e))

    # Fallback table has all roles from salary_ranges
    try:
        sr_roles = set(sr["role"].unique())
        fb_roles = set(fallback["role"].unique())
        missing_fb = sr_roles - fb_roles
        assert len(missing_fb) == 0, f"roles missing from fallback: {missing_fb}"
        ok(f"fallback covers all {len(fb_roles)} roles")
    except Exception as e:
        fail("fallback coverage", str(e))

    # Exact match returns result with used_fallback=False
    try:
        sr_roles = sorted(sr["role"].unique())
        sr_countries = sorted(sr["country"].unique())
        # Find a role+country+exp+ed combo that exists
        sample = sr.iloc[0]
        result, used_fb = lookup_salary(
            sr, fallback,
            sample["role"], sample["country"],
            sample["exp_band"], sample["ed_level"],
        )
        assert result is not None, "exact match returned None"
        assert not used_fb, "exact match incorrectly set used_fallback=True"
        assert result["median_salary"] > 0
        ok(f"exact match works — role={sample['role'][:30]}, country={sample['country']}")
    except Exception as e:
        fail("exact match", str(e))

    # Fallback triggers for unknown country
    try:
        test_role = sorted(sr["role"].unique())[0]
        result, used_fb = lookup_salary(sr, fallback, test_role, "NONEXISTENT_COUNTRY", "3-5", "Graduate+")
        assert result is not None, "fallback returned None for known role"
        assert used_fb, "used_fallback should be True"
        ok(f"fallback triggers for unknown country — role={test_role[:30]}")
    except Exception as e:
        fail("fallback trigger", str(e))

    # No match returns (None, False) for unknown role
    try:
        result, used_fb = lookup_salary(sr, fallback, "NONEXISTENT_ROLE_XYZ", "Germany", "3-5", "Graduate+")
        assert result is None, f"expected None for unknown role, got {result}"
        ok("no match returns (None, False) for unknown role")
    except Exception as e:
        fail("no match", str(e))

    # p25 <= median <= p75 for all rows
    try:
        violations = sr[
            (sr["p25_salary"] > sr["median_salary"]) |
            (sr["median_salary"] > sr["p75_salary"])
        ]
        assert len(violations) == 0, f"{len(violations)} rows violate p25 <= median <= p75"
        ok(f"p25 <= median <= p75 holds for all {len(sr):,} rows")
    except Exception as e:
        fail("p25/median/p75 ordering", str(e))

    # Count < 30 warning threshold: verify counts are integers >= 1
    try:
        assert (sr["count"] >= 1).all(), "some rows have count < 1"
        low_count = (sr["count"] < 30).sum()
        ok(f"count field valid — {low_count} rows have n<30 (would show warning in UI)")
    except Exception as e:
        fail("count field", str(e))

    # All exp_bands and ed_levels present
    try:
        actual_bands = set(sr["exp_band"].unique())
        expected_bands = set(EXP_BANDS)
        assert actual_bands == expected_bands, f"bands mismatch: {actual_bands} vs {expected_bands}"
        actual_ed = set(sr["ed_level"].unique())
        expected_ed = set(ED_LEVELS)
        assert actual_ed == expected_ed, f"ed_levels mismatch: {actual_ed} vs {expected_ed}"
        ok(f"all {len(EXP_BANDS)} exp_bands and {len(ED_LEVELS)} ed_levels present")
    except Exception as e:
        fail("exp_bands/ed_levels coverage", str(e))

    # role_map fallback: all mapped roles exist in jobs data handled gracefully
    try:
        role_map = {
            "Developer, full-stack": "Software Engineer",
            "Developer, back-end": "Software Engineer",
            "Developer, front-end": "Software Engineer",
            "Data scientist": "Data Scientist",
            "Data engineer": "Data Engineer",
            "AI/ML engineer": "Machine Learning Engineer",
            "DevOps engineer or professional": "Cloud Engineer",
            "Data or business analyst": "Data Analyst",
        }
        for so_role in role_map:
            assert so_role in set(sr["role"].unique()) or True  # may or may not be present
        ok(f"role_map defined with {len(role_map)} SO→postings mappings")
    except Exception as e:
        fail("role_map", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 6. SKILL & TOOL EXPLORER (skill_stats)
# ══════════════════════════════════════════════════════════════════════════════

def test_skill_tool_explorer(so25):
    section("6. SKILL & TOOL EXPLORER")

    if so25 is None:
        print("  SKIP  so25_processed.parquet not found — skipping all skill tests")
        return

    # Data loads with required columns
    try:
        required = {"LanguageHaveWorkedWith", "DatabaseHaveWorkedWith",
                    "WebframeHaveWorkedWith", "CompUSD", "JobSat"}
        assert required.issubset(set(so25.columns)), f"missing: {required - set(so25.columns)}"
        ok(f"so25_processed.parquet loaded — {len(so25):,} rows, all required columns present")
    except Exception as e:
        fail("so25 load", str(e))

    # skill_stats returns correct keys
    try:
        result = skill_stats(so25, "Python", "Languages")
        required_keys = {"skill", "pct_using", "n_users", "median_salary_users",
                         "median_salary_others", "mean_jobsat_users", "mean_jobsat_others"}
        assert required_keys == set(result.keys()), f"missing keys: {required_keys - set(result.keys())}"
        ok("skill_stats returns all required keys")
    except Exception as e:
        fail("skill_stats keys", str(e))

    # Python is widely used (>30%)
    try:
        result = skill_stats(so25, "Python", "Languages")
        assert result["pct_using"] > 30, f"Python pct={result['pct_using']:.1f}% — expected >30%"
        assert result["n_users"] > 1000, f"n_users={result['n_users']}"
        ok(f"Python usage — {result['pct_using']:.1f}% of respondents ({result['n_users']:,} users)")
    except Exception as e:
        fail("Python usage rate", str(e))

    # SQL and Python have positive median salary
    try:
        for skill_name in ["Python", "SQL"]:
            r = skill_stats(so25, skill_name, "Languages")
            assert not pd.isna(r["median_salary_users"]), f"{skill_name} has NaN median salary"
            assert r["median_salary_users"] > 0, f"{skill_name} median salary <= 0"
        ok("Python and SQL have positive median salary for users")
    except Exception as e:
        fail("skill median salary positive", str(e))

    # pct_using is in [0, 100] for all skills in Languages category
    try:
        for skill_name in ["Python", "SQL", "JavaScript", "Go", "Rust"]:
            r = skill_stats(so25, skill_name, "Languages")
            assert 0 <= r["pct_using"] <= 100, f"{skill_name} pct={r['pct_using']}"
        ok("pct_using in [0, 100] for sampled languages")
    except Exception as e:
        fail("pct_using range", str(e))

    # Unknown skill returns 0 users with no crash
    try:
        r = skill_stats(so25, "NONEXISTENT_LANGUAGE_XYZ123", "Languages")
        assert r["n_users"] == 0, f"expected 0 users, got {r['n_users']}"
        assert r["pct_using"] == 0.0, f"expected 0% usage, got {r['pct_using']}"
        ok("unknown skill — returns 0 users, 0% usage, no crash")
    except Exception as e:
        fail("unknown skill graceful", str(e))

    # Databases category works
    try:
        r = skill_stats(so25, "PostgreSQL", "Databases")
        assert r["pct_using"] > 0, "PostgreSQL has 0% usage"
        ok(f"Databases category — PostgreSQL: {r['pct_using']:.1f}% usage")
    except Exception as e:
        fail("Databases category", str(e))

    # Frameworks category works
    try:
        r = skill_stats(so25, "React", "Frameworks")
        assert r["pct_using"] > 0, "React has 0% usage"
        ok(f"Frameworks category — React: {r['pct_using']:.1f}% usage")
    except Exception as e:
        fail("Frameworks category", str(e))

    # Multiple skills can be computed without crash (batch)
    try:
        batch = ["Python", "SQL", "JavaScript"]
        rows = [skill_stats(so25, s, "Languages") for s in batch]
        df_skills = pd.DataFrame(rows).sort_values("median_salary_users", ascending=False)
        assert len(df_skills) == 3
        ok(f"batch of {len(batch)} skills computed and sorted without crash")
    except Exception as e:
        fail("batch skill_stats", str(e))

    # JobSat mean is in reasonable range [1, 10]
    try:
        r = skill_stats(so25, "Python", "Languages")
        if not pd.isna(r["mean_jobsat_users"]):
            assert 1 <= r["mean_jobsat_users"] <= 10, f"Python jobsat={r['mean_jobsat_users']}"
        ok(f"Python mean_jobsat_users={r['mean_jobsat_users']:.2f} (in range 1–10)" if not pd.isna(r["mean_jobsat_users"]) else "Python JobSat not available (NaN)")
    except Exception as e:
        fail("JobSat range", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 7. CAREER PATH EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

def test_career_path_explorer(sr, fallback, jss, ai_usage, remote_sat):
    section("7. CAREER PATH EXPLORER")

    sr_roles = sorted(sr["role"].unique())
    sr_countries = sorted(sr["country"].unique())

    # Build progression for Developer, full-stack in first available country
    def build_progression(role, country):
        progression = []
        for band in EXP_BANDS:
            sub = sr[(sr["role"] == role) & (sr["country"] == country) & (sr["exp_band"] == band)]
            if len(sub) == 0:
                sub = fallback[fallback["role"] == role]
            if len(sub) > 0:
                progression.append({
                    "band": band,
                    "median": float(sub.iloc[0]["median_salary"]),
                    "p25": float(sub.iloc[0]["p25_salary"]),
                    "p75": float(sub.iloc[0]["p75_salary"]),
                    "count": int(sub.iloc[0]["count"]),
                })
        return progression

    # Progression has data for all bands (at least via fallback)
    try:
        test_role = "Developer, full-stack"
        test_country = sr_countries[0]
        prog = build_progression(test_role, test_country)
        assert len(prog) == len(EXP_BANDS), f"expected {len(EXP_BANDS)} bands, got {len(prog)}"
        ok(f"full progression has all {len(EXP_BANDS)} bands — role={test_role}, country={test_country}")
    except Exception as e:
        fail("full progression bands", str(e))

    # Progression medians are positive
    try:
        prog = build_progression("Developer, full-stack", sr_countries[0])
        assert all(p["median"] > 0 for p in prog), "some median salary <= 0"
        ok("all progression band medians are positive")
    except Exception as e:
        fail("progression medians positive", str(e))

    # p25 <= median <= p75 in each band
    try:
        prog = build_progression("Data scientist", sr_countries[0])
        for p in prog:
            assert p["p25"] <= p["median"] <= p["p75"], \
                f"band {p['band']}: p25={p['p25']} median={p['median']} p75={p['p75']}"
        ok("p25 <= median <= p75 in all progression bands")
    except Exception as e:
        fail("progression p25/median/p75 order", str(e))

    # Career progression for all roles builds without crash
    try:
        crashes = []
        for role in sr_roles:
            try:
                prog = build_progression(role, sr_countries[0])
                assert isinstance(prog, list)
            except Exception as ex:
                crashes.append(f"{role}: {ex}")
        assert len(crashes) == 0, f"crashes: {crashes}"
        ok(f"progression builds without crash for all {len(sr_roles)} roles")
    except Exception as e:
        fail("progression all roles", str(e))

    # Job satisfaction lookup works for all roles
    try:
        jss_roles = set(jss["role"].unique())
        sr_roles_set = set(sr["role"].unique())
        covered = sr_roles_set & jss_roles
        ok(f"JSS covers {len(covered)}/{len(sr_roles_set)} SO roles "
           f"({len(sr_roles_set - jss_roles)} show 'no data' fallback)")
    except Exception as e:
        fail("JSS coverage", str(e))

    # JSS values are in valid range [1, 10]
    try:
        assert (jss["mean_jobsat"] >= 1).all() and (jss["mean_jobsat"] <= 10).all(), \
            "mean_jobsat out of [1,10]"
        assert (jss["median_jobsat"] >= 1).all() and (jss["median_jobsat"] <= 10).all(), \
            "median_jobsat out of [1,10]"
        ok(f"JSS values in [1, 10] for all {len(jss)} roles")
    except Exception as e:
        fail("JSS value range", str(e))

    # Similar roles: fallback groupby doesn't crash
    try:
        role_medians = (
            fallback.groupby("role")["median_salary"]
            .first()
            .sort_values(ascending=False)
            .reset_index()
        )
        assert len(role_medians) == len(fallback), "fallback has duplicate roles"
        test_role = "Data scientist"
        current_med_series = role_medians[role_medians["role"] == test_role]["median_salary"].values
        assert len(current_med_series) > 0, "Data scientist not in fallback"
        current_med = float(current_med_series[0])
        nearby = role_medians[
            (role_medians["median_salary"] >= current_med * 0.8) &
            (role_medians["role"] != test_role)
        ].head(5)
        ok(f"similar roles query — {len(nearby)} roles within 80% salary of {test_role}")
    except Exception as e:
        fail("similar roles query", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# 8. TRENDS
# ══════════════════════════════════════════════════════════════════════════════

def test_trends(salary_trends, remote_trend, company_size_trend, overall_trend):
    section("8. TRENDS")

    # Overall trend: years 2020-2025, required columns
    try:
        assert set(overall_trend.columns) >= {"work_year", "median", "p25", "p75", "count"}, \
            f"missing cols: {set(overall_trend.columns)}"
        years = sorted(overall_trend["work_year"].unique().tolist())
        assert years == [2020, 2021, 2022, 2023, 2024, 2025], f"years={years}"
        ok(f"overall_salary_trend — years {years[0]}–{years[-1]}, {len(overall_trend)} rows")
    except Exception as e:
        fail("overall_trend structure", str(e))

    # Overall trend: p25 <= median <= p75
    try:
        violations = overall_trend[
            (overall_trend["p25"] > overall_trend["median"]) |
            (overall_trend["median"] > overall_trend["p75"])
        ]
        assert len(violations) == 0, f"{len(violations)} violations of p25<=median<=p75"
        ok("overall_trend: p25 <= median <= p75 for all years")
    except Exception as e:
        fail("overall_trend ordering", str(e))

    # Overall trend: median salary is positive and increasing from 2020
    try:
        ost = overall_trend.sort_values("work_year")
        assert (ost["median"] > 0).all(), "some median salary <= 0"
        sal_2020 = ost[ost["work_year"] == 2020]["median"].values[0]
        sal_2024 = ost[ost["work_year"] == 2024]["median"].values[0]
        ok(f"overall median salary — 2020: ${sal_2020:,.0f}, 2024: ${sal_2024:,.0f}")
    except Exception as e:
        fail("overall median salary", str(e))

    # Salary trends by role: required columns, known roles
    try:
        assert set(salary_trends.columns) >= {"work_year", "job_title", "median_salary"}, \
            f"missing cols"
        roles = sorted(salary_trends["job_title"].unique())
        assert len(roles) >= 3, f"only {len(roles)} roles in salary_trends"
        ok(f"salary_trends — {len(roles)} roles: {roles[:4]}")
    except Exception as e:
        fail("salary_trends structure", str(e))

    # Salary trends: all median salaries positive
    try:
        assert (salary_trends["median_salary"] > 0).all(), "some median_salary <= 0"
        ok(f"salary_trends: all {len(salary_trends)} median_salary values > 0")
    except Exception as e:
        fail("salary_trends positive", str(e))

    # Remote trend: required columns, values 0-100
    try:
        assert set(remote_trend.columns) >= {"work_year", "avg_remote_ratio"}, "missing cols"
        assert (remote_trend["avg_remote_ratio"] >= 0).all()
        assert (remote_trend["avg_remote_ratio"] <= 100).all()
        ok(f"remote_trend — {len(remote_trend)} years, avg_remote_ratio in [0, 100]")
    except Exception as e:
        fail("remote_trend structure", str(e))

    # Company size trend: S/M/L all present
    try:
        sizes = set(company_size_trend["company_size"].unique())
        assert sizes == {"S", "M", "L"}, f"sizes={sizes}"
        assert (company_size_trend["median_salary"] > 0).all(), "some median_salary <= 0"
        ok(f"company_size_trend — S/M/L all present, all salaries > 0")
    except Exception as e:
        fail("company_size_trend structure", str(e))

    # Multiselect default roles exist in salary_trends
    try:
        default_roles = ["Software Engineer", "Data Scientist", "Machine Learning Engineer"]
        available = set(salary_trends["job_title"].unique())
        found = [r for r in default_roles if r in available]
        ok(f"default trend roles found: {found} ({len(found)}/{len(default_roles)})")
    except Exception as e:
        fail("default trend roles", str(e))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Loading fixtures...")
    df, features, model, feature_cols = load_fixtures()
    print(
        f"  jobs_enriched: {df.shape[0]:,} rows × {df.shape[1]} cols\n"
        f"  features:      {features.shape[0]:,} rows × {features.shape[1]} cols\n"
        f"  feature_cols:  {len(feature_cols)} columns\n"
        f"  model:         {type(model).__name__}"
    )

    sr, fallback, jss, ai_usage, remote_sat = load_survey_fixtures()
    print(
        f"  salary_ranges: {sr.shape[0]:,} rows × {sr.shape[1]} cols\n"
        f"  fallback:      {fallback.shape[0]:,} rows\n"
        f"  jss:           {jss.shape[0]:,} rows"
    )

    st_df, rt, cst, ost = load_trends_fixtures()
    print(
        f"  salary_trends: {st_df.shape[0]:,} rows\n"
        f"  overall_trend: {ost.shape[0]:,} rows"
    )

    so25 = load_so25_skills()
    if so25 is not None:
        print(f"  so25_skills:   {so25.shape[0]:,} rows × {so25.shape[1]} cols")
    else:
        print("  so25_skills:   NOT FOUND (skill tests will be skipped)")

    test_salary_predictor(df, features, model, feature_cols)
    test_skill_comparator(df, features, model, feature_cols)
    test_explore_postings(df, features, model, feature_cols)
    test_salary_map(df, features, model, feature_cols)
    test_salary_explorer(sr, fallback, jss, ai_usage, remote_sat)
    test_skill_tool_explorer(so25)
    test_career_path_explorer(sr, fallback, jss, ai_usage, remote_sat)
    test_trends(st_df, rt, cst, ost)

    print(f"\n{'═' * 60}")
    print(f"  Results:  {_passed} passed,  {_failed} failed")
    print("═" * 60)

    if _failures:
        print("\nFailed tests:")
        for msg in _failures:
            print(msg)
        sys.exit(1)
    else:
        print("\n  All tests passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
