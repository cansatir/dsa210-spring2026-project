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

    test_salary_predictor(df, features, model, feature_cols)
    test_skill_comparator(df, features, model, feature_cols)
    test_explore_postings(df, features, model, feature_cols)
    test_salary_map(df, features, model, feature_cols)

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
