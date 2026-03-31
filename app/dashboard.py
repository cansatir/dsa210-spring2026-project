"""
Tech Job Salary Dashboard — DSA210 Spring 2026
Run: streamlit run app/dashboard.py
"""
import pathlib
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import streamlit as st

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = pathlib.Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_PATH = BASE_DIR / "models" / "xgboost_salary.pkl"

# ── Constants ─────────────────────────────────────────────────────────────────
RMSE = 33_522  # ±1 RMSE confidence range (from model evaluation)

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

# Exact feature columns the model was trained on (56 total)
ALL_FEATURES = [
    "job_title_short_Business Analyst",
    "job_title_short_Cloud Engineer",
    "job_title_short_Data Analyst",
    "job_title_short_Data Engineer",
    "job_title_short_Data Scientist",
    "job_title_short_Machine Learning Engineer",
    "job_title_short_Senior Data Analyst",
    "job_title_short_Senior Data Engineer",
    "job_title_short_Senior Data Scientist",
    "job_title_short_Software Engineer",
    "job_country_grp_Australia",
    "job_country_grp_Canada",
    "job_country_grp_Colombia",
    "job_country_grp_France",
    "job_country_grp_Germany",
    "job_country_grp_Greece",
    "job_country_grp_India",
    "job_country_grp_Israel",
    "job_country_grp_Mexico",
    "job_country_grp_Netherlands",
    "job_country_grp_Other",
    "job_country_grp_Philippines",
    "job_country_grp_Poland",
    "job_country_grp_Portugal",
    "job_country_grp_Singapore",
    "job_country_grp_South Africa",
    "job_country_grp_South Korea",
    "job_country_grp_Sudan",
    "job_country_grp_Spain",
    "job_country_grp_United Kingdom",
    "job_country_grp_United States",
    "seniority_Junior",
    "seniority_Mid",
    "seniority_Senior",
    "job_work_from_home",
    "job_no_degree_mention",
    "skill_sql",
    "skill_python",
    "skill_r",
    "skill_tableau",
    "skill_excel",
    "skill_power bi",
    "skill_aws",
    "skill_azure",
    "skill_spark",
    "skill_java",
    "skill_snowflake",
    "skill_hadoop",
    "skill_nosql",
    "skill_scala",
    "skill_databricks",
    "skill_kafka",
    "skill_redshift",
    "skill_oracle",
    "skill_sas",
    "skill_airflow",
]

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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tech Job Salary Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data & model loading ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_parquet(DATA_DIR / "jobs_enriched.parquet")
    features = pd.read_parquet(DATA_DIR / "features.parquet")
    return df, features


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


# ── Helper functions ──────────────────────────────────────────────────────────
def make_feature_vector(job_title, country, seniority, remote, skills_selected):
    """Build a single-row DataFrame matching the model's training feature set."""
    vec = {col: 0 for col in ALL_FEATURES}
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
    return pd.DataFrame([vec])


def top_drivers(feature_vec, model, n=3):
    """Return the top-n most influential features for this prediction in plain English."""
    importances = dict(zip(ALL_FEATURES, model.feature_importances_))
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


# ── Navigation ────────────────────────────────────────────────────────────────
PAGES = [
    "💰 Salary Predictor",
    "📊 Skill Comparator",
    "🔍 Explore Job Postings",
    "🗺️ Salary Map",
]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")

# Load data once
df, features = load_data()
model = load_model()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Salary Predictor
# ════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("💰 Salary Predictor")
    st.write(
        "Fill in the job details below and click **Predict My Salary** to get an estimated "
        "annual salary. The estimate comes from a machine learning model trained on over "
        "21,000 real tech job postings. Results are in US dollars."
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        job_title = st.selectbox("Job Title", JOB_TITLES, index=2)
    with col2:
        country = st.selectbox("Country", COUNTRIES, index=0)
    with col3:
        seniority = st.selectbox("Seniority Level", SENIORITY, index=2)

    remote = st.checkbox("This is a remote / work-from-home position")

    st.write("**Which of these skills are required?**")
    skill_cols_ui = st.columns(5)
    selected_skills = []
    for i, skill in enumerate(TOP_10_SKILLS):
        with skill_cols_ui[i % 5]:
            label = skill.upper() if skill != "power bi" else "Power BI"
            if st.checkbox(label, key=f"pred_skill_{skill}"):
                selected_skills.append(skill)

    st.markdown("")
    if st.button("🔮 Predict My Salary", type="primary", use_container_width=False):
        feat_vec = make_feature_vector(job_title, country, seniority, remote, selected_skills)
        prediction = float(model.predict(feat_vec)[0])
        low = max(0.0, prediction - RMSE)
        high = prediction + RMSE

        st.markdown("---")
        result_col, space_col = st.columns([2, 1])
        with result_col:
            st.metric(
                label="Estimated Annual Salary",
                value=f"${prediction:,.0f}",
            )
            st.caption(
                f"Confidence range: **${low:,.0f} – ${high:,.0f}**  "
                f"(based on ±$33,522 model error)"
            )

        st.subheader("What's driving this estimate?")
        st.write(
            "Here are the three factors that most influenced this prediction:"
        )
        drivers = top_drivers(feat_vec, model, n=3)
        if drivers:
            for i, label in enumerate(drivers, 1):
                st.write(f"**{i}.** {label}")
        else:
            st.info(
                "No specific factors selected — the prediction uses the "
                "overall dataset average as a baseline."
            )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Skill Comparator
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("📊 Skill Comparator")
    st.write(
        "See how different technical skills affect pay and how often they appear in job postings. "
        "Select up to 5 skills to compare them side by side."
    )
    st.markdown("---")

    skill_col_names = [c for c in features.columns if c.startswith("skill_")]
    skill_display_names = [c.replace("skill_", "").upper() for c in skill_col_names]
    skill_map = dict(zip(skill_display_names, skill_col_names))

    default_skills = [s.upper() if s != "power bi" else "POWER BI" for s in TOP_10_SKILLS[:5]]
    default_skills = [s for s in default_skills if s in skill_map]

    selected_labels = st.multiselect(
        "Choose skills to compare (up to 5)",
        options=sorted(skill_map.keys()),
        default=default_skills,
    )
    selected_labels = selected_labels[:5]

    if selected_labels:
        overall_median = features["salary_year_avg"].median()
        total_postings = len(features)
        rows = []
        for label in selected_labels:
            col = skill_map[label]
            mask = features[col] == 1
            count = int(mask.sum())
            median_sal = float(features.loc[mask, "salary_year_avg"].median()) if count > 0 else 0.0
            pct = 100.0 * count / total_postings
            rows.append(
                {
                    "Skill": label,
                    "Median Salary ($)": median_sal,
                    "% of Job Postings": pct,
                    "# of Postings": count,
                }
            )

        comp_df = pd.DataFrame(rows).sort_values("Median Salary ($)", ascending=False)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Left: salary
        y_pos = range(len(comp_df))
        axes[0].barh(comp_df["Skill"], comp_df["Median Salary ($)"] / 1_000, color="#4C72B0")
        axes[0].axvline(
            overall_median / 1_000,
            color="#DD4444",
            linestyle="--",
            linewidth=1.5,
            label=f"Dataset median: ${overall_median / 1_000:.0f}K",
        )
        axes[0].set_xlabel("Median Annual Salary ($K)")
        axes[0].set_title("Median Salary by Skill")
        axes[0].legend(fontsize=9)

        # Right: demand
        axes[1].barh(comp_df["Skill"], comp_df["% of Job Postings"], color="#55A868")
        axes[1].set_xlabel("% of Job Postings Requiring This Skill")
        axes[1].set_title("Demand (Share of Job Postings)")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.subheader("Key Insights")
        for _, row in comp_df.iterrows():
            diff = row["Median Salary ($)"] - overall_median
            direction = "more" if diff >= 0 else "less"
            st.write(
                f"- Jobs requiring **{row['Skill']}** pay "
                f"**${abs(diff):,.0f} {direction}** than the dataset average "
                f"(${overall_median:,.0f}/yr). "
                f"Found in **{row['% of Job Postings']:.1f}%** of postings "
                f"({row['# of Postings']:,} jobs)."
            )
    else:
        st.info("Select at least one skill above to see the comparison.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Explore Job Postings
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("🔍 Explore Job Postings")
    st.write(
        "Browse and filter real job postings from the dataset. "
        "Use the filters on the left to narrow results by role, country, seniority, or work arrangement."
    )
    st.markdown("---")

    # Sidebar filters
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Filters")

    all_titles = sorted(df["job_title_short"].dropna().unique())
    sel_title = st.sidebar.multiselect("Job Title", all_titles)

    all_countries = sorted(df["job_country"].dropna().unique())
    sel_country = st.sidebar.multiselect("Country", all_countries)

    if "seniority" in df.columns:
        all_seniority = sorted(df["seniority"].dropna().unique())
        sel_seniority = st.sidebar.multiselect("Seniority", all_seniority)
    else:
        sel_seniority = []

    sel_remote = st.sidebar.checkbox("Remote positions only")

    # Apply filters
    filtered = df.copy()
    if sel_title:
        filtered = filtered[filtered["job_title_short"].isin(sel_title)]
    if sel_country:
        filtered = filtered[filtered["job_country"].isin(sel_country)]
    if sel_seniority and "seniority" in filtered.columns:
        filtered = filtered[filtered["seniority"].isin(sel_seniority)]
    if sel_remote:
        filtered = filtered[filtered["job_work_from_home"] == True]

    filtered_with_salary = filtered[filtered["salary_year_avg"].notna()]

    st.metric("Matching Job Postings", f"{len(filtered):,}")

    if len(filtered_with_salary) > 5:
        chart_col, skill_col_ui = st.columns(2)

        with chart_col:
            st.subheader("Salary Distribution")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(
                filtered_with_salary["salary_year_avg"] / 1_000,
                bins=min(40, max(10, len(filtered_with_salary) // 20)),
                color="#4C72B0",
                edgecolor="white",
                linewidth=0.5,
            )
            ax.set_xlabel("Annual Salary ($K)")
            ax.set_ylabel("Number of Postings")
            ax.set_title("Salary Distribution in Filtered Results")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with skill_col_ui:
            st.subheader("Top Skills in Results")
            if "skills_list" in filtered.columns:
                from collections import Counter
                all_sk = [s for row in filtered["skills_list"].dropna() for s in row]
                if all_sk:
                    top_sk = Counter(all_sk).most_common(10)
                    sk_names = [s.upper() for s, _ in reversed(top_sk)]
                    sk_counts = [c for _, c in reversed(top_sk)]
                    fig2, ax2 = plt.subplots(figsize=(7, 4))
                    ax2.barh(sk_names, sk_counts, color="#55A868")
                    ax2.set_xlabel("Number of Job Postings")
                    ax2.set_title("Most Common Skills")
                    plt.tight_layout()
                    st.pyplot(fig2)
                    plt.close(fig2)
                else:
                    st.info("No skill data available for this filter.")
            else:
                st.info("Skill data not available.")
    elif len(filtered_with_salary) > 0:
        st.info("Not enough postings with salary data to draw charts. Showing table only.")
    else:
        st.warning("No postings with salary data match the current filters.")

    st.markdown("---")
    st.subheader("Sample Job Postings (up to 10 rows)")
    display_cols = [
        c
        for c in [
            "job_title",
            "job_title_short",
            "company_name",
            "job_country",
            "seniority",
            "salary_year_avg",
            "job_work_from_home",
        ]
        if c in filtered.columns
    ]
    rename_map = {
        "job_title": "Title",
        "job_title_short": "Role",
        "company_name": "Company",
        "job_country": "Country",
        "seniority": "Seniority",
        "salary_year_avg": "Salary ($/yr)",
        "job_work_from_home": "Remote",
    }
    sample = filtered[display_cols].head(10).rename(columns=rename_map)
    if "Salary ($/yr)" in sample.columns:
        sample["Salary ($/yr)"] = sample["Salary ($/yr)"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
        )
    st.dataframe(sample, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Salary Map
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("🗺️ Salary Map")
    st.write(
        "Compare median salaries across countries. "
        "Only countries with at least 50 job postings are shown to ensure the numbers are reliable. "
        "Filter by job title to see country-level comparisons for a specific role."
    )
    st.markdown("---")

    all_titles_map = ["All Roles"] + sorted(df["job_title_short"].dropna().unique().tolist())
    sel_title_map = st.selectbox("Filter by Job Title", all_titles_map)

    map_df = df.copy()
    if sel_title_map != "All Roles":
        map_df = map_df[map_df["job_title_short"] == sel_title_map]

    country_stats = (
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

    if len(country_stats) == 0:
        st.warning(
            "Not enough data for this filter (need ≥50 postings per country). "
            "Try selecting 'All Roles' or a more common job title."
        )
    else:
        fig, ax = plt.subplots(figsize=(10, max(5, len(country_stats) * 0.38)))
        colors = [
            "#DD8452" if i == len(country_stats) - 1 else "#4C72B0"
            for i in range(len(country_stats))
        ]
        ax.barh(country_stats["job_country"], country_stats["median_salary"] / 1_000, color=colors)
        ax.set_xlabel("Median Annual Salary ($K)")
        ax.set_title(f"Median Salary by Country — {sel_title_map}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.subheader("Country Details")
        detail_col1, detail_col2 = st.columns([1, 2])
        with detail_col1:
            selected_country = st.selectbox(
                "Select a country to see its stats",
                sorted(country_stats["job_country"].tolist()),
            )

        if selected_country:
            row = country_stats[country_stats["job_country"] == selected_country].iloc[0]
            overall_med = map_df["salary_year_avg"].median()

            m1, m2, m3 = st.columns(3)
            m1.metric("Country", selected_country)
            m2.metric("Median Salary", f"${row['median_salary']:,.0f}")
            m3.metric("Job Postings", f"{row['posting_count']:,}")

            diff = row["median_salary"] - overall_med
            direction = "above" if diff >= 0 else "below"
            st.info(
                f"**{selected_country}** has a median salary of **${row['median_salary']:,.0f}**, "
                f"which is **${abs(diff):,.0f} {direction}** the overall median "
                f"of **${overall_med:,.0f}** for {sel_title_map.lower()} roles."
            )
