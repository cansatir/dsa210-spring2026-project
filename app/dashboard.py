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
RMSE = 33_522

JOB_TITLES = [
    "Business Analyst", "Cloud Engineer", "Data Analyst", "Data Engineer",
    "Data Scientist", "Machine Learning Engineer", "Senior Data Analyst",
    "Senior Data Engineer", "Senior Data Scientist", "Software Engineer",
]

COUNTRIES = [
    "United States", "Canada", "United Kingdom", "Australia", "Germany",
    "France", "India", "Netherlands", "Poland", "Spain", "Portugal",
    "Colombia", "Greece", "Israel", "Mexico", "Philippines", "Singapore",
    "South Africa", "South Korea", "Sudan", "Other",
]

SENIORITY = ["Junior", "Mid", "Senior"]

TOP_10_SKILLS = [
    "sql", "python", "aws", "azure", "tableau", "spark",
    "excel", "power bi", "r", "sas",
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Tech Job Salary Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_job_data():
    df = pd.read_parquet(DATA_DIR / "jobs_enriched.parquet")
    features = pd.read_parquet(DATA_DIR / "features.parquet")
    return df, features


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_cols():
    cols = pd.read_parquet(DATA_DIR / "features.parquet").columns.tolist()
    return [c for c in cols if c != "salary_year_avg"]


@st.cache_data
def load_salary_ranges():
    sr = pd.read_parquet(DATA_DIR / "salary_ranges.parquet")
    fallback = pd.read_parquet(DATA_DIR / "salary_ranges_by_role.parquet")
    return sr, fallback


@st.cache_data
def load_satisfaction():
    jss = pd.read_parquet(DATA_DIR / "job_satisfaction_summary.parquet")
    ai_usage = pd.read_parquet(DATA_DIR / "ai_usage_satisfaction.parquet")
    remote_sat = pd.read_parquet(DATA_DIR / "remote_satisfaction.parquet")
    return jss, ai_usage, remote_sat


@st.cache_data
def load_trends():
    st_df = pd.read_parquet(DATA_DIR / "salary_trends.parquet")
    rt = pd.read_parquet(DATA_DIR / "remote_trend.parquet")
    cst = pd.read_parquet(DATA_DIR / "company_size_trend.parquet")
    ost = pd.read_parquet(DATA_DIR / "overall_salary_trend.parquet")
    return st_df, rt, cst, ost


@st.cache_data
def load_so25_skills():
    so25_path = DATA_DIR / "so25_processed.parquet"
    if not so25_path.exists():
        return None
    cols = [
        "LanguageHaveWorkedWith", "DatabaseHaveWorkedWith",
        "WebframeHaveWorkedWith", "CompUSD", "JobSat",
    ]
    return pd.read_parquet(so25_path, columns=[c for c in cols if c])


# ── Helper functions ──────────────────────────────────────────────────────────
def make_feature_vector(job_title, country, seniority, remote, skills_selected, feature_cols):
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


def years_to_band(years):
    if years <= 2:
        return "0-2"
    if years <= 5:
        return "3-5"
    if years <= 10:
        return "6-10"
    if years <= 20:
        return "11-20"
    return "20+"


def lookup_salary(sr, fallback, role, country, exp_band, ed_level):
    """Return (row_dict, used_fallback). Tries exact match then role-level fallback."""
    mask = (
        (sr["role"] == role) &
        (sr["country"] == country) &
        (sr["exp_band"] == exp_band) &
        (sr["ed_level"] == ed_level)
    )
    subset = sr[mask]
    if len(subset) > 0:
        return subset.iloc[0].to_dict(), False
    # fallback: role only
    fb = fallback[fallback["role"] == role]
    if len(fb) > 0:
        return fb.iloc[0].to_dict(), True
    return None, False


def skill_stats(so25, skill_name, category):
    """Compute % using, median salary (users vs non), mean JobSat for a skill."""
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


# ── Navigation ────────────────────────────────────────────────────────────────
PAGES = [
    "💰 Salary Explorer",
    "🛠️ Skill & Tool Explorer",
    "🚀 Career Path Explorer",
    "🔍 Explore Job Postings",
    "🗺️ Salary Map",
    "📈 Trends",
]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", PAGES, label_visibility="collapsed")

# Load job data (always needed for pages 4 & 5)
df, features = load_job_data()
feature_cols = load_feature_cols()
model = load_model()


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Salary Explorer
# ════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    st.title("💰 Salary Explorer")
    st.write(
        "See real salary ranges from the **Stack Overflow Developer Survey 2025** "
        "(49,000+ responses). Results show the middle 50% of reported salaries "
        "(p25–p75) and the median."
    )
    st.markdown("---")

    sr, fallback = load_salary_ranges()
    sr_roles = sorted(sr["role"].unique())
    sr_countries = sorted(sr["country"].unique())

    col1, col2 = st.columns(2)
    with col1:
        sel_role = st.selectbox("Developer Role", sr_roles, index=sr_roles.index("Developer, full-stack") if "Developer, full-stack" in sr_roles else 0)
        sel_country = st.selectbox("Country", sr_countries, index=sr_countries.index("United States") if "United States" in sr_countries else 0)
    with col2:
        sel_exp = st.select_slider("Years of Experience", options=EXP_BANDS, value="3-5")
        sel_ed = st.selectbox("Education Level", ED_LEVELS)

    st.markdown("")
    row, used_fallback = lookup_salary(sr, fallback, sel_role, sel_country, sel_exp, sel_ed)

    if row is None:
        st.warning("No salary data available for this combination.")
    else:
        if used_fallback:
            st.info(
                "Not enough survey responses for this exact profile. "
                "Showing overall figures for this role across all countries and experience levels."
            )

        count = int(row["count"])
        median_sal = float(row["median_salary"])
        p25 = float(row["p25_salary"])
        p75 = float(row["p75_salary"])

        if count < 30:
            st.warning(f"Only {count} survey responses match this profile — treat these figures as estimates.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Median Salary", f"${median_sal:,.0f}")
        m2.metric("25th Percentile", f"${p25:,.0f}")
        m3.metric("75th Percentile", f"${p75:,.0f}")
        m4.metric("Survey Responses", f"{count:,}")

        # Visual salary bar
        fig, ax = plt.subplots(figsize=(10, 1.8))
        ax.barh(["Salary Range"], [p75 - p25], left=[p25], color="#4C72B0", height=0.4, label="Middle 50% (p25–p75)")
        ax.plot([median_sal], [0], "D", color="#DD4444", markersize=10, zorder=5, label=f"Median: ${median_sal/1000:.0f}K")
        ax.set_xlabel("Annual Salary (USD)")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}K"))
        ax.legend(loc="upper right", fontsize=9)
        ax.set_title(f"{sel_role} · {sel_country} · {sel_exp} yrs · {sel_ed}")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Side-by-side: survey salary vs job postings salary
        st.markdown("---")
        st.subheader("Survey Data vs Job Postings")

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
        mapped_title = role_map.get(sel_role)

        posting_med = None
        if mapped_title:
            sub = df[(df["job_title_short"] == mapped_title) & df["salary_year_avg"].notna()]
            if len(sub) >= 10:
                posting_med = sub["salary_year_avg"].median()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**From Stack Overflow Survey 2025**")
            st.markdown(f"Based on {count:,} real developer responses")
            st.markdown(f"- Median: **${median_sal:,.0f}**")
            st.markdown(f"- Range (p25–p75): **${p25:,.0f} – ${p75:,.0f}**")
        with c2:
            st.markdown("**From Job Postings Dataset**")
            if posting_med is not None:
                st.markdown(f"Based on job postings for *{mapped_title}*")
                st.markdown(f"- Median listed salary: **${posting_med:,.0f}**")
                diff = median_sal - posting_med
                direction = "higher" if diff >= 0 else "lower"
                st.markdown(f"- Survey median is **${abs(diff):,.0f} {direction}** than posted salaries")
            else:
                st.markdown("No matching job postings found for this role.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Skill & Tool Explorer
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[1]:
    st.title("🛠️ Skill & Tool Explorer")
    st.write(
        "Explore how programming languages, databases, and frameworks relate to salary "
        "and job satisfaction. Data from the Stack Overflow Developer Survey 2025."
    )
    st.markdown("---")

    so25 = load_so25_skills()
    if so25 is None:
        st.warning(
            "SO 2025 processed data not found. "
            "Run `notebooks/02_development/01_data_exploration.ipynb` to generate it."
        )
        st.stop()

    category = st.radio("Category", list(SKILL_CATEGORIES.keys()), horizontal=True)
    available_skills = SKILL_CATEGORIES[category]

    default_sel = available_skills[:5]
    selected_skills = st.multiselect(
        f"Select {category} to compare (up to 8)",
        available_skills,
        default=default_sel,
    )
    selected_skills = selected_skills[:8]

    if not selected_skills:
        st.info("Select at least one skill above.")
        st.stop()

    rows = [skill_stats(so25, s, category) for s in selected_skills]
    skill_df = pd.DataFrame(rows).sort_values("median_salary_users", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, len(skill_df) * 0.5 + 1)))

    # Median salary
    axes[0].barh(skill_df["skill"], skill_df["median_salary_users"] / 1000, color="#4C72B0", label="Users")
    axes[0].barh(skill_df["skill"], skill_df["median_salary_others"] / 1000, color="#CCCCCC", alpha=0.5, label="Non-users")
    axes[0].set_xlabel("Median Salary ($K)")
    axes[0].set_title("Median Salary")
    axes[0].legend(fontsize=8)

    # % using
    axes[1].barh(skill_df["skill"], skill_df["pct_using"], color="#55A868")
    axes[1].set_xlabel("% of Respondents Using")
    axes[1].set_title("Usage Rate")

    # Job satisfaction
    sat_users = skill_df["mean_jobsat_users"].fillna(0)
    sat_others = skill_df["mean_jobsat_others"].fillna(0)
    y = np.arange(len(skill_df))
    axes[2].barh(y - 0.2, sat_users, 0.35, color="#DD8452", label="Users")
    axes[2].barh(y + 0.2, sat_others, 0.35, color="#CCCCCC", label="Non-users")
    axes[2].set_yticks(y)
    axes[2].set_yticklabels(skill_df["skill"])
    axes[2].set_xlabel("Mean Job Satisfaction (1–10)")
    axes[2].set_title("Job Satisfaction")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Key Insights")
    for _, row in skill_df.iterrows():
        if pd.isna(row["median_salary_users"]):
            continue
        diff = row["median_salary_users"] - row["median_salary_others"]
        direction = "more" if diff >= 0 else "less"
        sat_diff = (row["mean_jobsat_users"] or 0) - (row["mean_jobsat_others"] or 0)
        sat_note = ""
        if abs(sat_diff) >= 0.2:
            sat_direction = "higher" if sat_diff > 0 else "lower"
            sat_note = f" Job satisfaction is **{abs(sat_diff):.1f} pts {sat_direction}** for users."
        st.write(
            f"- **{row['skill']}**: used by **{row['pct_using']:.1f}%** of respondents. "
            f"Users earn **${abs(diff):,.0f} {direction}** than non-users "
            f"(${row['median_salary_users']:,.0f} vs ${row['median_salary_others']:,.0f}).{sat_note}"
        )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Career Path Explorer
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[2]:
    st.title("🚀 Career Path Explorer")
    st.write(
        "See how salary and job satisfaction evolve across career stages for a given role. "
        "Data from Stack Overflow Developer Survey 2025."
    )
    st.markdown("---")

    sr, fallback = load_salary_ranges()
    jss, ai_usage, remote_sat = load_satisfaction()

    sr_roles = sorted(sr["role"].unique())
    sr_countries = sorted(sr["country"].unique())

    col1, col2 = st.columns(2)
    with col1:
        sel_role = st.selectbox(
            "Developer Role",
            sr_roles,
            index=sr_roles.index("Developer, full-stack") if "Developer, full-stack" in sr_roles else 0,
            key="career_role",
        )
    with col2:
        sel_country = st.selectbox(
            "Country",
            sr_countries,
            index=sr_countries.index("United States") if "United States" in sr_countries else 0,
            key="career_country",
        )

    # Salary progression across experience bands
    progression = []
    for band in EXP_BANDS:
        sub = sr[(sr["role"] == sel_role) & (sr["country"] == sel_country) & (sr["exp_band"] == band)]
        if len(sub) == 0:
            sub = fallback[fallback["role"] == sel_role]
        if len(sub) > 0:
            progression.append({
                "band": band,
                "median": float(sub.iloc[0]["median_salary"]),
                "p25": float(sub.iloc[0]["p25_salary"]),
                "p75": float(sub.iloc[0]["p75_salary"]),
                "count": int(sub.iloc[0]["count"]),
            })

    if progression:
        prog_df = pd.DataFrame(progression)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(prog_df["band"], prog_df["p25"] / 1000, prog_df["p75"] / 1000,
                        alpha=0.25, color="#4C72B0", label="p25–p75 range")
        ax.plot(prog_df["band"], prog_df["median"] / 1000, "o-", color="#4C72B0",
                linewidth=2.5, markersize=8, label="Median salary")
        ax.set_xlabel("Years of Experience")
        ax.set_ylabel("Annual Salary ($K)")
        ax.set_title(f"Salary Progression — {sel_role} ({sel_country})")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}K"))
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("Salary by Experience Band")
            for p in progression:
                st.markdown(
                    f"**{p['band']} yrs**: ${p['median']:,.0f} median "
                    f"(${p['p25']:,.0f}–${p['p75']:,.0f}), n={p['count']}"
                )

        with col_b:
            st.subheader("Job Satisfaction")
            jss_row = jss[jss["role"] == sel_role]
            if len(jss_row) > 0:
                r = jss_row.iloc[0]
                st.metric("Mean Job Satisfaction", f"{r['mean_jobsat']:.1f} / 10")
                st.metric("Median Job Satisfaction", f"{r['median_jobsat']:.0f} / 10")
                st.metric("Survey Respondents", f"{r['count_jobsat']:,}")
            else:
                st.info("No satisfaction data for this role.")

        # Similar roles
        st.markdown("---")
        st.subheader("Similar Roles by Salary")
        role_medians = (
            fallback.groupby("role")["median_salary"]
            .first()
            .sort_values(ascending=False)
            .reset_index()
        )
        current_med = role_medians[role_medians["role"] == sel_role]["median_salary"].values
        if len(current_med) > 0:
            current_med = float(current_med[0])
            nearby = role_medians[
                (role_medians["median_salary"] >= current_med * 0.8) &
                (role_medians["role"] != sel_role)
            ].head(5)
            for _, r in nearby.iterrows():
                diff = r["median_salary"] - current_med
                direction = "more" if diff >= 0 else "less"
                st.write(f"- **{r['role']}**: ${r['median_salary']:,.0f} (${abs(diff):,.0f} {direction})")
    else:
        st.warning("No salary data found for this combination.")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Explore Job Postings
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[3]:
    st.title("🔍 Explore Job Postings")
    st.write(
        "Browse and filter real job postings from the dataset. "
        "Use the filters on the left to narrow results by role, country, seniority, or work arrangement."
    )
    st.markdown("---")

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
                color="#4C72B0", edgecolor="white", linewidth=0.5,
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
                all_sk = [s for row in filtered["skills_list"].dropna() for s in (row if hasattr(row, '__iter__') and not isinstance(row, str) else [])]
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
        c for c in [
            "job_title", "job_title_short", "company_name", "job_country",
            "seniority", "salary_year_avg", "job_work_from_home",
        ] if c in filtered.columns
    ]
    rename_map = {
        "job_title": "Title", "job_title_short": "Role", "company_name": "Company",
        "job_country": "Country", "seniority": "Seniority",
        "salary_year_avg": "Salary ($/yr)", "job_work_from_home": "Remote",
    }
    sample = filtered[display_cols].head(10).rename(columns=rename_map)
    if "Salary ($/yr)" in sample.columns:
        sample["Salary ($/yr)"] = sample["Salary ($/yr)"].apply(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
        )
    st.dataframe(sample, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Salary Map
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[4]:
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
        detail_col1, _ = st.columns([1, 2])
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
            role_label = "all roles" if sel_title_map == "All Roles" else f"{sel_title_map.lower()} roles"
            st.info(
                f"**{selected_country}** has a median salary of **${row['median_salary']:,.0f}**, "
                f"which is **${abs(diff):,.0f} {direction}** the overall median "
                f"of **${overall_med:,.0f}** for {role_label}."
            )


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Trends
# ════════════════════════════════════════════════════════════════════════════
elif page == PAGES[5]:
    st.title("📈 Trends")
    st.write(
        "Salary trends from 2020 to 2025 based on the aijobs.net dataset (151,000+ postings). "
        "Tracks how pay, remote work, and company size preferences have evolved."
    )
    st.markdown("---")

    salary_trends, remote_trend, company_size_trend, overall_trend = load_trends()

    tab1, tab2, tab3 = st.tabs(["Salary Trends", "Remote Work", "Company Size"])

    with tab1:
        st.subheader("Overall Salary Trend (All Roles)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.fill_between(
            overall_trend["work_year"],
            overall_trend["p25"] / 1000,
            overall_trend["p75"] / 1000,
            alpha=0.2, color="#4C72B0", label="p25–p75 range",
        )
        ax.plot(overall_trend["work_year"], overall_trend["median"] / 1000,
                "o-", color="#4C72B0", linewidth=2.5, markersize=8, label="Median")
        ax.set_xlabel("Year")
        ax.set_ylabel("Median Salary ($K)")
        ax.set_title("Median AI/Tech Salary 2020–2025")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}K"))
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.subheader("Salary Trend by Role")
        roles_trend = sorted(salary_trends["job_title"].unique())
        default_roles = [r for r in ["Software Engineer", "Data Scientist", "Machine Learning Engineer"] if r in roles_trend]
        sel_roles = st.multiselect("Select roles", roles_trend, default=default_roles or roles_trend[:3])

        if sel_roles:
            fig2, ax2 = plt.subplots(figsize=(10, 5))
            colors_list = plt.cm.tab10.colors
            for i, role in enumerate(sel_roles):
                sub = salary_trends[salary_trends["job_title"] == role].sort_values("work_year")
                ax2.plot(sub["work_year"], sub["median_salary"] / 1000, "o-",
                         color=colors_list[i % 10], linewidth=2, markersize=7, label=role)
            ax2.set_xlabel("Year")
            ax2.set_ylabel("Median Salary ($K)")
            ax2.set_title("Salary by Role — 2020–2025")
            ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}K"))
            ax2.legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("Select at least one role above.")

    with tab2:
        st.subheader("Remote Work Ratio Trend")
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.bar(remote_trend["work_year"], remote_trend["avg_remote_ratio"],
                color="#55A868", alpha=0.8)
        ax3.set_xlabel("Year")
        ax3.set_ylabel("Average Remote Ratio (%)")
        ax3.set_title("Remote Work in AI/Tech Jobs 2020–2025")
        ax3.set_ylim(0, 100)
        for _, r in remote_trend.iterrows():
            ax3.text(r["work_year"], r["avg_remote_ratio"] + 1,
                     f"{r['avg_remote_ratio']:.0f}%", ha="center", fontsize=9)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        st.markdown("---")
        st.subheader("Job Satisfaction by Work Arrangement (SO 2025)")
        _, _, remote_sat = load_satisfaction()
        remote_sat_sorted = remote_sat.sort_values("median_jobsat", ascending=True)
        fig4, ax4 = plt.subplots(figsize=(10, max(3, len(remote_sat_sorted) * 0.5)))
        ax4.barh(remote_sat_sorted["remote_work"], remote_sat_sorted["median_jobsat"], color="#DD8452")
        ax4.set_xlabel("Median Job Satisfaction (1–10)")
        ax4.set_title("Job Satisfaction by Work Arrangement")
        ax4.set_xlim(0, 10)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    with tab3:
        st.subheader("Median Salary by Company Size")
        size_labels = {"S": "Small", "M": "Medium", "L": "Large"}
        cst_plot = company_size_trend.copy()
        cst_plot["size_label"] = cst_plot["company_size"].map(size_labels)

        fig5, ax5 = plt.subplots(figsize=(10, 4))
        colors_size = {"Small": "#4C72B0", "Medium": "#55A868", "Large": "#DD8452"}
        for size in ["Small", "Medium", "Large"]:
            sub = cst_plot[cst_plot["size_label"] == size].sort_values("work_year")
            ax5.plot(sub["work_year"], sub["median_salary"] / 1000, "o-",
                     color=colors_size[size], linewidth=2, markersize=7, label=size)
        ax5.set_xlabel("Year")
        ax5.set_ylabel("Median Salary ($K)")
        ax5.set_title("Salary by Company Size 2020–2025")
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:.0f}K"))
        ax5.legend(title="Company Size")
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

        st.markdown("---")
        st.subheader("AI Usage & Job Satisfaction (SO 2025)")
        _, ai_usage, _ = load_satisfaction()
        ai_sorted = ai_usage.sort_values("median_jobsat", ascending=True)
        fig6, ax6 = plt.subplots(figsize=(10, 3.5))
        ax6.barh(ai_sorted["ai_usage"], ai_sorted["median_jobsat"], color="#4C72B0")
        ax6.set_xlabel("Median Job Satisfaction (1–10)")
        ax6.set_title("Job Satisfaction by AI Tool Usage")
        ax6.set_xlim(0, 10)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close(fig6)
