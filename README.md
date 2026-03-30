# DSA210 Spring 2026 — Tech Job Market Salary Analysis

## Project Overview

This project analyzes what factors affect salaries in the tech job market.
It combines real-world job postings with official U.S. Bureau of Labor Statistics (BLS) wage data to explore how job title, location, remote work, and required skills relate to salary.

## Data Sources

| Dataset | Source | Description |
|---|---|---|
| `lukebarousse/data_jobs` | HuggingFace | ~786k tech job postings |
| OEWS (Occupational Employment and Wage Statistics) | data.gov / BLS | Official U.S. wage estimates by occupation |

## Project Structure

```
dsa210-spring2026-project/
├── data/
│   ├── raw/          # Original downloaded datasets (not tracked by git)
│   └── processed/    # Cleaned and merged datasets (not tracked by git)
├── notebooks/
│   └── analysis.ipynb
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## How to Run

### Requirements
- Docker
- Docker Compose

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/cansatir/dsa210-spring2026-project.git
cd dsa210-spring2026-project

# 2. Build and start the container
docker compose up --build

# 3. Open Jupyter in your browser
# Go to: http://localhost:8888
```

To stop the container:
```bash
docker compose down
```

## Timeline

| Deadline | Task |
|---|---|
| 31 March | Project proposal |
| 14 April | Data collection, EDA, hypothesis tests |
| 5 May | ML methods |
| 18 May | Final report |
