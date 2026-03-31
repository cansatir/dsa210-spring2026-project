.PHONY: build start stop jupyter download local

build:
	docker compose up --build -d

start:
	docker compose up -d

stop:
	docker compose down

jupyter:
	open http://localhost:8888

download:
	docker compose exec jupyter python scripts/download_data.py

local:
	pip install -r requirements.txt && jupyter notebook notebooks/analysis.ipynb
