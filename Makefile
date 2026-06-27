.PHONY: help benchmark experiment dashboard web api kaggle clean

help:
	@echo "Teknofest 2026 Deep-Pipeline Komutlari:"
	@echo "  make benchmark    - Benchmark testlerini calistir"
	@echo "  make experiment   - Yeni bir deney baslat"
	@echo "  make dashboard    - Streamlit XAI dashboard'u baslat"
	@echo "  make api          - FastAPI servisini baslat"
	@echo "  make kaggle       - Kaggle submission uret"
	@echo "  make web          - 3D Web UI projesini baslat"
	@echo "  make reports      - PDF raporlari olustur"
	@echo "  make clean        - Gereksiz dosyalari temizle"

benchmark:
	python scripts/run_benchmark.py

experiment:
	python scripts/run_experiment.py --config configs/base_config.yaml

dashboard:
	streamlit run scripts/run_dashboard.py

api:
	python -m uvicorn src.deployment.api:app --host 0.0.0.0 --port 8000 --reload

kaggle:
	python scripts/kaggle_submission.py --config configs/model/kaggle.yaml

xai-demo:
	python scripts/run_xai_demo.py

prepare-kaggle:
	@echo "Kullanim: python scripts/prepare_kaggle_data.py --train PATH --test PATH"

reports:
	python scripts/generate_reports.py

web:
	cd deep-pipeline-web && npm run dev

web-build:
	cd deep-pipeline-web && npm install && npm run build

web-deploy:
	@echo "Deploying Next.js frontend to production..."
	cd deep-pipeline-web && npm run start

clean:
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/*/__pycache__
	rm -rf .pytest_cache
