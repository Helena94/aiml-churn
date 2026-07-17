# Project Context: AI/ML Customer Churn Prediction

## 1. Project Overview

This project focuses on building an end-to-end Machine Learning pipeline for customer churn prediction. It utilizes a
Kaggle dataset and implements standard MLOps practices for data processing, model training, batch inference, monitoring,
and visualization.

## 2. Current Project Status

* **EDA & Raw Data:** ✅ Completed.
* **ETL Pipeline (Prefect):** ✅ Completed. Data is extracted, cleaned, validated, and loaded into `.parquet` format.
* **ML Training (MLflow + Prefect):** 🔄 In Progress (Current Focus). Training models, logging metrics, and storing the
  best model.
* **Batch Predictions (Prefect):** ⏳ Pending.
* **Monitoring (EvidentlyAI):** ⏳ Pending.
* **Dashboard (Streamlit):** ⏳ Pending.

## 3. Directory Structure

The repository strictly follows this organizational structure to maintain modularity:

```text
aiml-churn/
├── data/
│   ├── raw/                # Original Kaggle data (read-only)
│   ├── processed/          # ETL outputs (train/valid/test.parquet)
│   └── predictions/        # Inference outputs
├── src/
│   └── churn/
│       ├── etl/            # ✅ Completed (extract.py, transform.py, load.py)
│       ├── features/       # Shared feature logic & schema definitions
│       └── models/         # Model definitions only   └── random_forest.py