# MuleX Project Structure & File Guide

This document provides a simple explanation of every file and folder currently present in your MuleX fraud detection repository. 

## 📁 Core Directories

- **`data/`**
  - **`Base.csv`**: This is your main dataset (the NeurIPS 2022 Bank Account Fraud dataset). It contains roughly 1 million historical credit application records used to train the machine learning models.

- **`models/`**
  - This folder holds the trained machine learning artifacts (saved as `.joblib` files) that are generated after `main.py` finishes training. It contains:
    - **`engineer.joblib`**: The fitted data pipeline. It remembers how to scale features and apply the exact WOE (Weight of Evidence) encodings to new data.
    - **`ensemble.joblib`**: The core AI model. It stores the fully trained LightGBM, XGBoost, and Logistic Regression Meta-Learner weights.
    - **`calibrator.joblib`**: The Isotonic Regression model that converts raw tree outputs into perfectly calibrated probabilities (0.0 to 1.0).
    - **`explainer.joblib`**: The SHAP (SHapley Additive exPlanations) explainer. It calculates exactly which features influenced the model to flag or approve an application.

- **`src/`** *(Source Code)*
  - This folder contains all the Python scripts that make up the machine learning pipeline. (See the Python Scripts section below for details).

---

## 🐍 Python Scripts (inside `src/`)

- **`src/pipeline.py`**
  - **Purpose**: Data Preparation & Feature Engineering.
  - **What it does**: It cleans the raw data (e.g., converting `-1` placeholders to missing values), encodes categorical text into numbers using WOE (Weight of Evidence), creates new mathematical features (like credit-to-income ratios), and properly splits the data into past/future chunks to prevent time-travel leakage.

- **`src/models.py`**
  - **Purpose**: Machine Learning Algorithms.
  - **What it does**: This file defines the actual AI brains. It contains the code for the LightGBM and XGBoost models, the Custom Focal Loss function (to handle the 1% extreme fraud rarity), and the Stacking Meta-Learner that intelligently combines their predictions.

- **`src/main.py`**
  - **Purpose**: The Training Orchestrator.
  - **What it does**: You run this file when you want to train the model from scratch. It loads the data using `pipeline.py`, runs Optuna to find the absolute best hyperparameter settings, trains the models defined in `models.py`, and saves the finished models into the `models/` folder.

- **`src/inference.py`**
  - **Purpose**: Real-Time Fraud Scoring (Production).
  - **What it does**: You run this file to evaluate a new customer application. It takes a JSON payload, passes it through the pre-trained models, assigns a Risk Score (0-100), decides the Risk Tier (Approved 🟢, MFA 🟡, or Frozen 🔴), and prints out the red/green flag explanations.

---

## 📄 Documentation & Configuration Files

- **`README.md`**
  - **Purpose**: Project Overview & Quickstart.
  - **What it does**: Comprehensive repository guide containing architecture highlights, performance metrics, CLI commands, and directory layouts.

- **`HACKATHON_PITCH_DECK.md`**
  - **Purpose**: Master Pitch Deck & Defense Brief.
  - **What it does**: Executive summary, problem/solution breakdown, technical specification, inference examples, and Judge Q&A defense guide for hackathon presentation.

- **`batch_test_cases.json`**
  - **Purpose**: Inference Benchmarking.
  - **What it does**: Contains 20 simulated customer applications (a mix of low, medium, and high risk). It is used to test if `inference.py` is correctly identifying fraud.

- **`model_summary.md`**
  - **Purpose**: Architecture Documentation.
  - **What it does**: A concise explanation of the machine learning techniques used in this project (Optuna, Stacking, Focal Loss, OOT evaluation metrics).

- **`prompt.md`**
  - **Purpose**: System Specification.
  - **What it does**: The strict prompt/instruction file detailing the overall rules of the system, including how the data is structured and what the risk thresholds should be.

- **`requirements.txt`**
  - **Purpose**: Dependencies.
  - **What it does**: A list of all the Python packages (like `pandas`, `lightgbm`, `xgboost`, `optuna`, `shap`) required to run the code. You install these using `pip install -r requirements.txt`.
