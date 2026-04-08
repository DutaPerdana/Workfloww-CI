
import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import sys 
# update final
warnings.filterwarnings("ignore")

# --- Fungsi Evaluasi dan Manual Logging ---
def eval_and_log_manual(model, X_test, y_test, run_id, input_example=None):
    """Menghitung metrik dan mencatat semuanya secara manual ke MLflow."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Hitung Metrik
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    try:
        auc_roc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc_roc = 0.0

    # Log Metrik Secara Manual 
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", auc_roc)

    # Log Confusion Matrix sebagai Artefak Visual
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix (Acc: {accuracy:.4f})")
    
    cm_path = f"confusion_matrix_{run_id}.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path, artifact_path="visual_artifacts")
    os.remove(cm_path)
    plt.close()

    # Log Model Terbaik
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        input_example=input_example,
        registered_model_name="resiko_kesehatan_ci_model" 
    )
    return accuracy

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--max_depth", type=int, default=15)
    args = parser.parse_args() 
    
    warnings.filterwarnings("ignore")
    RANDOM_STATE = 42
    np.random.seed(RANDOM_STATE)
    
    n_estimators = args.n_estimators # Ambil dari argparse
    max_depth = args.max_depth       # Ambil dari argparse

    
    GITHUB_ROOT = os.environ.get('GITHUB_WORKSPACE', os.getcwd()) 
    RELATIVE_DATA_PATH = "MLproject/dataset_preprocessing/preprocessed_data.csv"
    file_path = os.path.join(GITHUB_ROOT, RELATIVE_DATA_PATH)
    
    print(f"CI Run Parameters: n_estimators={n_estimators}, max_depth={max_depth}, Data Path={file_path}")

    # --- 3. Pemuatan Data ---
    try:
        data = pd.read_csv(file_path) 
    except FileNotFoundError:
        print(f"ERROR FATAL: File data preprocessing tidak ditemukan di {file_path}. Gagal Memuat.")
        sys.exit(1)
        
   
    # --- 4. Split Data dan Training ---
    
    # Pisahkan Fitur (X) dan Target (y) - K
    X = data.drop("Status_Resiko", axis=1) 
    y = data["Status_Resiko"]

    # Train-Test Split 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=RANDOM_STATE, test_size=0.2, stratify=y
    )
    input_example = X_train.head(5)

    # --- 5. Memulai MLflow Run 
    if "MLFLOW_RUN_ID" not in os.environ:
        mlflow.set_experiment("CI Workflow Resiko Kesehatan")

    current_run_id = os.environ.get("MLFLOW_RUN_ID")

    with mlflow.start_run(run_id=current_run_id) as run:
        run_id = run.info.run_id
    
    # Ambil Run ID aktif 
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        
        # Set Run Name
        mlflow.set_tag("mlflow.runName", f"CI_n{n_estimators}_d{max_depth}")
        
        # Log Parameter Secara Manual
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("data_source", RELATIVE_DATA_PATH)
        mlflow.log_param("random_state", RANDOM_STATE) # Log state yang di-hardcode
        
        # Model Training
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=RANDOM_STATE # Gunakan state yang di-hardcode
        )
        model.fit(X_train, y_train)
        
        # Evaluasi dan Log Artefak Model
        current_accuracy = eval_and_log_manual(
            model, X_test, y_test, run_id, 
            input_example=input_example
        )
        
        print(f"\nCI Run Selesai. Akurasi: {current_accuracy:.4f}")