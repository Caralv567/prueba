import os
import pytest
import pandas as pd
import numpy as np
import cantilever_fatigue_ml as cfm

def test_single_fatigue_sim():
    res = cfm.run_single_fatigue_simulation(P=100.0)
    assert res["status"] == "SUCCESS"
    assert res["load_balanced"] is True
    assert 0.70 < res["displacement"] < 0.90
    assert 100.0 < res["von_mises"] < 130.0
    assert res["PASS_FAIL"] == "PASS"

def test_dataset_generation():
    df = cfm.generate_dataset([50.0, 100.0, 150.0])
    assert len(df) == 3
    assert "P" in df.columns
    assert "goodman_utilization" in df.columns
    assert "PASS_FAIL" in df.columns
    assert (df["displacement"] > 0).all()

def test_ml_pipeline():
    df = cfm.generate_dataset([50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0])
    ml_res = cfm.train_and_evaluate_ml(df)
    df_metrics = ml_res["df_metrics"]
    assert "LR_R2" in df_metrics.columns
    assert (df_metrics["LR_R2"] > 0.999).all()

def test_output_files_exist():
    assert os.path.exists("dataset_fatiga_fea.csv")
    assert os.path.exists("informe_fatiga_ml.csv")
    assert os.path.exists("README.md")
    assert os.path.exists("grafico_carga_vs_vonmises.png")
    assert os.path.exists("grafico_carga_vs_desplazamiento.png")
    assert os.path.exists("grafico_carga_vs_goodman.png")
    assert os.path.exists("grafico_reales_vs_predichos.png")

if __name__ == "__main__":
    print("Running fatigue + ML unit tests...")
    test_single_fatigue_sim()
    test_dataset_generation()
    test_ml_pipeline()
    test_output_files_exist()
    print("All fatigue + ML unit tests passed successfully!")
