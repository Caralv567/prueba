import os
import pytest
import pandas as pd
import numpy as np
import cantilever_modal as cm

def test_single_modal_simulation():
    res = cm.run_modal_simulation(L=100.0, b=20.0, h=5.0, lc=2.5, num_modes=6)
    assert res["status"] == "SUCCESS"
    assert res["solver_finished"] is True
    assert res["dat_exists"] is True
    assert res["frd_exists"] is True
    assert res["non_negative_eigenvalues"] is True
    assert res["non_nan"] is True
    assert res["all_modes_found"] is True
    assert res["num_nodes"] > 0
    assert res["num_elements"] > 0
    # First natural frequency for h=5 mm should be ~411 Hz
    assert 400.0 < res["f1_hz"] < 420.0

def test_modal_files_exist():
    assert os.path.exists("resultados_modal_sweep.csv")
    assert os.path.exists("informe_modal_final.csv")
    assert os.path.exists("resumen_tecnico_modal.md")
    assert os.path.exists("espesor_vs_frecuencia_natural.png")
    assert os.path.exists("grafico_modos_vibracion.png")

def test_modal_csv_contents():
    df_sweep = pd.read_csv("resultados_modal_sweep.csv")
    assert len(df_sweep) == 5
    assert "f1_Hz" in df_sweep.columns
    assert "status" in df_sweep.columns
    assert (df_sweep["status"] == "SUCCESS").all()

    # Check linear trend f1_Hz vs h
    f1_vals = df_sweep["f1_Hz"].values
    h_vals = df_sweep["h"].values
    r_matrix = np.corrcoef(h_vals, f1_vals)
    r = r_matrix[0, 1]
    assert r > 0.999  # Almost perfectly linear

if __name__ == "__main__":
    print("Running modal unit tests...")
    test_single_modal_simulation()
    test_modal_files_exist()
    test_modal_csv_contents()
    print("All modal unit tests passed successfully!")
