import os
import pytest
import pandas as pd
import numpy as np
import cantilever_opt as co

def test_single_simulation():
    res = co.run_simulation(L=100.0, b=20.0, h=5.0, P=100.0, lc=2.5)
    assert res["status"] == "SUCCESS"
    assert res["ccx_finished"] is True
    assert res["frd_exists"] is True
    assert res["load_balanced"] is True
    assert res["non_nan"] is True
    assert res["num_nodes"] > 0
    assert res["num_elements"] > 0
    assert 0.70 < res["max_disp"] < 0.90
    assert 100.0 < res["max_von_mises"] < 130.0

def test_generated_files_exist():
    assert os.path.exists("resultados_parametric_sweep.csv")
    assert os.path.exists("informe_final.csv")
    assert os.path.exists("resumen_tecnico.md")
    assert os.path.exists("grafico_espesor_vs_desplazamiento.png")
    assert os.path.exists("grafico_espesor_vs_vonmises.png")
    assert os.path.exists("grafico_espesor_vs_masa.png")
    assert os.path.exists("grafico_resumen_optimizacion.png")

def test_csv_contents():
    df_sweep = pd.read_csv("resultados_parametric_sweep.csv")
    assert len(df_sweep) >= 6
    assert "max_disp" in df_sweep.columns
    assert "max_von_mises" in df_sweep.columns
    assert "status" in df_sweep.columns
    assert (df_sweep["status"] == "SUCCESS").all()

    df_final = pd.read_csv("informe_final.csv")
    assert "Métrica / Parámetro" in df_final.columns
    assert ("Diseño Optimizado" in df_final.columns or "Diseño Candidato Confirmado" in df_final.columns)

def test_resultados_finales_deliverables():
    assert os.path.exists("resultados_finales/01_resultados_barrido.csv")
    assert os.path.exists("resultados_finales/02_comparacion_disenos.csv")
    assert os.path.exists("resultados_finales/03_verificaciones.csv")
    assert os.path.exists("resultados_finales/04_desplazamiento_vs_espesor.png")
    assert os.path.exists("resultados_finales/05_von_mises_vs_espesor.png")
    assert os.path.exists("resultados_finales/06_masa_vs_espesor.png")
    assert os.path.exists("resultados_finales/07_resumen_tecnico.md")
    assert os.path.exists("resultados_finales/08_registro_ejecucion.log")
    assert os.path.exists("resultados_finales/09_manifest_entregables.csv")

    df_manifest = pd.read_csv("resultados_finales/09_manifest_entregables.csv")
    assert len(df_manifest) == 9
    assert (df_manifest["existe"] == True).all()
    assert (df_manifest["estado_validacion"] == "COMPLETO").all()

if __name__ == "__main__":
    print("Running tests...")
    test_single_simulation()
    test_generated_files_exist()
    test_csv_contents()
    print("All unit tests passed successfully!")
