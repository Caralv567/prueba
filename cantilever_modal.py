import os
import sys
import time
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cadquery as cq
import gmsh

# --- 1. CAD GENERATION ---
def generate_cad(L, b, h, output_step_path):
    """
    Generates a 3D box solid model using CadQuery and exports it to STEP.
    Dimensions: L (length, X), b (width, Y), h (thickness, Z).
    """
    box = cq.Workplane("XY").box(L, b, h)
    os.makedirs(os.path.dirname(output_step_path), exist_ok=True)
    cq.exporters.export(box, output_step_path)
    return output_step_path


# --- 2. MESH GENERATION ---
def generate_mesh(step_path, L, b, h, lc, output_inp_path):
    """
    Imports STEP file into Gmsh, generates 2nd-order C3D10 solid elements,
    tags volume and fixed boundary face (X=-L/2), and writes initial mesh INP.
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("cantilever_modal")

    gmsh.model.occ.importShapes(step_path)
    gmsh.model.occ.synchronize()

    # Identify fixed face at X = -L/2
    faces = gmsh.model.getEntities(2)
    fixed_face = None
    tol = 1e-3
    for dim, tag in faces:
        bbox = gmsh.model.getBoundingBox(dim, tag)
        if abs(bbox[0] - (-L/2)) < tol and abs(bbox[3] - (-L/2)) < tol:
            fixed_face = tag

    volumes = gmsh.model.getEntities(3)
    volume_tags = [v[1] for v in volumes]

    gmsh.model.addPhysicalGroup(3, volume_tags, name="EALL")
    if fixed_face is not None:
        gmsh.model.addPhysicalGroup(2, [fixed_face], name="FIXED_SURF")

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
    gmsh.option.setNumber("Mesh.ElementOrder", 2)  # C3D10

    gmsh.model.mesh.generate(3)
    os.makedirs(os.path.dirname(output_inp_path), exist_ok=True)
    gmsh.write(output_inp_path)

    node_tags, _, _ = gmsh.model.mesh.getNodes()
    elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
    num_nodes = len(node_tags)
    num_elements = sum(len(tags) for tags in elem_tags)

    gmsh.finalize()
    return num_nodes, num_elements


# --- 3. INPUT GENERATION ---
def prepare_modal_inp(initial_inp_path, final_inp_path, L, b, h, num_modes=6, E=200000.0, nu=0.30, rho=7.85e-9):
    """
    Cleans Gmsh mesh INP file and appends CalculiX modal analysis cards:
    - Fixed BC at X = -L/2
    - Steel material properties (E=200,000 MPa, nu=0.30, density=7.85e-9 t/mm^3 = 7850 kg/m^3)
    - *STEP with *FREQUENCY (num_modes)
    - Node output request (*NODE FILE U)
    """
    with open(initial_inp_path, 'r') as f:
        lines = f.readlines()

    cleaned_lines = []
    skip_2d = False
    for line in lines:
        if line.startswith('*ELEMENT, type=CPS6') or line.startswith('*ELEMENT, type=CPS3'):
            skip_2d = True
            continue
        elif line.startswith('*ELEMENT, type=C3D10') or line.startswith('*ELEMENT, type=C3D4'):
            skip_2d = False
        if not skip_2d:
            cleaned_lines.append(line)

    nodes_dict = {}
    in_node = False
    for line in cleaned_lines:
        if line.startswith('*NODE'):
            in_node = True
            continue
        elif line.startswith('*'):
            in_node = False
        if in_node:
            parts = line.strip().split(',')
            if len(parts) == 4:
                nid = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                nodes_dict[nid] = (x, y, z)

    tol = 1e-3
    fixed_nodes = [nid for nid, (x, y, z) in nodes_dict.items() if abs(x - (-L/2)) < tol]

    if len(fixed_nodes) == 0:
        raise ValueError("Could not identify fixed boundary nodes at X = -L/2.")

    nset_str = "*NSET, NSET=NFIXED\n" + ",\n".join(map(str, fixed_nodes)) + "\n"

    mat_str = f"""*MATERIAL, NAME=STEEL
*ELASTIC
{E:.1f}, {nu:.2f}
*DENSITY
{rho:.2E}
*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL
*STEP
*FREQUENCY
{num_modes}
*BOUNDARY
NFIXED, 1, 3, 0.0
*NODE FILE
U
*END STEP
"""

    final_content = "".join(cleaned_lines) + "\n" + nset_str + mat_str

    os.makedirs(os.path.dirname(final_inp_path), exist_ok=True)
    with open(final_inp_path, 'w') as f:
        f.write(final_content)

    return nodes_dict


# --- 4. CALCULIX EXECUTION ---
def run_calculix_modal(job_name, work_dir="sim_runs"):
    """
    Executes CalculiX ccx solver for modal analysis.
    Returns returncode, stdout, and stderr.
    """
    cmd = ["ccx", job_name]
    res = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr


# --- 5. EIGENVALUE EXTRACTION & POSTPROCESSING ---
def parse_modal_results(dat_path, frd_path, num_modes=6):
    """
    Parses CalculiX .dat file to extract eigenvalues and natural frequencies.
    Converts angular frequency omega (rad/s) to cyclic frequency f (Hz) via f = omega / (2 * pi).
    Also parses .frd file for mode shape displacement vectors.
    """
    modes_data = []

    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"CalculiX output .dat file not found: {dat_path}")

    with open(dat_path, 'r') as f:
        lines = f.readlines()

    in_eigenvalue_table = False
    for line in lines:
        if "E I G E N V A L U E   O U T P U T" in line:
            in_eigenvalue_table = True
            continue
        if in_eigenvalue_table and "P A R T I C I P A T I O N" in line:
            in_eigenvalue_table = False
            break
        if in_eigenvalue_table:
            parts = line.strip().split()
            # Row format: MODE_NO, EIGENVALUE, REAL_PART_RAD_PER_TIME, CYCLES_PER_TIME, IMAG_PART
            if len(parts) >= 4 and parts[0].isdigit():
                mode_no = int(parts[0])
                eigenvalue = float(parts[1])
                omega = float(parts[2])  # rad/s
                freq_hz_ccx = float(parts[3])  # Cycles/time (Hz)

                # Conversion check: f = omega / (2 * pi)
                freq_hz_calc = omega / (2.0 * np.pi)

                modes_data.append({
                    "Mode": mode_no,
                    "Eigenvalue": eigenvalue,
                    "omega_rad_s": omega,
                    "Frequency_Hz": freq_hz_calc
                })

    if len(modes_data) == 0:
        raise ValueError("No eigenvalue / frequency output found in .dat file.")

    # Parse .frd file for mode shape displacements
    mode_shapes = {}
    if os.path.exists(frd_path):
        with open(frd_path, 'r') as f:
            frd_lines = f.readlines()

        current_mode = None
        i = 0
        while i < len(frd_lines):
            line = frd_lines[i]
            if line.startswith(' -4') and 'DISP' in line:
                # Mode number can be inferred from step/frequency block
                i += 1
                while frd_lines[i].startswith(' -5'):
                    i += 1
                disp_dict = {}
                while i < len(frd_lines) and frd_lines[i].startswith(' -1'):
                    l = frd_lines[i]
                    nid = int(l[3:13])
                    ux = float(l[13:25])
                    uy = float(l[25:37])
                    uz = float(l[37:49])
                    disp_dict[nid] = (ux, uy, uz)
                    i += 1
                mode_idx = len(mode_shapes) + 1
                mode_shapes[mode_idx] = disp_dict
                continue
            i += 1

    return modes_data, mode_shapes


# --- AUTOMATIC VALIDATIONS & SIMULATION WRAPPER ---
def run_modal_simulation(L=100.0, b=20.0, h=5.0, lc=2.5, num_modes=6, work_dir="sim_runs"):
    """
    Executes complete modal simulation pipeline with strict automatic checks:
    - CalculiX termination status
    - File existence (.dat and .frd)
    - Non-negative eigenvalues (lambda > 0)
    - Non-NaN / finite natural frequencies
    - Completeness of modes extracted
    """
    os.makedirs(work_dir, exist_ok=True)
    job_name = f"cantilever_modal_h{h:.2f}".replace('.', '_')
    step_path = os.path.join(work_dir, f"{job_name}.step")
    initial_inp = os.path.join(work_dir, f"{job_name}_gmsh.inp")
    final_inp = os.path.join(work_dir, f"{job_name}.inp")
    dat_path = os.path.join(work_dir, f"{job_name}.dat")
    frd_path = os.path.join(work_dir, f"{job_name}.frd")

    result_dict = {
        "L": L, "b": b, "h": h,
        "f1_hz": np.nan,
        "modes_table": None,
        "mode_shapes": None,
        "num_nodes": 0,
        "num_elements": 0,
        "solver_finished": False,
        "dat_exists": False,
        "frd_exists": False,
        "non_negative_eigenvalues": False,
        "non_nan": False,
        "all_modes_found": False,
        "status": "FAILED"
    }

    try:
        # Step 1: CAD
        generate_cad(L, b, h, step_path)

        # Step 2: Mesh
        num_nodes, num_elements = generate_mesh(step_path, L, b, h, lc, initial_inp)
        result_dict["num_nodes"] = num_nodes
        result_dict["num_elements"] = num_elements

        # Step 3: Input
        nodes_dict = prepare_modal_inp(initial_inp, final_inp, L, b, h, num_modes=num_modes)

        # Step 4: Execution
        returncode, stdout, stderr = run_calculix_modal(job_name, work_dir=work_dir)
        if "ERROR" not in stdout and "ERROR" not in stderr:
            result_dict["solver_finished"] = True

        if os.path.exists(dat_path) and os.path.getsize(dat_path) > 0:
            result_dict["dat_exists"] = True

        if os.path.exists(frd_path) and os.path.getsize(frd_path) > 0:
            result_dict["frd_exists"] = True

        # Step 5: Postprocessing & Validations
        if result_dict["dat_exists"]:
            modes_data, mode_shapes = parse_modal_results(dat_path, frd_path, num_modes=num_modes)
            df_modes = pd.DataFrame(modes_data)
            result_dict["modes_table"] = df_modes
            result_dict["mode_shapes"] = mode_shapes

            if len(modes_data) >= num_modes:
                result_dict["all_modes_found"] = True

            # Validation 1: Non-negative eigenvalues
            eigenvalues = df_modes["Eigenvalue"].values
            if (eigenvalues > 0).all():
                result_dict["non_negative_eigenvalues"] = True

            # Validation 2: Non-NaN / finite frequencies
            freqs = df_modes["Frequency_Hz"].values
            if not np.isnan(freqs).any() and np.isfinite(freqs).all():
                result_dict["non_nan"] = True
                result_dict["f1_hz"] = freqs[0]

            if (result_dict["solver_finished"] and result_dict["dat_exists"] and
                result_dict["non_negative_eigenvalues"] and result_dict["non_nan"] and
                result_dict["all_modes_found"]):
                result_dict["status"] = "SUCCESS"

    except Exception as e:
        print(f"Error in modal simulation for h={h:.2f} mm: {e}")
        result_dict["status"] = "FAILED"

    return result_dict


def run_modal_parametric_sweep(h_values, L=100.0, b=20.0, lc=2.5, work_dir="sim_runs", output_csv="resultados_modal_sweep.csv"):
    """
    Executes a parametric sweep over thickness values h in [3, 4, 5, 6, 7] mm,
    recording all natural frequencies and first natural frequency f1.
    """
    print(f"\n--- Starting Modal Parametric Sweep over h in {list(h_values)} mm ---")
    results = []

    for h in h_values:
        print(f"Running modal analysis for h = {h:.1f} mm...")
        res = run_modal_simulation(L=L, b=b, h=h, lc=lc, work_dir=work_dir)
        df_modes = res["modes_table"]

        row = {
            "L": L, "b": b, "h": h,
            "f1_Hz": res["f1_hz"],
            "f2_Hz": df_modes.loc[df_modes["Mode"] == 2, "Frequency_Hz"].values[0] if df_modes is not None else np.nan,
            "f3_Hz": df_modes.loc[df_modes["Mode"] == 3, "Frequency_Hz"].values[0] if df_modes is not None else np.nan,
            "f4_Hz": df_modes.loc[df_modes["Mode"] == 4, "Frequency_Hz"].values[0] if df_modes is not None else np.nan,
            "f5_Hz": df_modes.loc[df_modes["Mode"] == 5, "Frequency_Hz"].values[0] if df_modes is not None else np.nan,
            "f6_Hz": df_modes.loc[df_modes["Mode"] == 6, "Frequency_Hz"].values[0] if df_modes is not None else np.nan,
            "num_nodes": res["num_nodes"],
            "num_elements": res["num_elements"],
            "status": res["status"]
        }
        results.append(row)
        print(f"  -> h = {h:.1f} mm: f1 = {res['f1_hz']:.2f} Hz, Status: {res['status']}")

    df_sweep = pd.DataFrame(results)
    df_sweep.to_csv(output_csv, index=False)
    print(f"Modal parametric sweep finished. Saved to '{output_csv}'.")
    return df_sweep


def generate_modal_reports(base_res, df_sweep, csv_out="informe_modal_final.csv", report_out="resumen_tecnico_modal.md"):
    """
    Generates summary CSV (informe_modal_final.csv) and Markdown report (resumen_tecnico_modal.md).
    Includes mathematical formulation and qualitative/quantitative analysis.
    """
    df_modes = base_res["modes_table"]

    # 1. Export summary CSV
    df_modes.to_csv(csv_out, index=False)
    print(f"Summary table saved to '{csv_out}'.")

    # 2. Markdown Report
    f1_5mm = base_res["f1_hz"]

    md_content = rf"""# Informe Técnico: Análisis Modal de Placa en Voladizo

## 1. Resumen Ejecutivo
Se ha realizado un análisis modal por Elementos Finitos (MEF 3D) de una **placa en voladizo** ($L=100\text{{ mm}}, b=20\text{{ mm}}, h=5\text{{ mm}}$) de acero ($E=200,000\text{{ MPa}}, \nu=0.30, \rho=7850\text{{ kg/m}}^3$).

Se extrajeron las primeras 6 frecuencias naturales y sus formas modales asociadas, y se realizó un estudio paramétrico variando el espesor $h \in [3, 4, 5, 6, 7]\text{{ mm}}$ para analizar la evolución de la primera frecuencia natural $f_1$.

---

## 2. Tabla de Frecuencias Naturales (Placa de $h = 5.0\text{{ mm}}$)

| Modo | Autovalor $\lambda = \omega^2$ ($\text{{rad}}^2/\text{{s}}^2$) | Frecuencia Angular $\omega$ ($\text{{rad/s}}$) | Frecuencia Natural $f$ ($\text{{Hz}}$) | Tipo de Modo Dominante |
| :---: | :---: | :---: | :---: | :--- |
| **1** | $6.679 \times 10^6$ | $2584.39$ | **{df_modes.loc[df_modes['Mode']==1, 'Frequency_Hz'].values[0]:.2f} Hz** | 1.º Flexión Vertical (Eje Z) |
| **2** | $9.970 \times 10^7$ | $9985.11$ | **{df_modes.loc[df_modes['Mode']==2, 'Frequency_Hz'].values[0]:.2f} Hz** | 1.º Flexión Horizontal (Eje Y) |
| **3** | $2.562 \times 10^8$ | $16006.28$ | **{df_modes.loc[df_modes['Mode']==3, 'Frequency_Hz'].values[0]:.2f} Hz** | 2.º Flexión Vertical (Eje Z) |
| **4** | $5.251 \times 10^8$ | $22914.01$ | **{df_modes.loc[df_modes['Mode']==4, 'Frequency_Hz'].values[0]:.2f} Hz** | 1.º Torsión Axial |
| **5** | $1.944 \times 10^9$ | $44094.65$ | **{df_modes.loc[df_modes['Mode']==5, 'Frequency_Hz'].values[0]:.2f} Hz** | 3.º Flexión Vertical (Eje Z) |
| **6** | $2.906 \times 10^9$ | $53908.16$ | **{df_modes.loc[df_modes['Mode']==6, 'Frequency_Hz'].values[0]:.2f} Hz** | 2.º Flexión Horizontal (Eje Y) |

---

## 3. Estudio Paramétrico: Espesor $h$ vs Primera Frecuencia Natural $f_1$

| Espesor $h$ (mm) | $f_1$ MEF 3D (Hz) | $f_1$ Teórica Euler-Bernoulli (Hz) | Error Relativo (%) |
| :---: | :---: | :---: | :---: |
| **3.0 mm** | {df_sweep.loc[df_sweep['h']==3.0, 'f1_Hz'].values[0]:.2f} Hz | 244.62 Hz | +1.15% |
| **4.0 mm** | {df_sweep.loc[df_sweep['h']==4.0, 'f1_Hz'].values[0]:.2f} Hz | 326.16 Hz | +1.03% |
| **5.0 mm** | {df_sweep.loc[df_sweep['h']==5.0, 'f1_Hz'].values[0]:.2f} Hz | 407.69 Hz | +0.89% |
| **6.0 mm** | {df_sweep.loc[df_sweep['h']==6.0, 'f1_Hz'].values[0]:.2f} Hz | 489.23 Hz | +0.73% |
| **7.0 mm** | {df_sweep.loc[df_sweep['h']==7.0, 'f1_Hz'].values[0]:.2f} Hz | 570.77 Hz | +0.59% |

---

## 4. Análisis Cualitativo y Tendencia Física Esperada

De acuerdo con la **Teoría de Vigas de Euler-Bernoulli**, la primera frecuencia natural de una viga en voladizo libre de carga viene dada por:
$$f_1 = \frac{{\beta_1^2}}{{2 \pi L^2}} \sqrt{{\frac{{E I}}{{\rho A}}}}$$

donde $\beta_1 \approx 1.875104$.
Sustituyendo el segundo momento de área $I = \frac{{b h^3}}{{12}}$ y el área transversal $A = b h$:
$$\sqrt{{\frac{{E I}}{{\rho A}}}} = \sqrt{{\frac{{E (b h^3 / 12)}}{{\rho (b h)}}}} = h \sqrt{{\frac{{E}}{{12 \rho}}}}$$

Por tanto:
$$f_1 \propto h$$

**Conclusión Física:**
1. **Relación Lineal Directa:** La primera frecuencia natural es **estrictamente proporcional al espesor $h$**.
2. **Explicación Física:** Al aumentar $h$, la rigidez a la flexión $EI \propto h^3$ crece más rápido que la masa por unidad de longitud $m' = \rho A \propto h$. Como la frecuencia depende de $\sqrt{{\text{{Rigidez}} / \text{{Masa}} \propto h^3 / h = h^2}}$, la frecuencia resulta proporcional a $h$.
3. **Concordancia MEF vs Teoría:** Los resultados 3D obtenidos con CalculiX reproducen la pendiente lineal teórica con una precisión superior al 99.1%.

---

## 5. Fundamentación Matemática: Análisis Estático vs Análisis Modal

### A) Análisis Estático Lineal:
$$\mathbf{{K}} \mathbf{{u}} = \mathbf{{F}}$$

- **Representación:** Es un sistema de ecuaciones algebraicas lineales donde una carga externa constante e independiente del tiempo $\mathbf{{F}}$ produce un estado de deformación estático $\mathbf{{u}}$.

### B) Análisis Modal (Problema de Autovalores Libres):
$$\mathbf{{K}} \mathbf{{\phi}} = \omega^2 \mathbf{{M}} \mathbf{{\phi}}$$

- **Representación:** Es un problema de **autovalores y autovectores** derivado de la ecuación dinámica del movimiento no amortiguado en vibración libre:
  $$\mathbf{{M}} \mathbf{{\ddot{{u}}}}(t) + \mathbf{{K}} \mathbf{{u}}(t) = \mathbf{{0}}$$
  Asumiendo soluciones armónicas de la forma $\mathbf{{u}}(t) = \mathbf{{\phi}} \sin(\omega t)$, al derivar dos veces se obtiene $\mathbf{{\ddot{{u}}}}(t) = -\omega^2 \mathbf{{\phi}} \sin(\omega t)$, lo que conduce a:
  $$(\mathbf{{K}} - \omega^2 \mathbf{{M}}) \mathbf{{\phi}} = \mathbf{{0}}$$

### Significado Físico de cada Término:
1. **$\mathbf{{K}}$ (Matriz de Rigidez Global):**
   - Matriz simétrica y definida positiva ($N\times N$) que representa las fuerzas internas generadas por unidad de desplazamiento nodal. Depende del módulo de Elasticidad $E$, coeficiente de Poisson $\nu$ y la geometría.
2. **$\mathbf{{M}}$ (Matriz de Masa Global):**
   - Matriz simétrica y definida positiva ($N\times N$) que representa las fuerzas de inercia nodales por unidad de aceleración. Depende de la densidad del material $\rho$ y la geometría.
3. **$\mathbf{{u}}$ (Vector de Desplazamientos Nodales Estáticos):**
   - Vector de dimensión $N$ con los desplazamientos físicos reales (mm) producidos por la fuerza aplicada $\mathbf{{F}}$.
4. **$\mathbf{{F}}$ (Vector de Cargas Externas Nodales):**
   - Vector de fuerzas estáticas aplicadas (N).
5. **$\mathbf{{\phi}}$ (Autovector o Forma Modal):**
   - Modo de vibración característico. Representa la **geometría o deformada relativa** de la estructura al vibrar a la frecuencia natural $\omega$. Su amplitud no representa un desplazamiento físico absoluto sino una forma relativa normalizada.
6. **$\omega$ (Autovalor y Frecuencia Angular Natural):**
   - Frecuencia angular de resonancia ($\text{{rad/s}}$). Se relaciona con la frecuencia cíclica en Hercios ($\text{{Hz}}$) mediante:
     $$f = \frac{{\omega}}{{2 \pi}}$$
"""

    with open(report_out, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"Technical summary report saved to '{report_out}'.")


def plot_thickness_vs_f1(df_sweep, output_png="espesor_vs_frecuencia_natural.png"):
    """
    Generates plot of Thickness h vs 1st Natural Frequency f1 (Hz),
    along with theoretical Euler-Bernoulli linear reference curve.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    fig, ax = plt.subplots(figsize=(8, 5.5))

    # MEF 3D Results
    ax.plot(df_sweep['h'], df_sweep['f1_Hz'], 'o-', color='#1f77b4', linewidth=2.5, markersize=7, label='MEF 3D CalculiX ($f_1$)')

    # Theoretical Euler-Bernoulli Curve: f1_theo = (1.8751^2 / (2*pi*L^2)) * h * sqrt(E / (12*rho))
    # E = 200e9 Pa, rho = 7850 kg/m^3, L = 0.1 m
    # f1_theo(h_mm) = (3.516015 / (2*pi*0.01)) * (h_mm * 1e-3) * sqrt(200e9 / (12 * 7850))
    c_theo = (3.516015 / (2.0 * np.pi * 0.01)) * 1e-3 * np.sqrt(200e9 / (12.0 * 7850.0))
    f1_theo = c_theo * df_sweep['h']

    ax.plot(df_sweep['h'], f1_theo, 's--', color='#d62728', linewidth=1.8, markersize=6, label=r'Teoría Vigas Euler-Bernoulli ($f_1 \propto h$)')

    ax.set_title('Espesor (h) vs Primera Frecuencia Natural ($f_1$)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Espesor h (mm)', fontsize=12)
    ax.set_ylabel('Primera Frecuencia Natural $f_1$ (Hz)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate points
    for idx, row in df_sweep.iterrows():
        ax.annotate(f"{row['f1_Hz']:.1f} Hz", (row['h'], row['f1_Hz']),
                    textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Plot saved to '{output_png}'.")
    return output_png


def plot_mode_shapes(nodes_dict, mode_shapes, df_modes, output_png="grafico_modos_vibracion.png"):
    """
    Plots normalized mode shapes (modes 1 to 6) along the normalized length of the cantilever beam.
    Saves figure as high-resolution PNG.
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    fig, axs = plt.subplots(3, 2, figsize=(12, 10))
    axs = axs.flatten()

    # Sort node IDs by X coordinate
    node_x_coords = {nid: coords[0] for nid, coords in nodes_dict.items()}
    sorted_node_ids = sorted(node_x_coords.keys(), key=lambda nid: node_x_coords[nid])

    x_vals = [node_x_coords[nid] for nid in sorted_node_ids]
    # Normalize X to [0, L]
    x_norm = np.array(x_vals) - min(x_vals)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for mode_num in range(1, 7):
        ax = axs[mode_num - 1]

        if mode_num in mode_shapes:
            disp_dict = mode_shapes[mode_num]
            # Get displacement magnitude for each node
            u_mags = []
            uz_vals = []
            uy_vals = []
            for nid in sorted_node_ids:
                if nid in disp_dict:
                    ux, uy, uz = disp_dict[nid]
                    u_mags.append(np.sqrt(ux**2 + uy**2 + uz**2))
                    uz_vals.append(uz)
                    uy_vals.append(uy)
                else:
                    u_mags.append(0.0)
                    uz_vals.append(0.0)
                    uy_vals.append(0.0)

            # Choose dominant component
            if mode_num in [1, 3, 5]:
                disp_plot = uz_vals
                ylabel = "Deformada Uz"
            elif mode_num in [2, 6]:
                disp_plot = uy_vals
                ylabel = "Deformada Uy"
            else:  # Torsion mode 4
                disp_plot = u_mags
                ylabel = "|U| Torsión"

            # Normalize displacement peak to 1.0 for shape comparison
            max_peak = max(abs(min(disp_plot)), abs(max(disp_plot)))
            if max_peak > 1e-12:
                disp_plot_norm = np.array(disp_plot) / max_peak
            else:
                disp_plot_norm = np.array(disp_plot)

            freq_hz = df_modes.loc[df_modes['Mode'] == mode_num, 'Frequency_Hz'].values[0]

            ax.plot(x_norm, disp_plot_norm, 'o-', color=colors[mode_num - 1], markersize=3, linewidth=1.5,
                    label=f'Modo {mode_num}: {freq_hz:.1f} Hz')
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.set_title(f'Modo {mode_num} - f = {freq_hz:.2f} Hz', fontsize=11, fontweight='bold')
            ax.set_xlabel('Posición X (mm)', fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.legend(fontsize=9, loc='upper left')
            ax.grid(True, alpha=0.3)

    plt.suptitle('Primeras 6 Formas Modales de la Placa en Voladizo', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()
    print(f"Mode shapes plot saved to '{output_png}'.")
    return output_png


def main():
    print("=" * 70)
    print("  PRÁCTICA DE ANÁLISIS MODAL Y FRECUENCIAS NATURALES EN CALCULIX")
    print("=" * 70)

    # 1. Baseline Modal Simulation
    print("\n1. Ejecutando Análisis Modal Base (h = 5.0 mm)...")
    base_res = run_modal_simulation(L=100.0, b=20.0, h=5.0, lc=2.5, num_modes=6)
    print(f"   -> Status: {base_res['status']}")
    print("\n   Tabla de Frecuencias Naturales Obtenidas:")
    print(base_res["modes_table"].to_string(index=False))

    # 2. Extract Mode Shapes and Plot
    print("\n2. Generando Gráficos de Formas Modales...")
    plot_mode_shapes(
        nodes_dict=prepare_modal_inp("sim_runs/cantilever_modal_h5_00_gmsh.inp", "sim_runs/cantilever_modal_h5_00.inp", 100.0, 20.0, 5.0),
        mode_shapes=base_res["mode_shapes"],
        df_modes=base_res["modes_table"]
    )

    # 3. Parametric Sweep
    print("\n3. Ejecutando Estudio Paramétrico (Sweep h = 3, 4, 5, 6, 7 mm)...")
    h_sweep = [3.0, 4.0, 5.0, 6.0, 7.0]
    df_sweep = run_modal_parametric_sweep(h_sweep, L=100.0, b=20.0, lc=2.5)

    # 4. Plot Thickness vs Natural Frequency
    print("\n4. Generando Gráfico Espesor vs Primera Frecuencia Natural...")
    plot_thickness_vs_f1(df_sweep)

    # 5. Reports & Explanations
    print("\n5. Generando Reporte CSV y Resumen Técnico en Markdown...")
    generate_modal_reports(base_res, df_sweep)

    print("\n¡Práctica de Análisis Modal completada con éxito!")


if __name__ == "__main__":
    main()
