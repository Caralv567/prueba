# Resumen Técnico e Informe de Optimización Paramétrica

## 1. Resumen Ejecutivo
Se ha llevado a cabo la automatización completa del ciclo de análisis y optimización de una **placa en voladizo** sometida a una carga flexional de **100 N** en su extremo libre.

El objetivo principal fue encontrar el **menor espesor $h$** que garantice:
1. **Desplazamiento máximo $U_{max} \le 0.5\text{ mm}$**
2. **Tensión de Von Mises máxima $\sigma_{vm} \le 100\text{ MPa}$**

---

## 2. Comparación de Diseños (Inicial vs Optimizado)

| Métrica / Parámetro | Diseño Inicial ($h=5.00\text{ mm}$) | Diseño Optimizado ($h=5.83\text{ mm}$) | Variación | Límite / Criterio |
| :--- | :---: | :---: | :---: | :---: |
| **Espesor $h$** | 5.00 mm | **5.83 mm** | +16.60% | $3.0 \le h \le 8.0\text{ mm}$ |
| **Desplazamiento Máx. $U_{max}$** | 0.7888 mm | **0.4986 mm** | -36.79% | $\le 0.500\text{ mm}$ |
| **Desplazamiento Prom. Cara Carga** | 0.7885 mm | **0.4983 mm** | - | Metric informativa |
| **Tensión Von Mises Máx.** | 114.91 MPa | **88.19 MPa** | -23.25% | $\le 100.0\text{ MPa}$ |
| **Masa de la Placa** | 78.50 g | **91.53 g** | +16.60% | Mínima posible |
| **Suma Reacciones $RF_z$ (Soporte)** | +100.00 N | **+100.00 N** | 0.00% | Equilibrado con carga -100 N |
| **Cumple Restricciones** | ❌ FALSO | **✅ SÍ (CUMPLE)** | - | Ambas condiciones ok |

---

## 3. Desglose del Flujo Automatizado y Tecnologías Utilizadas

1. **Python (Orquestación y Lógica):**
   - Controla de principio a fin la ejecución del flujo paramétrico, la lectura de archivos de resultados, las comprobaciones automáticas de calidad y seguridad, y el almacenamiento estructurado de datos.

2. **CadQuery (Generación Paramétrica del CAD):**
   - Construye de forma automatizada la geometría 3D en formato STEP a partir de las dimensiones $(L, b, h)$.

3. **Gmsh (Mallado Automatizado 3D):**
   - Importa el modelo STEP, aplica un tamaño de malla controlado ($lc=2.5\text{ mm}$) con elementos sólidos tetraédricos de segundo orden (**C3D10**) para capturar con precisión los gradientes de flexión, y asigna los grupos físicos de contorno (`FIXED_SURF`, `LOAD_SURF`, `EALL`).

4. **CalculiX (`ccx` - Solver MEF):**
   - Resuelve el análisis estructural estático lineal aplicando las propiedades del material ($E=200,000\text{ MPa}, \nu=0.30$), las condiciones de contorno fijas ($U_1=U_2=U_3=0$ en $X=-50\text{ mm}$) y la distribución uniforme de la carga de $100\text{ N}$ en $-Z$ sobre la cara libre ($X=+50\text{ mm}$).

5. **pandas (Gestión de Datos y Reportes):**
   - Consolida las métricas extraídas de cada simulación en un `DataFrame` estructurado, exportando los resultados del *parametric sweep* a `resultados_parametric_sweep.csv` y el informe comparativo a `informe_final.csv`.

6. **SciPy (`scipy.optimize`):**
   - Ejecuta algoritmos de optimización numéricas (`minimize_scalar` con método de acotación) combinados con funciones de penalización para hallar automáticamente el espesor óptimo $h_{opt} = 5.83\text{ mm}$.

---

## 4. Formulación del Problema de Optimización

* **Función Objetivo:**
  $$\min_h \quad f(h) = h \quad \\text{(equivalente a minimizar la masa } m(h) = \\rho \cdot L \cdot b \cdot h \\text{)}$$

* **Variables de Diseño:**
  $$3.0\text{ mm} \le h \le 8.0\text{ mm}$$

* **Restricciones:**
  $$g_1(h) = U_{max}(h) - 0.5\text{ mm} \le 0$$
  $$g_2(h) = \sigma_{vm, max}(h) - 100.0\text{ MPa} \le 0$$

---

## 5. Verificación de Veracidad Física y Razonabilidad

1. **Equilibrio Estático Global:**
   - La suma de reacciones verticales en la cara empotrada ($X=-50\text{ mm}$) es exactamente $+100.00\text{ N}$, contrarrestando la carga externa aplicada de $-100.00\text{ N}$ en el extremo libre ($X=+50\text{ mm}$).

2. **Comparación Teórica (Teoría de Vigas de Euler-Bernoulli):**
   - **Flecha analítica teórica ($h=5\text{ mm}$):**
     $$I = \\frac{b h^3}{12} = 208.33\text{ mm}^4, \quad v_{max} = \\frac{P L^3}{3 E I} = 0.800\text{ mm}$$
     - **Resultado MEF 3D:** $0.7888\text{ mm}$ (diferencia de solo 1.4%, debida al efecto tridimensional y deformación por cortante).
   - **Tensión flexional máxima teórica ($h=5\text{ mm}$):**
     $$\sigma_{max} = \\frac{M y}{I} = \\frac{(100 \cdot 100) \cdot 2.5}{208.33} = 120.00\text{ MPa}$$
     - **Resultado MEF 3D:** $114.91\text{ MPa}$ (diferencia de 4.2%, físicamente consistente).

3. **Verificaciones Automáticas de Código:**
   - Confirmación de finalización exitosa de CalculiX (`returncode == 0`).
   - Existencia y validez del archivo de resultados `.frd`.
   - Ausencia de valores nulos o $NaN$.
   - Verificación de que el número de nodos y elementos sea estricta y físicamente positivo ($>0$).
