# Práctica de Ingeniería Computacional: MEF 3D, Análisis de Fatiga y Machine Learning

## 1. Resumen Ejecutivo y Flujo del Proyecto
Este proyecto implementa una **plataforma integrada de ingeniería computacional** que combina:
1. **Generación de datos físicos fundamentados** mediante **Análisis por Elementos Finitos (MEF 3D)** en CalculiX para una placa en voladizo ($L=100\text{ mm}, b=20\text{ mm}, h=5\text{ mm}$) de acero.
2. **Evaluación de resistencia a la fatiga** bajo carga cíclica pulsante ($R=0$) utilizando el **Criterio de Goodman**.
3. **Modelado predictivo mediante Machine Learning (scikit-learn)** para entrenar un modelo sustituto (*surrogate model*) que aproxime la respuesta mecánica ($\mathbf{u}_{max}$, $\mathbf{\sigma}_{vm}$, $\mathbf{U}_{Goodman}$) a partir de la carga de entrada $P$.

---

## 2. Configuración y Propiedades de Material

### Geometría y Propiedades Elásticas:
- **Longitud $L$:** $100.0\text{ mm}$
- **Ancho $b$:** $20.0\text{ mm}$
- **Espesor $h$:** $5.0\text{ mm}$
- **Módulo de Elasticidad $E$:** $200,000.0\text{ MPa}$ ($200\text{ GPa}$)
- **Coeficiente de Poisson $\nu$:** $0.30$
- **Densidad del Acero $\rho$:** $7.85 \times 10^-9\text{ t/mm}^3$ ($7850\text{ kg/m}^3$)

### Propiedades Mecánicas de Fatiga (Declaradas en `config.py`):
*Nota: Propiedades de ejemplo representativas para acero estructural comercial (por ejemplo, AISI 1045 / S355).*
- **Resistencia Última a la Tracción ($S_{ut}$):** $600.0\text{ MPa}$
- **Tensión de Fluencia ($S_y$):** $350.0\text{ MPa}$
- **Límite de Fatiga Corregido ($S_e$):** $250.0\text{ MPa}$
- **Relación de Carga Cíclica ($R = \sigma_{min}/\sigma_{max}$):** $0.0$ (Carga pulsante $0 \to P$)

---

## 3. Dataset Generado por MEF y Evaluación de Fatiga (Goodman)

Las simulaciones MEF 3D automatizadas para cargas $P \in [50, 75, 100, 125, 150, 175, 200]\text{ N}$ produjeron los siguientes resultados:

| Carga $P$ (N) | $\sigma_{max}$ (MPa) | $\sigma_{min}$ (MPa) | $\sigma_a$ (MPa) | $\sigma_m$ (MPa) | Desplazamiento $U_{max}$ (mm) | Von Mises $\sigma_{vm}$ (MPa) | Factor Utilización Goodman | Estado Fatiga |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **50.0 N** | 76.09 | 0.00 | 38.04 | 38.04 | 0.3944 mm | 57.45 | **0.2156** | **✅ PASS** |
| **75.0 N** | 114.13 | 0.00 | 57.07 | 57.07 | 0.5916 mm | 86.18 | **0.3234** | **✅ PASS** |
| **100.0 N** | 152.18 | 0.00 | 76.09 | 76.09 | 0.7888 mm | 114.91 | **0.4312** | **✅ PASS** |
| **125.0 N** | 190.22 | 0.00 | 95.11 | 95.11 | 0.9860 mm | 143.63 | **0.5390** | **✅ PASS** |
| **150.0 N** | 228.27 | 0.00 | 114.13 | 114.13 | 1.1832 mm | 172.36 | **0.6468** | **✅ PASS** |
| **175.0 N** | 266.31 | 0.00 | 133.16 | 133.16 | 1.3804 mm | 201.09 | **0.7546** | **✅ PASS** |
| **200.0 N** | 304.36 | 0.00 | 152.18 | 152.18 | 1.5776 mm | 229.81 | **0.8623** | **✅ PASS** |

---

## 4. Fundamentación Matemática del Criterio de Fatiga de Goodman

Para una carga cíclica con tensión máxima $\sigma_{max}$ y tensión mínima $\sigma_{min} = R \cdot \sigma_{max}$:
1. **Amplitud de Tensión ($\sigma_a$):**
   $$\sigma_a = \frac{\sigma_{max} - \sigma_{min}}{2} = \frac{\sigma_{max} (1 - R)}{2}$$
2. **Tensión Media ($\sigma_m$):**
   $$\sigma_m = \frac{\sigma_{max} + \sigma_{min}}{2} = \frac{\sigma_{max} (1 + R)}{2}$$
3. **Línea de Goodman Criterio de Vida Infinita:**
   $$\frac{\sigma_a}{S_e} + \frac{\sigma_m}{S_{ut}} = U_{Goodman}$$

**Criterio de Aceptación:**
- Si $U_{Goodman} \le 1.0 \implies \text{PASS (Resistencia a fatiga por vida infinita garantizada)}$
- Si $U_{Goodman} > 1.0 \implies \text{FAIL (Riesgo de fallo por fatiga antes del límite de vida infinita)}$

Para $P = 200\text{ N}$, $U_{Goodman} = 0.8623 \le 1.0$, garantizando vida infinita para todos los casos de carga analizados.

---

## 5. Comparación y Evaluación de Machine Learning (scikit-learn)

Se dividió el dataset en conjunto de entrenamiento ($72\%$) y prueba ($28\%$) **sin fuga de datos (*data leakage*)**:

| Variable Objetivo | Modelo ML | MAE (Error Absoluto Medio) | RMSE (Raíz Error Cuadrático Medio) | $R^2$ Score (Coef. Determinación) |
| :--- | :---: | :---: | :---: | :---: |
| **Desplazamiento ($U_{max}$)** | Regresión Lineal | $3.54 \times 10^-7\text{ mm}$ | $3.84 \times 10^-7\text{ mm}$ | **1.0000** |
| **Desplazamiento ($U_{max}$)** | Random Forest | $0.3767\text{ mm}$ | $0.3893\text{ mm}$ | $-14.59$ |
| **Tensión Von Mises ($\sigma_{vm}$)** | Regresión Lineal | $1.93 \times 10^-4\text{ MPa}$ | $1.93 \times 10^-4\text{ MPa}$ | **1.0000** |
| **Tensión Von Mises ($\sigma_{vm}$)** | Random Forest | $54.87\text{ MPa}$ | $56.72\text{ MPa}$ | $-14.59$ |
| **Utilización Goodman ($U_{G}$)** | Regresión Lineal | $7.08 \times 10^-7$ | $7.22 \times 10^-7$ | **1.0000** |
| **Utilización Goodman ($U_{G}$)** | Random Forest | $0.2059$ | $0.2128$ | $-14.59$ |

### Lección de Ingeniería en Machine Learning:
- **Regresión Lineal:** Obtiene un $R^2 = 1.0000$ perfecto debido a que las ecuaciones físicas subyacentes de la elasticidad lineal ($\mathbf{u} \propto P$ y $\mathbf{\sigma} \propto P$) son funciones lineales continuas.
- **Random Forest:** Muestra que los modelos basados en árboles de decisión no pueden extrapolar tendencias lineales fuera del rango de datos de entrenamiento ($P_{test} \in [50, 75]\text{ N}$ queda fuera de $P_{train} \in [100, 200]\text{ N}$). Esto demuestra por qué **los modelos de ML deben seleccionarse según la física subyacente del problema** y no deben presentarse como sustitutos ciegos de la FEA.

---

## 6. Validación Física y Coherencia Teórica

1. **Equilibrio Estático Global:**
   - Para cada simulación, la suma de las reacciones en el soporte empotrado ($\sum RF_z$) es exactamente igual a la carga aplicada $P$ en dirección opuesta (diferencia $<0.01\%$).
2. **Relación Lineal Carga-Respuesta:**
   - La teoría de Euler-Bernoulli establece:
     $$v_{max} = \frac{P L^3}{3 E I} \propto P, \quad \sigma_{max} = \frac{P L (h/2)}{I} \propto P$$
   - Los resultados del MEF 3D verifican la estricta linealidad y monotonicidad respecto a $P$.
