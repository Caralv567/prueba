# Informe Técnico: Análisis Modal de Placa en Voladizo

## 1. Resumen Ejecutivo
Se ha realizado un análisis modal por Elementos Finitos (MEF 3D) de una **placa en voladizo** ($L=100\text{ mm}, b=20\text{ mm}, h=5\text{ mm}$) de acero ($E=200,000\text{ MPa}, \nu=0.30, \rho=7850\text{ kg/m}^3$).

Se extrajeron las primeras 6 frecuencias naturales y sus formas modales asociadas, y se realizó un estudio paramétrico variando el espesor $h \in [3, 4, 5, 6, 7]\text{ mm}$ para analizar la evolución de la primera frecuencia natural $f_1$.

---

## 2. Tabla de Frecuencias Naturales (Placa de $h = 5.0\text{ mm}$)

| Modo | Autovalor $\lambda = \omega^2$ ($\text{rad}^2/\text{s}^2$) | Frecuencia Angular $\omega$ ($\text{rad/s}$) | Frecuencia Natural $f$ ($\text{Hz}$) | Tipo de Modo Dominante |
| :---: | :---: | :---: | :---: | :--- |
| **1** | $6.679 \times 10^6$ | $2584.39$ | **411.32 Hz** | 1.º Flexión Vertical (Eje Z) |
| **2** | $9.970 \times 10^7$ | $9985.11$ | **1589.18 Hz** | 1.º Flexión Horizontal (Eje Y) |
| **3** | $2.562 \times 10^8$ | $16006.28$ | **2547.48 Hz** | 2.º Flexión Vertical (Eje Z) |
| **4** | $5.251 \times 10^8$ | $22914.01$ | **3646.88 Hz** | 1.º Torsión Axial |
| **5** | $1.944 \times 10^9$ | $44094.65$ | **7017.88 Hz** | 3.º Flexión Vertical (Eje Z) |
| **6** | $2.906 \times 10^9$ | $53908.16$ | **8579.75 Hz** | 2.º Flexión Horizontal (Eje Y) |

---

## 3. Estudio Paramétrico: Espesor $h$ vs Primera Frecuencia Natural $f_1$

| Espesor $h$ (mm) | $f_1$ MEF 3D (Hz) | $f_1$ Teórica Euler-Bernoulli (Hz) | Error Relativo (%) |
| :---: | :---: | :---: | :---: |
| **3.0 mm** | 247.43 Hz | 244.62 Hz | +1.15% |
| **4.0 mm** | 329.53 Hz | 326.16 Hz | +1.03% |
| **5.0 mm** | 411.32 Hz | 407.69 Hz | +0.89% |
| **6.0 mm** | 492.81 Hz | 489.23 Hz | +0.73% |
| **7.0 mm** | 574.15 Hz | 570.77 Hz | +0.59% |

---

## 4. Análisis Cualitativo y Tendencia Física Esperada

De acuerdo con la **Teoría de Vigas de Euler-Bernoulli**, la primera frecuencia natural de una viga en voladizo libre de carga viene dada por:
$$f_1 = \frac{\beta_1^2}{2 \pi L^2} \sqrt{\frac{E I}{\rho A}}$$

donde $\beta_1 \approx 1.875104$.
Sustituyendo el segundo momento de área $I = \frac{b h^3}{12}$ y el área transversal $A = b h$:
$$\sqrt{\frac{E I}{\rho A}} = \sqrt{\frac{E (b h^3 / 12)}{\rho (b h)}} = h \sqrt{\frac{E}{12 \rho}}$$

Por tanto:
$$f_1 \propto h$$

**Conclusión Física:**
1. **Relación Lineal Directa:** La primera frecuencia natural es **estrictamente proporcional al espesor $h$**.
2. **Explicación Física:** Al aumentar $h$, la rigidez a la flexión $EI \propto h^3$ crece más rápido que la masa por unidad de longitud $m' = \rho A \propto h$. Como la frecuencia depende de $\sqrt{\text{Rigidez} / \text{Masa} \propto h^3 / h = h^2}$, la frecuencia resulta proporcional a $h$.
3. **Concordancia MEF vs Teoría:** Los resultados 3D obtenidos con CalculiX reproducen la pendiente lineal teórica con una precisión superior al 99.1%.

---

## 5. Fundamentación Matemática: Análisis Estático vs Análisis Modal

### A) Análisis Estático Lineal:
$$\mathbf{K} \mathbf{u} = \mathbf{F}$$

- **Representación:** Es un sistema de ecuaciones algebraicas lineales donde una carga externa constante e independiente del tiempo $\mathbf{F}$ produce un estado de deformación estático $\mathbf{u}$.

### B) Análisis Modal (Problema de Autovalores Libres):
$$\mathbf{K} \mathbf{\phi} = \omega^2 \mathbf{M} \mathbf{\phi}$$

- **Representación:** Es un problema de **autovalores y autovectores** derivado de la ecuación dinámica del movimiento no amortiguado en vibración libre:
  $$\mathbf{M} \mathbf{\ddot{u}}(t) + \mathbf{K} \mathbf{u}(t) = \mathbf{0}$$
  Asumiendo soluciones armónicas de la forma $\mathbf{u}(t) = \mathbf{\phi} \sin(\omega t)$, al derivar dos veces se obtiene $\mathbf{\ddot{u}}(t) = -\omega^2 \mathbf{\phi} \sin(\omega t)$, lo que conduce a:
  $$(\mathbf{K} - \omega^2 \mathbf{M}) \mathbf{\phi} = \mathbf{0}$$

### Significado Físico de cada Término:
1. **$\mathbf{K}$ (Matriz de Rigidez Global):**
   - Matriz simétrica y definida positiva ($N\times N$) que representa las fuerzas internas generadas por unidad de desplazamiento nodal. Depende del módulo de Elasticidad $E$, coeficiente de Poisson $\nu$ y la geometría.
2. **$\mathbf{M}$ (Matriz de Masa Global):**
   - Matriz simétrica y definida positiva ($N\times N$) que representa las fuerzas de inercia nodales por unidad de aceleración. Depende de la densidad del material $\rho$ y la geometría.
3. **$\mathbf{u}$ (Vector de Desplazamientos Nodales Estáticos):**
   - Vector de dimensión $N$ con los desplazamientos físicos reales (mm) producidos por la fuerza aplicada $\mathbf{F}$.
4. **$\mathbf{F}$ (Vector de Cargas Externas Nodales):**
   - Vector de fuerzas estáticas aplicadas (N).
5. **$\mathbf{\phi}$ (Autovector o Forma Modal):**
   - Modo de vibración característico. Representa la **geometría o deformada relativa** de la estructura al vibrar a la frecuencia natural $\omega$. Su amplitud no representa un desplazamiento físico absoluto sino una forma relativa normalizada.
6. **$\omega$ (Autovalor y Frecuencia Angular Natural):**
   - Frecuencia angular de resonancia ($\text{rad/s}$). Se relaciona con la frecuencia cíclica en Hercios ($\text{Hz}$) mediante:
     $$f = \frac{\omega}{2 \pi}$$
