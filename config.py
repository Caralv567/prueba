"""
Configuración de Propiedades de Material y Parámetros de Análisis de Fatiga
-----------------------------------------------------------------------------
NOTA: Las propiedades de fatiga y resistencia mecánica utilizadas en este proyecto
son valores de ejemplo representativos para acero estructural comercial (por ejemplo, S355 / AISI 1045),
declarados explícitamente para fines educativos y de simulación.
"""

# Geometría por defecto de la placa en voladizo (mm)
LENGTH_L = 100.0  # Longitud (mm)
WIDTH_B = 20.0    # Ancho (mm)
THICKNESS_H = 5.0 # Espesor (mm)

# Propiedades elásticas del material
YOUNG_MODULUS_E = 200000.0  # Módulo de Elasticidad (MPa o N/mm^2)
POISSON_RATIO_NU = 0.30     # Coeficiente de Poisson
DENSITY_RHO = 7.85e-9       # Densidad del acero en t/mm^3 (7850 kg/m^3)

# Propiedades mecánicas de resistencia y fatiga (MPa)
S_UT = 600.0  # Resistencia última a la tracción (Ultimate Tensile Strength, MPa)
S_Y = 350.0   # Límite elástico / Tensión de fluencia (Yield Strength, MPa)
S_E = 250.0   # Límite de fatiga corregido (Endurance Limit, MPa)

# Parámetros del ciclo de carga de fatiga
# R_RATIO = sigma_min / sigma_max
# R = 0.0 representa una carga cíclica pulsante (de 0 a P)
R_RATIO = 0.0

# Vector de cargas a evaluar en el estudio MEF (N)
LOAD_CASES_P = [50.0, 75.0, 100.0, 125.0, 150.0, 175.0, 200.0]
