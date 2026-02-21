"""
=============================================================================
config.py — Configuración global del proyecto TFM
=============================================================================
Este archivo centraliza TODOS los parámetros del proyecto en un solo lugar.
Cualquier cambio en la configuración se hace aquí y se propaga a todos los scripts.

¿Por qué centralizar?
- Reproducibilidad: la semilla aleatoria se define una sola vez.
- Consistencia: si cambias el número de usuarios, no tienes que buscarlo
  en 7 archivos diferentes.
- Documentación: aquí queda claro qué parámetros usa el sistema.
=============================================================================
"""

import os
import numpy as np
import random

# =============================================================================
# SEMILLA ALEATORIA GLOBAL
# =============================================================================
# Fijar la semilla garantiza que los números "aleatorios" sean siempre los mismos.
# Usamos 42 por convención (es la semilla más usada en machine learning).
SEED = 42

def fijar_semillas(seed=SEED):
    """
    Fija las semillas de TODOS los generadores aleatorios que usamos.
    Debe llamarse al inicio de cada script.
    """
    random.seed(seed)
    np.random.seed(seed)
    # Si usas PyTorch (sentence-transformers lo usa internamente):
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

# =============================================================================
# RUTAS DE ARCHIVOS
# =============================================================================
# Directorio base del proyecto (donde está este archivo)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carpeta para datos generados
DATOS_DIR = os.path.join(BASE_DIR, "datos")
os.makedirs(DATOS_DIR, exist_ok=True)

# Carpeta para gráficas
GRAFICAS_DIR = os.path.join(BASE_DIR, "graficas")
os.makedirs(GRAFICAS_DIR, exist_ok=True)

# Carpeta para modelos entrenados
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")
os.makedirs(MODELOS_DIR, exist_ok=True)

# Rutas específicas de archivos de datos
RUTA_USUARIOS = os.path.join(DATOS_DIR, "usuarios.csv")
RUTA_RECURSOS = os.path.join(DATOS_DIR, "recursos.csv")
RUTA_INTERACCIONES = os.path.join(DATOS_DIR, "interacciones.csv")

# =============================================================================
# PARÁMETROS DE GENERACIÓN DE DATOS (Paso 1)
# =============================================================================

# Número de entidades — estos valores están en tu documento (Tabla 5)
N_USUARIOS = 1_000
N_RECURSOS = 500
N_INTERACCIONES = 50_000

# Distribución de carreras — tu documento especifica estos porcentajes
CARRERAS = {
    "Ingeniería en Sistemas":      0.30,  # 30%
    "Ciencia de Datos":            0.25,  # 25%
    "Ingeniería en Software":      0.25,  # 25%
    "Ingeniería en Telecomunicaciones": 0.20,  # 20%
}

# 12 temas de recursos educativos — alineados con las carreras
TEMAS = [
    "Programación en Python",
    "Machine Learning",
    "Bases de Datos",
    "Desarrollo Web",
    "Inteligencia Artificial",
    "Redes de Computadoras",
    "Estadística y Probabilidad",
    "Estructuras de Datos",
    "Sistemas Operativos",
    "Seguridad Informática",
    "Computación en la Nube",
    "Procesamiento de Lenguaje Natural",
]

# Mapeo de intereses por carrera (qué temas le interesan más a cada carrera)
# Esto crea el sesgo realista del 70% del que habla tu documento
INTERESES_POR_CARRERA = {
    "Ingeniería en Sistemas": [
        "Programación en Python", "Estructuras de Datos", "Sistemas Operativos",
        "Bases de Datos", "Desarrollo Web", "Redes de Computadoras"
    ],
    "Ciencia de Datos": [
        "Machine Learning", "Estadística y Probabilidad", "Programación en Python",
        "Inteligencia Artificial", "Procesamiento de Lenguaje Natural", "Bases de Datos"
    ],
    "Ingeniería en Software": [
        "Desarrollo Web", "Programación en Python", "Bases de Datos",
        "Estructuras de Datos", "Computación en la Nube", "Seguridad Informática"
    ],
    "Ingeniería en Telecomunicaciones": [
        "Redes de Computadoras", "Seguridad Informática", "Sistemas Operativos",
        "Computación en la Nube", "Programación en Python", "Estadística y Probabilidad"
    ],
}

# Tipos de recursos y sus pesos (probabilidad de aparición)
TIPOS_RECURSO = {
    "video": 0.35,
    "pdf": 0.30,
    "enlace": 0.20,
    "actividad": 0.15,
}

# Niveles de dificultad
DIFICULTADES = ["básico", "intermedio", "avanzado"]

# Tipos de interacción
TIPOS_INTERACCION = ["view", "download", "complete", "like"]

# Proporción de interacciones alineadas vs. aleatorias
PROP_INTERACCION_ALINEADA = 0.70  # 70% según tu documento

# =============================================================================
# PARÁMETROS DE MODELOS (Pasos 3-5)
# =============================================================================

# SBERT
SBERT_MODELO = "all-MiniLM-L6-v2"  # Genera embeddings de 384 dimensiones
SBERT_DIMENSIONES = 384

# LightFM
LIGHTFM_COMPONENTES = 64     # Factores latentes (mencionado en tu documento)
LIGHTFM_LOSS = "warp"         # Optimiza Precision@K directamente
LIGHTFM_EPOCHS = 30           # Épocas de entrenamiento
LIGHTFM_LEARNING_RATE = 0.05
LIGHTFM_REGULARIZATION = 1e-5  # Regularización L2

# Modelo Híbrido
ALPHA_VALORES = [round(x * 0.1, 1) for x in range(11)]  # [0.0, 0.1, ..., 1.0]
N_FOLDS_CV = 5               # Folds para validación cruzada

# =============================================================================
# PARÁMETROS DE EVALUACIÓN (Paso 6)
# =============================================================================

# Valores de K para las métricas
K_VALORES = [5, 10, 20]

# Partición de datos
PROP_ENTRENAMIENTO = 0.80     # 80% entrenamiento, 20% prueba

# Umbral de relevancia: interacciones con rating >= este valor son "relevantes"
UMBRAL_RELEVANCIA = 4

# Cold-start: usuarios con menos de este número de interacciones en entrenamiento
UMBRAL_COLD_START = 5

# Nivel de significancia para pruebas estadísticas
ALPHA_SIGNIFICANCIA = 0.05

# =============================================================================
# PARÁMETROS DE VISUALIZACIÓN (Pasos 2 y 7)
# =============================================================================

# Estilo de gráficas
ESTILO_MATPLOTLIB = "seaborn-v0_8-whitegrid"
FIGSIZE_NORMAL = (10, 6)
FIGSIZE_GRANDE = (14, 8)
FIGSIZE_CUADRADA = (8, 8)
DPI = 150  # Resolución de las gráficas guardadas

# Paleta de colores consistente para los 3 modelos
COLORES_MODELOS = {
    "Basado en Contenido": "#2196F3",   # Azul
    "Filtrado Colaborativo": "#FF9800",  # Naranja
    "Híbrido": "#4CAF50",               # Verde
}

print(" Configuración cargada correctamente.")
print(f"   Semilla: {SEED}")
print(f"   Datos: {N_USUARIOS} usuarios, {N_RECURSOS} recursos, {N_INTERACCIONES} interacciones")
print(f"   Modelo SBERT: {SBERT_MODELO} ({SBERT_DIMENSIONES}d)")
print(f"   LightFM: {LIGHTFM_COMPONENTES} factores, loss={LIGHTFM_LOSS}, {LIGHTFM_EPOCHS} épocas")
