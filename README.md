# TFM: Sistema Híbrido de Recomendación de Recursos Educativos en LMS

## Estructura del proyecto

```
tfm_project/
├── requirements.txt          # Dependencias del proyecto
├── README.md                 # Este archivo
├── config.py                 # Configuración global (semillas, rutas, parámetros)
├── paso1_generar_datos.py    # Paso 1: Genera datos sintéticos con Faker
├── paso2_analisis_exploratorio.py  # Paso 2: Gráficas descriptivas y estadísticas
├── paso3_modelo_contenido.py # Paso 3: Modelo basado en contenido (SBERT + FAISS)
├── paso4_modelo_colaborativo.py    # Paso 4: Modelo de filtrado colaborativo (LightFM)
├── paso5_modelo_hibrido.py   # Paso 5: Modelo híbrido y optimización de α
├── paso6_evaluacion.py       # Paso 6: Evaluación comparativa y pruebas estadísticas
├── paso7_graficas_resultados.py    # Paso 7: Gráficas finales de resultados
├── datos/                    # Carpeta donde se guardan los CSVs generados
│   ├── usuarios.csv
│   ├── recursos.csv
│   └── interacciones.csv
└── graficas/                 # Carpeta donde se guardan las gráficas PNG
```

## Instalación

Este proyecto requiere **Python 3.11** debido a dependencias específicas de C-extensions en `lightfm-next` y `faiss-cpu`. 
Se recomienda el uso de **uv** para la gestión de dependencias.

### Opción A: Usando uv (Recomendado)

1. **Inicializar el entorno y sincronizar:**
```bash
uv python install 3.11
uv python pin 3.11
uv sync
```

3. Ejecutar los pasos en orden:
```bash
uv run paso1_generar_datos.py
uv run paso2_analisis_exploratorio.py
uv run paso3_modelo_contenido.py
uv run paso4_modelo_colaborativo.py
uv run paso5_modelo_hibrido.py
uv run paso6_evaluacion.py
uv run paso7_graficas_resultados.py
```

### Opción B: Usando pip tradicional
1. Crear entorno virtual con python 3.11:
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
# o bien: venv\Scripts\activate  # Windows
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecutar los pasos en orden:
```bash
python paso1_generar_datos.py
python paso2_analisis_exploratorio.py
python paso3_modelo_contenido.py
python paso4_modelo_colaborativo.py
python paso5_modelo_hibrido.py
python paso6_evaluacion.py
python paso7_graficas_resultados.py
```

## Nota sobre reproducibilidad

Todos los scripts usan la semilla `SEED = 42` definida en `config.py`.
Esto garantiza que los resultados sean idénticos cada vez que se ejecute.
