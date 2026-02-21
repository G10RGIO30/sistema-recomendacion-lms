"""
=============================================================================
paso1_generar_datos.py — Generación de datos sintéticos con Faker
=============================================================================

PROPÓSITO:
    Generar un dataset sintético que simula un entorno LMS (Learning Management
    System) con usuarios, recursos educativos e interacciones. Este dataset
    será la base para entrenar y evaluar los modelos de recomendación.

SALIDAS:
    - datos/usuarios.csv      (1,000 registros)
    - datos/recursos.csv      (500 registros)
    - datos/interacciones.csv (50,000 registros)

EJECUCIÓN:
    python paso1_generar_datos.py
=============================================================================
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importar configuración centralizada
from config import (
    fijar_semillas, SEED, N_USUARIOS, N_RECURSOS, N_INTERACCIONES,
    CARRERAS, TEMAS, INTERESES_POR_CARRERA, TIPOS_RECURSO, DIFICULTADES,
    TIPOS_INTERACCION, PROP_INTERACCION_ALINEADA,
    RUTA_USUARIOS, RUTA_RECURSOS, RUTA_INTERACCIONES
)

# =============================================================================
# PASO 0: Fijar semillas para reproducibilidad
# =============================================================================
fijar_semillas(SEED)
fake = Faker('es_MX')  # Generador de datos falsos en español (México)
Faker.seed(SEED)

print("=" * 70)
print("PASO 1: Generación de datos sintéticos")
print("=" * 70)


# =============================================================================
# PASO 1.1: Generar USUARIOS
# =============================================================================
# Cada usuario representa un estudiante inscrito en el LMS.
# La distribución por carrera sigue los porcentajes definidos en el documento:
#   - Ing. Sistemas: 30%, Ciencia de Datos: 25%, Ing. Software: 25%, Telecomunicaciones: 20%
# =============================================================================

def generar_usuarios(n=N_USUARIOS):
    """
    Genera n usuarios sintéticos con perfiles realistas.
    
    La distribución por semestre sigue una distribución normal truncada
    centrada en el semestre 5, simulando la pirámide de matrícula típica
    (más estudiantes en semestres intermedios).
    
    Los intereses de cada usuario se seleccionan de los temas asociados
    a su carrera (3 intereses por usuario), lo cual creará el sesgo
    de interacción realista más adelante.
    """
    print(f"\n Generando {n} usuarios...")
    
    usuarios = []
    
    # Crear lista de carreras según la distribución de probabilidad
    # Ejemplo: si hay 1000 usuarios, 300 serán de Ing. Sistemas, 250 de Ciencia de Datos, etc.
    lista_carreras = []
    for carrera, proporcion in CARRERAS.items():
        cantidad = int(n * proporcion)
        lista_carreras.extend([carrera] * cantidad)
    
    # Ajustar si sobran o faltan por redondeo
    while len(lista_carreras) < n:
        lista_carreras.append(list(CARRERAS.keys())[0])
    lista_carreras = lista_carreras[:n]
    np.random.shuffle(lista_carreras)
    
    for i in range(n):
        carrera = lista_carreras[i]
        
        # Semestre: distribución normal truncada centrada en 5, σ=2.3
        # Esto genera la pirámide de matrícula descrita en el documento
        semestre = int(np.clip(np.random.normal(5, 2.3), 1, 9))
        
        # Seleccionar 3 intereses de los temas asociados a la carrera del usuario
        temas_carrera = INTERESES_POR_CARRERA[carrera]
        intereses = list(np.random.choice(temas_carrera, size=3, replace=False))
        
        # Fecha de registro: entre 2022 y 2025
        fecha_registro = fake.date_between(
            start_date=datetime(2022, 1, 1),
            end_date=datetime(2025, 6, 30)
        )
        
        usuario = {
            "user_id": f"U{i+1:04d}",  # U0001, U0002, ..., U1000
            "nombre": fake.name(),
            "email": fake.email(),
            "carrera": carrera,
            "semestre": semestre,
            "intereses": "|".join(intereses),  # Separados por | para guardar en CSV
            "fecha_registro": fecha_registro,
        }
        usuarios.append(usuario)
    
    df = pd.DataFrame(usuarios)
    
    # Verificar distribución
    print(f"   Distribución por carrera:")
    for carrera, count in df['carrera'].value_counts().items():
        print(f"     {carrera}: {count} ({count/n*100:.1f}%)")
    print(f"   Semestre promedio: {df['semestre'].mean():.1f} (esperado: ~5)")
    
    return df


# =============================================================================
# PASO 1.2: Generar RECURSOS EDUCATIVOS
# =============================================================================
# Cada recurso representa un material educativo disponible en el LMS.
# La descripción textual es crucial porque es lo que SBERT procesará
# para generar los embeddings semánticos.
# =============================================================================

# Plantillas de descripción por tema — estas generan texto realista que
# SBERT puede procesar para entender la semántica de cada recurso
PLANTILLAS_DESCRIPCION = {
    "Programación en Python": [
        "Curso sobre {subtema} en Python. Cubre {detalle} con ejercicios prácticos y ejemplos de código aplicados a proyectos reales.",
        "Tutorial de {subtema} usando Python. Incluye {detalle} y buenas prácticas de programación orientada a objetos.",
        "Guía completa de {subtema} en el lenguaje Python. Explora {detalle} con casos de uso en desarrollo de software.",
    ],
    "Machine Learning": [
        "Introducción a {subtema} en aprendizaje automático. Se estudian {detalle} con implementaciones en scikit-learn.",
        "Material sobre {subtema} aplicado a machine learning. Cubre {detalle} y técnicas de validación de modelos.",
        "Recurso avanzado de {subtema} para machine learning. Analiza {detalle} con datasets reales y métricas de evaluación.",
    ],
    "Bases de Datos": [
        "Fundamentos de {subtema} en bases de datos. Se abordan {detalle} con ejemplos en SQL y NoSQL.",
        "Material sobre {subtema} para gestión de bases de datos. Incluye {detalle} y optimización de consultas.",
        "Guía práctica de {subtema} en sistemas de bases de datos. Cubre {detalle} y modelado relacional.",
    ],
    "Desarrollo Web": [
        "Tutorial de {subtema} para desarrollo web moderno. Incluye {detalle} con frameworks populares como React y Django.",
        "Recurso sobre {subtema} en desarrollo web full-stack. Cubre {detalle} y arquitectura de aplicaciones.",
        "Material de {subtema} aplicado al desarrollo web. Explora {detalle} con HTML5, CSS3 y JavaScript.",
    ],
    "Inteligencia Artificial": [
        "Estudio de {subtema} en inteligencia artificial. Se analizan {detalle} con aplicaciones en visión por computadora.",
        "Recurso sobre {subtema} en IA y sistemas inteligentes. Cubre {detalle} y redes neuronales profundas.",
        "Material avanzado de {subtema} en inteligencia artificial. Incluye {detalle} y procesamiento de imágenes.",
    ],
    "Redes de Computadoras": [
        "Fundamentos de {subtema} en redes de computadoras. Cubre {detalle} y protocolos de comunicación.",
        "Material sobre {subtema} en arquitectura de redes. Incluye {detalle} y configuración de routers y switches.",
        "Guía de {subtema} para ingeniería de redes. Analiza {detalle} y seguridad perimetral.",
    ],
    "Estadística y Probabilidad": [
        "Curso de {subtema} en estadística aplicada. Se cubren {detalle} con ejemplos en R y Python.",
        "Material sobre {subtema} en probabilidad y estadística. Incluye {detalle} y distribuciones de probabilidad.",
        "Recurso de {subtema} para análisis estadístico. Aborda {detalle} y pruebas de hipótesis.",
    ],
    "Estructuras de Datos": [
        "Fundamentos de {subtema} en estructuras de datos. Cubre {detalle} con implementaciones en Python y Java.",
        "Material sobre {subtema} para algoritmos y estructuras de datos. Incluye {detalle} y análisis de complejidad.",
        "Guía de {subtema} en estructuras de datos avanzadas. Analiza {detalle} y optimización algorítmica.",
    ],
    "Sistemas Operativos": [
        "Estudio de {subtema} en sistemas operativos. Cubre {detalle} con ejemplos en Linux y Windows.",
        "Material sobre {subtema} para administración de sistemas operativos. Incluye {detalle} y gestión de procesos.",
        "Recurso de {subtema} en diseño de sistemas operativos. Analiza {detalle} y virtualización.",
    ],
    "Seguridad Informática": [
        "Fundamentos de {subtema} en ciberseguridad. Cubre {detalle} y técnicas de protección de datos.",
        "Material sobre {subtema} en seguridad informática. Incluye {detalle} y análisis de vulnerabilidades.",
        "Guía de {subtema} para seguridad de sistemas. Aborda {detalle} y criptografía aplicada.",
    ],
    "Computación en la Nube": [
        "Introducción a {subtema} en computación en la nube. Cubre {detalle} con AWS, Azure y Google Cloud.",
        "Material sobre {subtema} en servicios cloud. Incluye {detalle} y arquitectura de microservicios.",
        "Recurso de {subtema} para cloud computing. Analiza {detalle} y contenedores Docker y Kubernetes.",
    ],
    "Procesamiento de Lenguaje Natural": [
        "Estudio de {subtema} en procesamiento de lenguaje natural. Cubre {detalle} con transformers y BERT.",
        "Material sobre {subtema} en NLP aplicado. Incluye {detalle} y modelos de lenguaje preentrenados.",
        "Recurso avanzado de {subtema} en PLN. Analiza {detalle} y embeddings semánticos.",
    ],
}

# Subtemas y detalles para dar variedad a las descripciones
SUBTEMAS = {
    "Programación en Python": ["funciones y módulos", "manejo de archivos", "decoradores y generadores",
                                "programación funcional", "manejo de excepciones", "testing"],
    "Machine Learning": ["regresión lineal y logística", "árboles de decisión", "redes neuronales",
                          "clustering y reducción de dimensionalidad", "ensemble methods", "selección de features"],
    "Bases de Datos": ["normalización", "índices y optimización", "transacciones ACID",
                        "modelado ER", "bases de datos distribuidas", "consultas avanzadas"],
    "Desarrollo Web": ["diseño responsive", "APIs RESTful", "autenticación y seguridad",
                        "testing de aplicaciones", "despliegue continuo", "WebSockets"],
    "Inteligencia Artificial": ["búsqueda heurística", "aprendizaje por refuerzo", "redes generativas",
                                 "sistemas expertos", "lógica difusa", "agentes inteligentes"],
    "Redes de Computadoras": ["modelo OSI", "protocolos TCP/IP", "redes inalámbricas",
                               "VPNs y tunneling", "calidad de servicio", "SDN"],
    "Estadística y Probabilidad": ["inferencia estadística", "regresión múltiple", "análisis de varianza",
                                    "series de tiempo", "estadística bayesiana", "muestreo"],
    "Estructuras de Datos": ["árboles binarios", "grafos y algoritmos", "tablas hash",
                              "pilas y colas", "heaps y prioridad", "tries"],
    "Sistemas Operativos": ["gestión de memoria", "sistemas de archivos", "planificación de procesos",
                             "sincronización", "entrada/salida", "sistemas embebidos"],
    "Seguridad Informática": ["criptografía simétrica", "autenticación multifactor", "firewalls",
                               "pentesting", "forense digital", "ingeniería social"],
    "Computación en la Nube": ["infraestructura como servicio", "serverless", "balanceo de carga",
                                "auto-escalado", "almacenamiento distribuido", "CI/CD en la nube"],
    "Procesamiento de Lenguaje Natural": ["tokenización y stemming", "word embeddings", "análisis de sentimiento",
                                           "traducción automática", "generación de texto", "chatbots"],
}


def generar_recursos(n=N_RECURSOS):
    """
    Genera n recursos educativos con descripciones textuales ricas.
    
    Las descripciones son fundamentales porque SBERT las procesará
    para generar los embeddings semánticos. Por eso usamos plantillas
    que producen texto variado y coherente con el tema.
    """
    print(f"\nGenerando {n} recursos educativos...")
    
    recursos = []
    
    # Distribuir recursos entre temas (aproximadamente uniforme)
    temas_lista = []
    recursos_por_tema = n // len(TEMAS)
    residuo = n % len(TEMAS)
    for i, tema in enumerate(TEMAS):
        cantidad = recursos_por_tema + (1 if i < residuo else 0)
        temas_lista.extend([tema] * cantidad)
    np.random.shuffle(temas_lista)
    
    # Generar tipos de recurso según los pesos definidos
    tipos_lista = np.random.choice(
        list(TIPOS_RECURSO.keys()),
        size=n,
        p=list(TIPOS_RECURSO.values())
    )
    
    for i in range(n):
        tema = temas_lista[i]
        tipo = tipos_lista[i]
        
        # Seleccionar plantilla y subtema aleatorios
        plantilla = np.random.choice(PLANTILLAS_DESCRIPCION[tema])
        subtema = np.random.choice(SUBTEMAS[tema])
        detalle = np.random.choice(SUBTEMAS[tema])
        
        # Generar descripción usando la plantilla
        descripcion = plantilla.format(subtema=subtema, detalle=detalle)
        
        # Generar título coherente con el tema
        titulo = f"{subtema.title()} - {tema}"
        
        # Dificultad ponderada: más recursos básicos e intermedios
        dificultad = np.random.choice(
            DIFICULTADES,
            p=[0.35, 0.40, 0.25]  # 35% básico, 40% intermedio, 25% avanzado
        )
        
        # Duración solo para videos (entre 5 y 120 minutos)
        duracion_min = None
        if tipo == "video":
            duracion_min = int(np.random.lognormal(mean=3.3, sigma=0.6))
            duracion_min = int(np.clip(duracion_min, 5, 120))
        
        recurso = {
            "resource_id": f"R{i+1:04d}",
            "titulo": titulo,
            "descripcion": descripcion,
            "tipo": tipo,
            "tema": tema,
            "dificultad": dificultad,
            "duracion_min": duracion_min,
        }
        recursos.append(recurso)
    
    df = pd.DataFrame(recursos)
    
    print(f"   Distribución por tema:")
    for tema, count in df['tema'].value_counts().head(6).items():
        print(f"     {tema}: {count} ({count/n*100:.1f}%)")
    print(f"   ... y {len(TEMAS) - 6} temas más")
    print(f"   Distribución por tipo: {dict(df['tipo'].value_counts())}")
    
    return df


# =============================================================================
# PASO 1.3: Generar INTERACCIONES
# =============================================================================
# Las interacciones son el corazón del sistema de recomendación.
# Aquí implementamos el sesgo realista: 70% de las interacciones
# son entre usuarios y recursos alineados con sus intereses.
# =============================================================================

def generar_interacciones(df_usuarios, df_recursos, n=N_INTERACCIONES):
    """
    Genera n interacciones usuario-recurso con sesgo realista.
    
    LÓGICA DEL SESGO (70/30):
    - Para el 70% de las interacciones:
      Seleccionamos un recurso cuyo TEMA coincida con alguno de los
      INTERESES del usuario. Esto simula que los estudiantes tienden
      a consumir contenido de sus áreas de interés.
    
    - Para el 30% restante:
      Seleccionamos un recurso completamente al azar. Esto simula
      la exploración: a veces un estudiante de Ciencia de Datos
      mira un video de Redes por curiosidad.
    
    LÓGICA DEL RATING:
    - Si la interacción es alineada (tema coincide con interés): 
      rating tiende a ser alto (media 4.0)
    - Si es aleatoria: rating tiende a ser medio (media 3.0)
    - Esto simula que los usuarios valoran mejor el contenido de su interés.
    """
    print(f"\n Generando {n} interacciones...")
    
    interacciones = []
    
    # Precalcular: para cada tema, qué recursos existen
    recursos_por_tema = {}
    for tema in TEMAS:
        recursos_por_tema[tema] = df_recursos[df_recursos['tema'] == tema]['resource_id'].tolist()
    
    todos_los_recursos = df_recursos['resource_id'].tolist()
    
    # Rango de fechas para las interacciones (2022-2025)
    fecha_inicio = datetime(2022, 6, 1)
    fecha_fin = datetime(2025, 12, 31)
    rango_dias = (fecha_fin - fecha_inicio).days
    
    # Generar fechas ordenadas (para poder hacer partición temporal después)
    timestamps = sorted([
        fecha_inicio + timedelta(
            days=int(np.random.uniform(0, rango_dias)),
            hours=int(np.random.uniform(8, 22)),
            minutes=int(np.random.uniform(0, 59))
        )
        for _ in range(n)
    ])
    
    for idx in range(n):
        # Seleccionar usuario al azar
        usuario = df_usuarios.iloc[np.random.randint(0, len(df_usuarios))]
        user_id = usuario['user_id']
        intereses_usuario = usuario['intereses'].split("|")
        
        # Decidir si la interacción es alineada (70%) o aleatoria (30%)
        es_alineada = np.random.random() < PROP_INTERACCION_ALINEADA
        
        if es_alineada:
            # Seleccionar un tema de los intereses del usuario
            tema_elegido = np.random.choice(intereses_usuario)
            # Seleccionar un recurso de ese tema
            recursos_del_tema = recursos_por_tema.get(tema_elegido, [])
            if recursos_del_tema:
                resource_id = np.random.choice(recursos_del_tema)
            else:
                resource_id = np.random.choice(todos_los_recursos)
            # Rating alto para contenido alineado (media 4.0, desv 0.8)
            rating = int(np.clip(np.random.normal(4.0, 0.8), 1, 5))
        else:
            # Seleccionar recurso al azar (exploración)
            resource_id = np.random.choice(todos_los_recursos)
            # Rating medio para contenido aleatorio (media 3.0, desv 1.0)
            rating = int(np.clip(np.random.normal(3.0, 1.0), 1, 5))
        
        # Tipo de interacción (ponderado)
        tipo_interaccion = np.random.choice(
            TIPOS_INTERACCION,
            p=[0.40, 0.25, 0.20, 0.15]  # view es más común
        )
        
        # Tiempo dedicado (solo para videos, en segundos)
        recurso_info = df_recursos[df_recursos['resource_id'] == resource_id].iloc[0]
        tiempo_dedicado = None
        if recurso_info['tipo'] == 'video' and pd.notna(recurso_info['duracion_min']):
            # El usuario ve entre 30% y 100% del video
            proporcion_vista = np.random.uniform(0.3, 1.0)
            tiempo_dedicado = int(recurso_info['duracion_min'] * 60 * proporcion_vista)
        
        interaccion = {
            "user_id": user_id,
            "resource_id": resource_id,
            "tipo_interaccion": tipo_interaccion,
            "rating": rating,
            "tiempo_dedicado": tiempo_dedicado,
            "timestamp": timestamps[idx],
        }
        interacciones.append(interaccion)
        
        # Mostrar progreso cada 10,000 interacciones
        if (idx + 1) % 10_000 == 0:
            print(f"   Generadas {idx + 1:,} / {n:,} interacciones...")
    
    df = pd.DataFrame(interacciones)
    
    # Estadísticas
    print(f"\n   Estadísticas de interacciones:")
    print(f"   Rating promedio: {df['rating'].mean():.2f}")
    print(f"   Rating mediana: {df['rating'].median():.0f}")
    print(f"   Interacciones por usuario (promedio): {n / len(df_usuarios):.0f}")
    print(f"   Usuarios únicos con interacciones: {df['user_id'].nunique()}")
    print(f"   Recursos únicos con interacciones: {df['resource_id'].nunique()}")
    
    return df


# =============================================================================
# PASO 1.4: Validación de integridad
# =============================================================================

def validar_datos(df_usuarios, df_recursos, df_interacciones):
    """
    Ejecuta las validaciones de integridad descritas en el documento.
    Cualquier error aquí indica un bug en la generación.
    """
    print(f"\n Validando integridad de los datos...")
    errores = 0
    
    # 1. Unicidad de identificadores
    if df_usuarios['user_id'].nunique() != len(df_usuarios):
        print("    ERROR: user_id duplicados")
        errores += 1
    else:
        print("    Unicidad de user_id: OK")
    
    if df_recursos['resource_id'].nunique() != len(df_recursos):
        print("    ERROR: resource_id duplicados")
        errores += 1
    else:
        print("    Unicidad de resource_id: OK")
    
    # 2. Integridad referencial
    users_en_inter = set(df_interacciones['user_id'].unique())
    users_validos = set(df_usuarios['user_id'].unique())
    if not users_en_inter.issubset(users_validos):
        print("    ERROR: interacciones referencian usuarios inexistentes")
        errores += 1
    else:
        print("    Integridad referencial usuarios: OK")
    
    resources_en_inter = set(df_interacciones['resource_id'].unique())
    resources_validos = set(df_recursos['resource_id'].unique())
    if not resources_en_inter.issubset(resources_validos):
        print("    ERROR: interacciones referencian recursos inexistentes")
        errores += 1
    else:
        print("    Integridad referencial recursos: OK")
    
    # 3. Rangos válidos
    if df_interacciones['rating'].between(1, 5).all():
        print("    Ratings en rango [1, 5]: OK")
    else:
        print("    ERROR: ratings fuera de rango")
        errores += 1
    
    if df_usuarios['semestre'].between(1, 9).all():
        print("    Semestres en rango [1, 9]: OK")
    else:
        print("    ERROR: semestres fuera de rango")
        errores += 1
    
    if errores == 0:
        print("    TODAS las validaciones pasaron correctamente")
    else:
        print(f"     Se encontraron {errores} errores")
    
    return errores == 0


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Generar datos
    df_usuarios = generar_usuarios()
    df_recursos = generar_recursos()
    df_interacciones = generar_interacciones(df_usuarios, df_recursos)
    
    # Validar
    validar_datos(df_usuarios, df_recursos, df_interacciones)
    
    # Guardar en CSV
    print(f"\n Guardando datos...")
    df_usuarios.to_csv(RUTA_USUARIOS, index=False, encoding='utf-8')
    df_recursos.to_csv(RUTA_RECURSOS, index=False, encoding='utf-8')
    df_interacciones.to_csv(RUTA_INTERACCIONES, index=False, encoding='utf-8')
    
    print(f"    {RUTA_USUARIOS}")
    print(f"    {RUTA_RECURSOS}")
    print(f"    {RUTA_INTERACCIONES}")
    
    # Resumen final
    print(f"\n{'=' * 70}")
    print(f"RESUMEN DE DATOS GENERADOS")
    print(f"{'=' * 70}")
    print(f"  Usuarios:       {len(df_usuarios):,}")
    print(f"  Recursos:       {len(df_recursos):,}")
    print(f"  Interacciones:  {len(df_interacciones):,}")
    print(f"  Densidad:       {len(df_interacciones) / (len(df_usuarios) * len(df_recursos)) * 100:.1f}%")
    print(f"  Rating promedio: {df_interacciones['rating'].mean():.2f}")
    print(f"  Semilla usada:  {SEED}")
    print(f"{'=' * 70}")
