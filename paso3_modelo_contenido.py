"""
=============================================================================
paso3_modelo_contenido.py — Modelo basado en contenido (SBERT + FAISS)
=============================================================================

PROPÓSITO:
    Implementar el modelo basado en contenido del sistema de recomendación:
    Este modelo:
    
    1. Toma las descripciones textuales de los 500 recursos educativos
    2. Las convierte en vectores numéricos de 384 dimensiones usando SBERT
    3. Indexa estos vectores en FAISS para búsqueda rápida de similitud
    4. Para cada usuario, construye un "perfil" promediando los embeddings
       de los recursos con los que ha interactuado (ponderados por rating)
    5. Recomienda los recursos más similares al perfil del usuario

¿POR QUÉ SBERT?
    BERT por sí solo no puede comparar oraciones eficientemente: encontrar
    el par más similar entre 10,000 oraciones tomaría ~65 horas. SBERT
    (Sentence-BERT) resuelve esto con una arquitectura siamesa que genera
    embeddings de oraciones directamente comparables mediante similitud
    del coseno, reduciendo el tiempo a ~5 segundos (Reimers & Gurevych, 2019).

¿POR QUÉ FAISS?
    Con 500 recursos y embeddings de 384 dimensiones, podríamos usar
    búsqueda por fuerza bruta. Pero FAISS (Facebook AI Similarity Search)
    nos da una interfaz limpia y escalable. Usamos IndexFlatIP (producto
    interno = similitud del coseno con vectores normalizados) para búsqueda
    exacta. Si el catálogo creciera a millones, FAISS permite cambiar a
    índices aproximados (IVF, HNSW) sin modificar el resto del código.

SALIDAS:
    - modelos/embeddings_recursos.npy    (matriz 500 × 384)
    - modelos/faiss_index.bin            (índice FAISS)
    - modelos/perfiles_usuario.npy       (matriz 1000 × 384)
    - graficas/11_similitud_embeddings_tsne.png
    - graficas/12_similitud_entre_temas.png

EJECUCIÓN:
    python paso3_modelo_contenido.py
    (requiere haber ejecutado paso1_generar_datos.py primero)
=============================================================================
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from config import (
    fijar_semillas, SEED,
    RUTA_USUARIOS, RUTA_RECURSOS, RUTA_INTERACCIONES,
    MODELOS_DIR, GRAFICAS_DIR,
    SBERT_MODELO, SBERT_DIMENSIONES,
    PROP_ENTRENAMIENTO, ESTILO_MATPLOTLIB, DPI
)

fijar_semillas(SEED)

print("=" * 70)
print("PASO 3: Modelo basado en contenido (SBERT + FAISS)")
print("=" * 70)

# =============================================================================
# PASO 3.1: Cargar datos
# =============================================================================
print("\n Cargando datos...")
df_usuarios = pd.read_csv(RUTA_USUARIOS)
df_recursos = pd.read_csv(RUTA_RECURSOS)
df_interacciones = pd.read_csv(RUTA_INTERACCIONES)
df_interacciones['timestamp'] = pd.to_datetime(df_interacciones['timestamp'])

print(f"   Usuarios: {len(df_usuarios):,}")
print(f"   Recursos: {len(df_recursos):,}")
print(f"   Interacciones: {len(df_interacciones):,}")

# =============================================================================
# PASO 3.2: Partición temporal de datos
# =============================================================================
# Dividimos las interacciones en entrenamiento (80% más antiguas) y
# prueba (20% más recientes). Usamos partición TEMPORAL, no aleatoria,
# porque simula el escenario real: entrenamos con el pasado, evaluamos
# con el futuro.
# =============================================================================
print("\n  Particionando datos temporalmente (80/20)...")

df_interacciones = df_interacciones.sort_values('timestamp').reset_index(drop=True)
corte = int(len(df_interacciones) * PROP_ENTRENAMIENTO)

df_train = df_interacciones.iloc[:corte].copy()
df_test = df_interacciones.iloc[corte:].copy()

print(f"   Entrenamiento: {len(df_train):,} interacciones")
print(f"   Prueba:        {len(df_test):,} interacciones")
print(f"   Fecha de corte: {df_train['timestamp'].max().strftime('%Y-%m-%d')}")

# Guardar la partición para que los siguientes pasos usen exactamente la misma
df_train.to_csv(os.path.join(MODELOS_DIR, 'train.csv'), index=False)
df_test.to_csv(os.path.join(MODELOS_DIR, 'test.csv'), index=False)
print("    Partición guardada en modelos/train.csv y modelos/test.csv")

# =============================================================================
# PASO 3.3: Generar embeddings con SBERT
# =============================================================================
# Aquí ocurre la "magia" del modelo basado en contenido:
# Cada recurso tiene un título, una descripción y un tema. Concatenamos
# estos tres campos en un solo texto y se lo pasamos a SBERT.
# SBERT lo convierte en un vector de 384 números reales.
#
# ¿Qué significan esos 384 números?
# Son una representación semántica del contenido. Recursos con descripciones
# similares tendrán vectores similares (cercanos en el espacio), incluso si
# usan palabras diferentes. Por ejemplo, un recurso sobre "redes neuronales"
# estará cerca de uno sobre "deep learning" porque SBERT entiende que son
# conceptos relacionados.
# =============================================================================
print(f"\n Generando embeddings con SBERT ({SBERT_MODELO})...")
print(f"   Esto puede tardar unos segundos la primera vez (descarga del modelo)...")

from sentence_transformers import SentenceTransformer

# Cargar el modelo preentrenado
modelo_sbert = SentenceTransformer(SBERT_MODELO)

# Preparar textos: concatenar título + descripción + tema para cada recurso
# Esto le da a SBERT el máximo contexto posible sobre cada recurso
textos_recursos = []
for _, recurso in df_recursos.iterrows():
    texto = f"{recurso['titulo']}. {recurso['descripcion']}. Tema: {recurso['tema']}"
    textos_recursos.append(texto)

print(f"   Ejemplo de texto a procesar:")
print(f"   '{textos_recursos[0][:120]}...'")

# Generar embeddings para todos los recursos
# show_progress_bar=True muestra una barra de progreso
embeddings_recursos = modelo_sbert.encode(
    textos_recursos,
    show_progress_bar=True,
    normalize_embeddings=True,  # Normalizar a longitud 1 (para similitud del coseno)
    batch_size=64
)

# Convertir a numpy array
embeddings_recursos = np.array(embeddings_recursos, dtype=np.float32)

print(f"\n    Embeddings generados: {embeddings_recursos.shape}")
print(f"   Dimensiones: {embeddings_recursos.shape[1]}")
print(f"   Norma del primer vector: {np.linalg.norm(embeddings_recursos[0]):.4f} (≈1.0 si normalizado)")

# Guardar embeddings
np.save(os.path.join(MODELOS_DIR, 'embeddings_recursos.npy'), embeddings_recursos)
print(f"    Embeddings guardados en modelos/embeddings_recursos.npy")


# =============================================================================
# PASO 3.4: Indexar en FAISS
# =============================================================================
# FAISS permite buscar los vecinos más cercanos de un vector de forma
# eficiente. Usamos IndexFlatIP (Inner Product = producto interno).
# Con vectores normalizados, el producto interno es equivalente a la
# similitud del coseno: cos(u,v) = u·v / (||u|| × ||v||) = u·v cuando
# ||u|| = ||v|| = 1.
# =============================================================================
print("\n Indexando embeddings en FAISS...")

import faiss

# Crear índice de producto interno (equivalente a similitud del coseno)
dimension = embeddings_recursos.shape[1]  # 384
index = faiss.IndexFlatIP(dimension)

# Agregar todos los vectores de recursos al índice
index.add(embeddings_recursos)

print(f"    Índice FAISS creado con {index.ntotal} vectores de {dimension} dimensiones")

# Verificación rápida: buscar los 5 recursos más similares al primer recurso
D, I = index.search(embeddings_recursos[0:1], 6)  # 6 porque el primero es él mismo
print(f"\n   Verificación — Recurso R0001 ({df_recursos.iloc[0]['tema']}):")
print(f"   Recursos más similares:")
for rank, (idx, score) in enumerate(zip(I[0], D[0])):
    if rank == 0:
        continue  # Saltar el recurso mismo
    recurso = df_recursos.iloc[idx]
    print(f"     {rank}. {recurso['resource_id']} ({recurso['tema']}) — similitud: {score:.4f}")

# Guardar índice FAISS
faiss.write_index(index, os.path.join(MODELOS_DIR, 'faiss_index.bin'))
print(f"\n    Índice FAISS guardado en modelos/faiss_index.bin")


# =============================================================================
# PASO 3.5: Construir perfiles de usuario
# =============================================================================
# El perfil de un usuario es un vector de 384 dimensiones que representa
# sus preferencias. Se calcula como el PROMEDIO PONDERADO de los embeddings
# de los recursos con los que ha interactuado en el conjunto de entrenamiento.
#
# ¿Por qué ponderado por rating?
# Porque un recurso que el usuario calificó con 5 refleja mejor sus
# preferencias que uno que calificó con 1. Al ponderar, el perfil se
# "acerca" más a los recursos que el usuario valoró positivamente.
#
# Fórmula:
#   perfil(u) = Σ (rating_i × embedding_i) / Σ rating_i
#   para todos los recursos i con los que u interactuó en entrenamiento
# =============================================================================
print("\n Construyendo perfiles de usuario...")

# Crear diccionario de resource_id -> índice en la matriz de embeddings
resource_to_idx = {rid: i for i, rid in enumerate(df_recursos['resource_id'])}

# Crear diccionario de user_id -> índice
user_ids = df_usuarios['user_id'].tolist()
user_to_idx = {uid: i for i, uid in enumerate(user_ids)}

# Inicializar matriz de perfiles
perfiles_usuario = np.zeros((len(df_usuarios), SBERT_DIMENSIONES), dtype=np.float32)

# Para cada usuario, calcular su perfil con las interacciones de ENTRENAMIENTO
usuarios_sin_perfil = 0
for user_id in user_ids:
    # Obtener interacciones del usuario en el conjunto de entrenamiento
    inter_usuario = df_train[df_train['user_id'] == user_id]
    
    if len(inter_usuario) == 0:
        usuarios_sin_perfil += 1
        continue
    
    # Obtener embeddings y ratings de los recursos interactuados
    embeddings_interactuados = []
    ratings = []
    
    for _, inter in inter_usuario.iterrows():
        rid = inter['resource_id']
        if rid in resource_to_idx:
            idx = resource_to_idx[rid]
            embeddings_interactuados.append(embeddings_recursos[idx])
            ratings.append(inter['rating'])
    
    if embeddings_interactuados:
        embeddings_array = np.array(embeddings_interactuados)
        ratings_array = np.array(ratings, dtype=np.float32)
        
        # Promedio ponderado por rating
        perfil = np.average(embeddings_array, axis=0, weights=ratings_array)
        
        # Normalizar el perfil para que tenga norma 1
        norma = np.linalg.norm(perfil)
        if norma > 0:
            perfil = perfil / norma
        
        perfiles_usuario[user_to_idx[user_id]] = perfil

print(f"    Perfiles construidos: {len(user_ids) - usuarios_sin_perfil:,}")
print(f"   Usuarios sin interacciones en entrenamiento: {usuarios_sin_perfil}")

# Guardar perfiles
np.save(os.path.join(MODELOS_DIR, 'perfiles_usuario.npy'), perfiles_usuario)
print(f"    Perfiles guardados en modelos/perfiles_usuario.npy")


# =============================================================================
# PASO 3.6: Función de recomendación basada en contenido
# =============================================================================
# Para generar recomendaciones:
# 1. Tomar el perfil del usuario (vector de 384d)
# 2. Buscar en FAISS los K recursos más similares a ese perfil
# 3. Excluir los recursos que el usuario ya consumió
# 4. Devolver la lista ordenada por similitud
# =============================================================================

def recomendar_contenido(user_id, k=10, excluir_consumidos=True):
    """
    Genera K recomendaciones basadas en contenido para un usuario.
    
    Parámetros:
        user_id: ID del usuario (str, ej: "U0001")
        k: número de recomendaciones a generar
        excluir_consumidos: si True, no recomienda recursos ya consumidos
    
    Retorna:
        Lista de tuplas (resource_id, score) ordenada por similitud descendente
    """
    idx_usuario = user_to_idx[user_id]
    perfil = perfiles_usuario[idx_usuario:idx_usuario+1]  # Shape: (1, 384)
    
    # Recursos ya consumidos en entrenamiento (para excluirlos)
    if excluir_consumidos:
        consumidos = set(df_train[df_train['user_id'] == user_id]['resource_id'])
    else:
        consumidos = set()
    
    # Buscar más recursos de los necesarios para compensar los excluidos
    n_buscar = k + len(consumidos) + 10
    D, I = index.search(perfil, min(n_buscar, len(df_recursos)))
    
    # Filtrar consumidos y construir lista de recomendaciones
    recomendaciones = []
    for idx, score in zip(I[0], D[0]):
        rid = df_recursos.iloc[idx]['resource_id']
        if rid not in consumidos:
            recomendaciones.append((rid, float(score)))
        if len(recomendaciones) >= k:
            break
    
    return recomendaciones


# Demostración: generar recomendaciones para un usuario de ejemplo
print("\n Demostración de recomendaciones basadas en contenido:")
usuario_demo = "U0001"
info_usuario = df_usuarios[df_usuarios['user_id'] == usuario_demo].iloc[0]
print(f"\n   Usuario: {usuario_demo}")
print(f"   Carrera: {info_usuario['carrera']}")
print(f"   Intereses: {info_usuario['intereses']}")
print(f"   Interacciones en entrenamiento: {len(df_train[df_train['user_id'] == usuario_demo])}")

recs = recomendar_contenido(usuario_demo, k=10)
print(f"\n   Top 10 recomendaciones:")
for rank, (rid, score) in enumerate(recs, 1):
    recurso = df_recursos[df_recursos['resource_id'] == rid].iloc[0]
    print(f"   {rank:2d}. {rid} | {recurso['tema']:<35s} | similitud: {score:.4f}")

# Guardar la función y mapeos necesarios para los siguientes pasos
mapeos = {
    'resource_to_idx': resource_to_idx,
    'user_to_idx': user_to_idx,
    'user_ids': user_ids,
}
with open(os.path.join(MODELOS_DIR, 'mapeos.pkl'), 'wb') as f:
    pickle.dump(mapeos, f)
print(f"\n    Mapeos guardados en modelos/mapeos.pkl")


# =============================================================================
# PASO 3.7: Visualizaciones de los embeddings
# =============================================================================
# Estas gráficas permiten verificar visualmente que SBERT está capturando
# las relaciones semánticas entre los recursos.
# =============================================================================
print("\n Generando visualizaciones de embeddings...")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

plt.style.use(ESTILO_MATPLOTLIB)

# --- Gráfica 11: Proyección t-SNE de los embeddings por tema ---
# t-SNE reduce los 384 dimensiones a 2 para poder visualizar.
# Si SBERT funciona bien, los recursos del mismo tema deberían
# aparecer agrupados (clusters) en la visualización.

print("   Calculando t-SNE (puede tardar unos segundos)...")
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings_recursos)

# Asignar colores por tema
temas_unicos = sorted(df_recursos['tema'].unique())
colores_temas = plt.cm.tab20(np.linspace(0, 1, len(temas_unicos)))
tema_to_color = {tema: colores_temas[i] for i, tema in enumerate(temas_unicos)}

fig, ax = plt.subplots(figsize=(14, 10))
for tema in temas_unicos:
    mask = df_recursos['tema'] == tema
    ax.scatter(
        embeddings_2d[mask, 0], embeddings_2d[mask, 1],
        c=[tema_to_color[tema]], label=tema, alpha=0.7, s=50, edgecolors='white', linewidth=0.5
    )

ax.set_title('Proyección t-SNE de Embeddings SBERT por Tema')
ax.set_xlabel('t-SNE Dimensión 1')
ax.set_ylabel('t-SNE Dimensión 2')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '11_similitud_embeddings_tsne.png'), dpi=DPI, bbox_inches='tight')
plt.close()
print("    11_similitud_embeddings_tsne.png")


# --- Gráfica 12: Heatmap de similitud promedio entre temas ---
# Calcula la similitud del coseno promedio entre los embeddings de
# recursos de cada par de temas. Permite ver qué temas son
# "semánticamente cercanos" según SBERT.

print("   Calculando similitud entre temas...")

# Calcular centroide de cada tema (promedio de sus embeddings)
centroides_temas = {}
for tema in temas_unicos:
    mask = df_recursos['tema'] == tema
    centroides_temas[tema] = embeddings_recursos[mask].mean(axis=0)

# Calcular similitud del coseno entre todos los pares de centroides
n_temas = len(temas_unicos)
sim_matrix = np.zeros((n_temas, n_temas))
for i, tema_i in enumerate(temas_unicos):
    for j, tema_j in enumerate(temas_unicos):
        # Similitud del coseno (vectores ya normalizados por SBERT)
        sim = np.dot(centroides_temas[tema_i], centroides_temas[tema_j])
        sim = sim / (np.linalg.norm(centroides_temas[tema_i]) * np.linalg.norm(centroides_temas[tema_j]))
        sim_matrix[i, j] = sim

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(
    sim_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
    xticklabels=temas_unicos, yticklabels=temas_unicos,
    ax=ax, vmin=0.3, vmax=1.0, linewidths=0.5,
    cbar_kws={'label': 'Similitud del Coseno'}
)
ax.set_title('Similitud Semántica Promedio entre Temas (SBERT)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '12_similitud_entre_temas.png'), dpi=DPI, bbox_inches='tight')
plt.close()
print("    12_similitud_entre_temas.png")


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print(f"\n{'=' * 70}")
print("PASO 3 COMPLETADO: Modelo basado en contenido")
print(f"{'=' * 70}")
print(f"  Modelo SBERT:      {SBERT_MODELO}")
print(f"  Dimensiones:       {SBERT_DIMENSIONES}")
print(f"  Recursos indexados: {index.ntotal}")
print(f"  Perfiles creados:  {len(user_ids) - usuarios_sin_perfil}")
print(f"  Archivos generados:")
print(f"    - modelos/embeddings_recursos.npy  ({embeddings_recursos.shape})")
print(f"    - modelos/faiss_index.bin")
print(f"    - modelos/perfiles_usuario.npy     ({perfiles_usuario.shape})")
print(f"    - modelos/train.csv                ({len(df_train):,} filas)")
print(f"    - modelos/test.csv                 ({len(df_test):,} filas)")
print(f"    - modelos/mapeos.pkl")
print(f"    - graficas/11_similitud_embeddings_tsne.png")
print(f"    - graficas/12_similitud_entre_temas.png")
print(f"{'=' * 70}")
