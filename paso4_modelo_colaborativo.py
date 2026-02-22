"""
=============================================================================
paso4_modelo_colaborativo.py — Modelo de filtrado colaborativo (LightFM)
=============================================================================

PROPÓSITO:
    Implementar el modelo de filtrado colaborativo del sistema de recomendación
    mediante LightFM.

SALIDAS:
    - modelos/lightfm_model.pkl
    - modelos/lightfm_artefactos.pkl
    - graficas/13_curva_entrenamiento_lightfm.png
    - graficas/14_embeddings_lightfm_tsne.png

EJECUCIÓN:
    python paso4_modelo_colaborativo.py
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

from lightfm import LightFM
from lightfm.data import Dataset

from config import (
    fijar_semillas, SEED,
    RUTA_USUARIOS, RUTA_RECURSOS,
    MODELOS_DIR, GRAFICAS_DIR,
    LIGHTFM_COMPONENTES, LIGHTFM_LOSS, LIGHTFM_EPOCHS,
    LIGHTFM_LEARNING_RATE, LIGHTFM_REGULARIZATION,
    CARRERAS, TEMAS, DIFICULTADES, TIPOS_RECURSO, UMBRAL_RELEVANCIA,
    ESTILO_MATPLOTLIB, FIGSIZE_NORMAL, FIGSIZE_GRANDE, DPI
)

fijar_semillas(SEED)

print("=" * 70)
print("PASO 4: Modelo de filtrado colaborativo (LightFM)")
print("=" * 70)

# =============================================================================
# PASO 4.1: Cargar datos
# =============================================================================
print("\n Cargando datos...")
df_usuarios = pd.read_csv(RUTA_USUARIOS)
df_recursos = pd.read_csv(RUTA_RECURSOS)
df_train = pd.read_csv(os.path.join(MODELOS_DIR, 'train.csv'))
df_test = pd.read_csv(os.path.join(MODELOS_DIR, 'test.csv'))

print(f"   Usuarios: {len(df_usuarios):,}")
print(f"   Recursos: {len(df_recursos):,}")
print(f"   Entrenamiento: {len(df_train):,}")
print(f"   Prueba: {len(df_test):,}")


# =============================================================================
# PASO 4.2: Construir el Dataset de LightFM con features
# =============================================================================
print("\n Construyendo dataset de LightFM con features...")

user_features_list = []
for carrera in CARRERAS.keys():
    user_features_list.append(f"carrera:{carrera}")
for sem in range(1, 10):
    user_features_list.append(f"semestre:{sem}")

item_features_list = []
for tema in TEMAS:
    item_features_list.append(f"tema:{tema}")
for tipo in TIPOS_RECURSO.keys():
    item_features_list.append(f"tipo:{tipo}")
for dif in DIFICULTADES:
    item_features_list.append(f"dificultad:{dif}")

print(f"   Features de usuario definidas: {len(user_features_list)}")
print(f"   Features de ítem definidas: {len(item_features_list)}")

dataset = Dataset()
dataset.fit(
    users=df_usuarios['user_id'].unique(),
    items=df_recursos['resource_id'].unique(),
    user_features=user_features_list,
    item_features=item_features_list,
)

n_users, n_items = dataset.interactions_shape()
print(f"   Matriz de interacciones: {n_users} usuarios × {n_items} recursos")


# =============================================================================
# PASO 4.3: Construir la matriz de interacciones (SOLO entrenamiento)
# =============================================================================
# NOTA: Solo construimos la matriz de entrenamiento para LightFM.
# La evaluación la hacemos manualmente sobre df_test para evitar el error
# de intersección train/test que LightFM detecta cuando un mismo par
# (usuario, recurso) aparece en ambos conjuntos.
# =============================================================================
print("\n Construyendo matriz de interacciones (entrenamiento)...")

train_interactions, train_weights = dataset.build_interactions(
    ((row['user_id'], row['resource_id'], row['rating'])
     for _, row in df_train.iterrows())
)

print(f"   Entrenamiento: {train_interactions.nnz:,} interacciones no nulas")
print(f"   Densidad: {train_interactions.nnz / (n_users * n_items) * 100:.1f}%")


# =============================================================================
# PASO 4.4: Construir matrices de features
# =============================================================================
print("\n  Construyendo matrices de features...")

user_features_data = []
for _, user in df_usuarios.iterrows():
    features = [
        f"carrera:{user['carrera']}",
        f"semestre:{user['semestre']}",
    ]
    user_features_data.append((user['user_id'], features))

user_features_matrix = dataset.build_user_features(user_features_data)

item_features_data = []
for _, recurso in df_recursos.iterrows():
    features = [
        f"tema:{recurso['tema']}",
        f"tipo:{recurso['tipo']}",
        f"dificultad:{recurso['dificultad']}",
    ]
    item_features_data.append((recurso['resource_id'], features))

item_features_matrix = dataset.build_item_features(item_features_data)

print(f"   Features de usuario: {user_features_matrix.shape}")
print(f"   Features de ítem: {item_features_matrix.shape}")


# =============================================================================
# PASO 4.5: Entrenar el modelo LightFM con evaluación manual
# =============================================================================
print(f"\n Entrenando modelo LightFM...")
print(f"   Componentes: {LIGHTFM_COMPONENTES}")
print(f"   Loss: {LIGHTFM_LOSS}")
print(f"   Épocas: {LIGHTFM_EPOCHS}")
print(f"   Learning rate: {LIGHTFM_LEARNING_RATE}")
print(f"   Regularización L2: {LIGHTFM_REGULARIZATION}")

modelo_cf = LightFM(
    no_components=LIGHTFM_COMPONENTES,
    loss=LIGHTFM_LOSS,
    learning_rate=LIGHTFM_LEARNING_RATE,
    item_alpha=LIGHTFM_REGULARIZATION,
    user_alpha=LIGHTFM_REGULARIZATION,
    random_state=SEED,
)

# Obtener mapeos internos de LightFM
user_id_map, _, item_id_map, _ = dataset.mapping()
idx_to_rid = {v: k for k, v in item_id_map.items()}


def evaluar_modelo(modelo, df_eval, df_train_ref, k=10, n_muestra=200):
    """
    Evalúa Precision@K y Recall@K MANUALMENTE sobre una muestra de usuarios.
    
    ¿Por qué manual y no con lightfm.evaluation?
    Porque build_interactions() de LightFM colapsa múltiples interacciones
    del mismo par (usuario, recurso) en una sola entrada. Si un usuario
    interactuó con un recurso en train Y en test (con diferente timestamp),
    LightFM lo cuenta como intersección y lanza un error. Nuestra evaluación
    manual evita este problema.
    """
    usuarios_eval = df_eval['user_id'].unique()
    if len(usuarios_eval) > n_muestra:
        np.random.seed(SEED)
        usuarios_eval = np.random.choice(usuarios_eval, n_muestra, replace=False)

    precisions = []
    recalls = []

    for uid in usuarios_eval:
        if uid not in user_id_map:
            continue

        uid_interno = user_id_map[uid]

        # Recursos relevantes en evaluación (rating >= umbral)
        relevantes = set(
            df_eval[(df_eval['user_id'] == uid) &
                    (df_eval['rating'] >= UMBRAL_RELEVANCIA)]['resource_id']
        )
        if len(relevantes) == 0:
            continue

        # Recursos ya consumidos en entrenamiento (excluir)
        consumidos = set(df_train_ref[df_train_ref['user_id'] == uid]['resource_id'])

        # Predecir scores para todos los ítems
        scores = modelo.predict(
            user_ids=uid_interno,
            item_ids=np.arange(n_items),
            user_features=user_features_matrix,
            item_features=item_features_matrix,
        )

        # Top K excluyendo consumidos
        ranking = np.argsort(-scores)
        recomendados = []
        for idx in ranking:
            rid = idx_to_rid[idx]
            if rid not in consumidos:
                recomendados.append(rid)
            if len(recomendados) >= k:
                break

        recomendados_set = set(recomendados)
        hits = len(recomendados_set & relevantes)
        precisions.append(hits / k)
        recalls.append(hits / len(relevantes))

    return np.mean(precisions) if precisions else 0, np.mean(recalls) if recalls else 0


# Entrenar por bloques de 5 épocas y evaluar
historico_precision = []
historico_recall = []
epocas_evaluadas = []

print(f"\n   {'Época':>5s} | {'P@10 Test':>10s} | {'R@10 Test':>10s} | {'Tiempo':>8s}")
print(f"   {'-'*5} | {'-'*10} | {'-'*10} | {'-'*8}")

inicio_total = time.time()

for bloque in range(LIGHTFM_EPOCHS // 5):
    epoca_actual = (bloque + 1) * 5

    inicio = time.time()
    modelo_cf.fit_partial(
        train_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        sample_weight=train_weights,
        num_threads=4,
        epochs=5,
    )

    # Evaluar sobre muestra del conjunto de prueba
    p_test, r_test = evaluar_modelo(modelo_cf, df_test, df_train, k=10, n_muestra=200)

    tiempo_bloque = time.time() - inicio

    historico_precision.append(p_test)
    historico_recall.append(r_test)
    epocas_evaluadas.append(epoca_actual)

    print(f"   {epoca_actual:5d} | {p_test:10.4f} | {r_test:10.4f} | {tiempo_bloque:7.1f}s")

tiempo_total = time.time() - inicio_total
print(f"\n    Modelo entrenado en {tiempo_total:.1f} segundos")


# =============================================================================
# PASO 4.6: Demostración de recomendaciones
# =============================================================================
print("\n Demostración de recomendaciones colaborativas:")

def recomendar_colaborativo(user_id, k=10):
    """Genera K recomendaciones usando el modelo LightFM."""
    uid_interno = user_id_map[user_id]

    scores = modelo_cf.predict(
        user_ids=uid_interno,
        item_ids=np.arange(n_items),
        user_features=user_features_matrix,
        item_features=item_features_matrix,
    )

    consumidos = set(df_train[df_train['user_id'] == user_id]['resource_id'])

    ranking = np.argsort(-scores)
    recomendaciones = []
    for idx in ranking:
        rid = idx_to_rid[idx]
        if rid not in consumidos:
            recomendaciones.append((rid, float(scores[idx])))
        if len(recomendaciones) >= k:
            break

    return recomendaciones


usuario_demo = "U0001"
info_usuario = df_usuarios[df_usuarios['user_id'] == usuario_demo].iloc[0]
print(f"\n   Usuario: {usuario_demo}")
print(f"   Carrera: {info_usuario['carrera']}")
print(f"   Intereses: {info_usuario['intereses']}")

recs_cf = recomendar_colaborativo(usuario_demo, k=10)
print(f"\n   Top 10 recomendaciones (filtrado colaborativo):")
for rank, (rid, score) in enumerate(recs_cf, 1):
    recurso = df_recursos[df_recursos['resource_id'] == rid].iloc[0]
    print(f"   {rank:2d}. {rid} | {recurso['tema']:<35s} | score: {score:.4f}")


# =============================================================================
# PASO 4.7: Guardar modelo y artefactos
# =============================================================================
print("\n Guardando modelo y artefactos...")

with open(os.path.join(MODELOS_DIR, 'lightfm_model.pkl'), 'wb') as f:
    pickle.dump(modelo_cf, f)

artefactos_cf = {
    'dataset': dataset,
    'train_interactions': train_interactions,
    'train_weights': train_weights,
    'user_features_matrix': user_features_matrix,
    'item_features_matrix': item_features_matrix,
    'user_id_map': user_id_map,
    'item_id_map': item_id_map,
    'idx_to_rid': idx_to_rid,
    'historico_precision': historico_precision,
    'historico_recall': historico_recall,
    'epocas_evaluadas': epocas_evaluadas,
    'n_items': n_items,
}
with open(os.path.join(MODELOS_DIR, 'lightfm_artefactos.pkl'), 'wb') as f:
    pickle.dump(artefactos_cf, f)

print(f"    modelos/lightfm_model.pkl")
print(f"    modelos/lightfm_artefactos.pkl")


# =============================================================================
# PASO 4.8: Visualizaciones
# =============================================================================
print("\n Generando visualizaciones...")

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

plt.style.use(ESTILO_MATPLOTLIB)

# --- Gráfica 13: Curva de entrenamiento ---
fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_GRANDE)

axes[0].plot(epocas_evaluadas, historico_precision, 'o-', color='#FF9800', linewidth=2, markersize=8)
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Precision@10')
axes[0].set_title('Precision@10 en Conjunto de Prueba')
axes[0].grid(True, alpha=0.3)

axes[1].plot(epocas_evaluadas, historico_recall, 's-', color='#4CAF50', linewidth=2, markersize=8)
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Recall@10')
axes[1].set_title('Recall@10 en Conjunto de Prueba')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Curva de Aprendizaje — Modelo LightFM (WARP, k=64)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '13_curva_entrenamiento_lightfm.png'), dpi=DPI)
plt.close()
print("    13_curva_entrenamiento_lightfm.png")


# --- Gráfica 14: Proyección t-SNE de embeddings de ítems de LightFM ---
item_embeddings_cf = modelo_cf.item_embeddings[:n_items]
print(f"   Embeddings de ítems LightFM: {item_embeddings_cf.shape}")
print("   Calculando t-SNE...")

tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
embeddings_cf_2d = tsne.fit_transform(item_embeddings_cf)

temas_por_idx = []
for i in range(len(item_id_map)):
    rid = idx_to_rid[i]
    tema = df_recursos[df_recursos['resource_id'] == rid]['tema'].values[0]
    temas_por_idx.append(tema)

temas_unicos = sorted(set(temas_por_idx))
colores_temas = plt.cm.tab20(np.linspace(0, 1, len(temas_unicos)))
tema_to_color = {tema: colores_temas[i] for i, tema in enumerate(temas_unicos)}

fig, ax = plt.subplots(figsize=(14, 10))
for tema in temas_unicos:
    mask = np.array([t == tema for t in temas_por_idx])
    ax.scatter(
        embeddings_cf_2d[mask, 0], embeddings_cf_2d[mask, 1],
        c=[tema_to_color[tema]], label=tema, alpha=0.7, s=50,
        edgecolors='white', linewidth=0.5
    )

ax.set_title('Proyección t-SNE de Factores Latentes LightFM por Tema')
ax.set_xlabel('t-SNE Dimensión 1')
ax.set_ylabel('t-SNE Dimensión 2')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '14_embeddings_lightfm_tsne.png'), dpi=DPI, bbox_inches='tight')
plt.close()
print("    14_embeddings_lightfm_tsne.png")


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print(f"\n{'=' * 70}")
print("PASO 4 COMPLETADO: Modelo de filtrado colaborativo")
print(f"{'=' * 70}")
print(f"  Modelo: LightFM ({LIGHTFM_COMPONENTES} componentes, loss={LIGHTFM_LOSS})")
print(f"  Épocas: {LIGHTFM_EPOCHS}")
print(f"  Tiempo total: {tiempo_total:.1f} segundos")
print(f"  Métricas finales (época {LIGHTFM_EPOCHS}):")
print(f"    Precision@10: {historico_precision[-1]:.4f}")
print(f"    Recall@10:    {historico_recall[-1]:.4f}")
print(f"  Archivos generados:")
print(f"    - modelos/lightfm_model.pkl")
print(f"    - modelos/lightfm_artefactos.pkl")
print(f"    - graficas/13_curva_entrenamiento_lightfm.png")
print(f"    - graficas/14_embeddings_lightfm_tsne.png")
print(f"{'=' * 70}")