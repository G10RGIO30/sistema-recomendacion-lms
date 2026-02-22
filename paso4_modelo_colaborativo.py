"""
=============================================================================
paso4_modelo_colaborativo.py — Modelo de filtrado colaborativo (LightFM)
=============================================================================

PROPÓSITO:
    Implementar el segundo "cerebro" del sistema de recomendación: el modelo
    de filtrado colaborativo. A diferencia del modelo de contenido (que lee
    descripciones), este modelo observa PATRONES DE COMPORTAMIENTO:
    
    "Estudiantes similares a ti también encontraron útiles estos recursos"
    
    LightFM es un modelo de factorización matricial que descompone la matriz
    de interacciones (1000 usuarios × 500 recursos) en factores latentes.
    Cada usuario y recurso queda representado por un vector de k dimensiones
    (k=64) que captura características ocultas.

¿POR QUÉ LIGHTFM Y NO OTRO?
    LightFM (Kula, 2015) tiene una ventaja clave: puede incorporar
    METADATOS (features) de usuarios e ítems. Esto significa que:
    - Un usuario nuevo con carrera="Ciencia de Datos" y semestre=5
      ya tiene una representación, incluso sin historial de interacciones.
    - Un recurso nuevo con tema="Machine Learning" y dificultad="avanzado"
      ya puede ser recomendado.
    Esto es fundamental para mitigar el problema de cold-start.

¿QUÉ ES WARP?
    Usamos la función de pérdida WARP (Weighted Approximate-Rank Pairwise).
    WARP optimiza directamente la posición de los ítems relevantes en el
    ranking, lo cual es exactamente lo que miden nuestras métricas
    (Precision@K, NDCG@K). BPR es la alternativa, pero WARP tiende a
    superar a BPR en tareas de ranking top-k (Weston et al., 2011).

SALIDAS:
    - modelos/lightfm_model.pkl        (modelo entrenado)
    - modelos/lightfm_matrices.pkl     (matrices de interacción y features)
    - graficas/13_curva_entrenamiento_lightfm.png
    - graficas/14_embeddings_lightfm_tsne.png

EJECUCIÓN:
    python paso4_modelo_colaborativo.py
    (requiere haber ejecutado paso1 y paso3 primero — usa train.csv y test.csv)
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

from scipy.sparse import coo_matrix, csr_matrix
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k

from config import (
    fijar_semillas, SEED,
    RUTA_USUARIOS, RUTA_RECURSOS,
    MODELOS_DIR, GRAFICAS_DIR,
    LIGHTFM_COMPONENTES, LIGHTFM_LOSS, LIGHTFM_EPOCHS,
    LIGHTFM_LEARNING_RATE, LIGHTFM_REGULARIZATION,
    CARRERAS, TEMAS, DIFICULTADES, TIPOS_RECURSO,
    ESTILO_MATPLOTLIB, FIGSIZE_NORMAL, FIGSIZE_GRANDE, DPI
)

fijar_semillas(SEED)

print("=" * 70)
print("PASO 4: Modelo de filtrado colaborativo (LightFM)")
print("=" * 70)

# =============================================================================
# PASO 4.1: Cargar datos
# =============================================================================
print("\n📂 Cargando datos...")
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
# LightFM requiere un objeto Dataset que mapea usuarios, ítems y features.
#
# Features de usuario: carrera y semestre
#   → Permite al modelo saber que "los de Ciencia de Datos prefieren ML"
#     incluso para usuarios nuevos.
#
# Features de ítem: tema, tipo y dificultad
#   → Permite al modelo saber que "un recurso de ML avanzado es similar a
#     otro de IA intermedio" incluso sin interacciones previas.
# =============================================================================
print("\n Construyendo dataset de LightFM con features...")

# Definir todas las features posibles
# Para usuario: "carrera:Ciencia de Datos", "semestre:5", etc.
user_features_list = []
for carrera in CARRERAS.keys():
    user_features_list.append(f"carrera:{carrera}")
for sem in range(1, 10):
    user_features_list.append(f"semestre:{sem}")

# Para ítem: "tema:Machine Learning", "tipo:video", "dificultad:avanzado", etc.
item_features_list = []
for tema in TEMAS:
    item_features_list.append(f"tema:{tema}")
for tipo in TIPOS_RECURSO.keys():
    item_features_list.append(f"tipo:{tipo}")
for dif in DIFICULTADES:
    item_features_list.append(f"dificultad:{dif}")

print(f"   Features de usuario definidas: {len(user_features_list)}")
print(f"   Features de ítem definidas: {len(item_features_list)}")

# Crear el dataset
dataset = Dataset()
dataset.fit(
    users=df_usuarios['user_id'].unique(),
    items=df_recursos['resource_id'].unique(),
    user_features=user_features_list,
    item_features=item_features_list,
)

# Verificar mapeos
n_users, n_items = dataset.interactions_shape()
print(f"   Matriz de interacciones: {n_users} usuarios × {n_items} recursos")


# =============================================================================
# PASO 4.3: Construir la matriz de interacciones
# =============================================================================
# LightFM trabaja con matrices dispersas (sparse). La mayoría de las celdas
# de la matriz usuario×recurso están vacías (90% en nuestro caso).
# Las matrices dispersas solo almacenan los valores no nulos, ahorrando
# memoria y acelerando los cálculos.
#
# Usamos los ratings como pesos de las interacciones. Un rating de 5 cuenta
# más que uno de 1 durante el entrenamiento.
# =============================================================================
print("\n Construyendo matrices de interacciones...")

# Matriz de entrenamiento (con ratings como pesos)
train_interactions, train_weights = dataset.build_interactions(
    ((row['user_id'], row['resource_id'], row['rating'])
     for _, row in df_train.iterrows())
)

# Matriz de prueba
test_interactions, test_weights = dataset.build_interactions(
    ((row['user_id'], row['resource_id'], row['rating'])
     for _, row in df_test.iterrows())
)

print(f"   Entrenamiento: {train_interactions.nnz:,} interacciones no nulas")
print(f"   Prueba: {test_interactions.nnz:,} interacciones no nulas")
print(f"   Densidad entrenamiento: {train_interactions.nnz / (n_users * n_items) * 100:.1f}%")


# =============================================================================
# PASO 4.4: Construir matrices de features
# =============================================================================
# Aquí asignamos a cada usuario sus features (carrera, semestre) y a cada
# recurso sus features (tema, tipo, dificultad).
#
# LightFM representa cada usuario como la SUMA de los embeddings de sus
# features. Esto es lo que permite el cold-start: un usuario nuevo sin
# historial se representa por sus features demográficas.
# =============================================================================
print("\n  Construyendo matrices de features...")

# Features de usuario: para cada usuario, su carrera y semestre
user_features_data = []
for _, user in df_usuarios.iterrows():
    features = [
        f"carrera:{user['carrera']}",
        f"semestre:{user['semestre']}",
    ]
    user_features_data.append((user['user_id'], features))

user_features_matrix = dataset.build_user_features(user_features_data)

# Features de ítem: para cada recurso, su tema, tipo y dificultad
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
# PASO 4.5: Entrenar el modelo LightFM
# =============================================================================
# Entrenamos con WARP loss durante 30 épocas, registrando las métricas
# en cada época para generar la curva de aprendizaje.
#
# Hiperparámetros (definidos en config.py):
#   - no_components=64: dimensionalidad de los factores latentes
#   - loss='warp': optimiza ranking directamente
#   - learning_rate=0.05: tasa de aprendizaje
#   - item/user_alpha=1e-5: regularización L2 (previene overfitting)
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

# Entrenar época por época y registrar métricas
historico_train_precision = []
historico_test_precision = []
historico_train_recall = []
historico_test_recall = []

print(f"\n   {'Época':>5s} | {'P@10 Train':>10s} | {'P@10 Test':>10s} | {'R@10 Train':>10s} | {'R@10 Test':>10s}")
print(f"   {'-'*5} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

for epoch in range(1, LIGHTFM_EPOCHS + 1):
    # Entrenar una época
    modelo_cf.fit_partial(
        train_interactions,
        user_features=user_features_matrix,
        item_features=item_features_matrix,
        sample_weight=train_weights,
        num_threads=4,
        epochs=1,
    )
    
    # Evaluar cada 5 épocas (y en la última)
    if epoch % 5 == 0 or epoch == 1 or epoch == LIGHTFM_EPOCHS:
        p_train = precision_at_k(
            modelo_cf, train_interactions, k=10,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            num_threads=4
        ).mean()
        
        p_test = precision_at_k(
            modelo_cf, test_interactions,
            train_interactions=train_interactions, k=10,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            num_threads=4
        ).mean()
        
        r_train = recall_at_k(
            modelo_cf, train_interactions, k=10,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            num_threads=4
        ).mean()
        
        r_test = recall_at_k(
            modelo_cf, test_interactions,
            train_interactions=train_interactions, k=10,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            num_threads=4
        ).mean()
        
        historico_train_precision.append((epoch, p_train))
        historico_test_precision.append((epoch, p_test))
        historico_train_recall.append((epoch, r_train))
        historico_test_recall.append((epoch, r_test))
        
        print(f"   {epoch:5d} | {p_train:10.4f} | {p_test:10.4f} | {r_train:10.4f} | {r_test:10.4f}")

print(f"\n    Modelo entrenado exitosamente")


# =============================================================================
# PASO 4.6: Demostración de recomendaciones
# =============================================================================
# Para generar recomendaciones con LightFM:
# 1. El modelo calcula una puntuación para cada par (usuario, recurso)
#    usando los factores latentes aprendidos + features
# 2. Se ordenan los recursos por puntuación descendente
# 3. Se excluyen los ya consumidos
# =============================================================================
print("\n Demostración de recomendaciones colaborativas:")

# Obtener mapeos internos de LightFM
user_id_map, _, item_id_map, _ = dataset.mapping()

# Función de recomendación colaborativa
def recomendar_colaborativo(user_id, k=10):
    """
    Genera K recomendaciones usando el modelo LightFM.
    
    Retorna lista de tuplas (resource_id, score).
    """
    # Obtener índice interno de LightFM para este usuario
    uid_interno = user_id_map[user_id]
    
    # Calcular puntuaciones para TODOS los recursos
    n_items_total = len(item_id_map)
    scores = modelo_cf.predict(
        user_ids=uid_interno,
        item_ids=np.arange(n_items_total),
        user_features=user_features_matrix,
        item_features=item_features_matrix,
    )
    
    # Crear mapeo inverso: índice interno -> resource_id
    idx_to_rid = {v: k for k, v in item_id_map.items()}
    
    # Recursos ya consumidos en entrenamiento (para excluirlos)
    consumidos = set(df_train[df_train['user_id'] == user_id]['resource_id'])
    
    # Ordenar por puntuación y filtrar
    ranking = np.argsort(-scores)  # Orden descendente
    recomendaciones = []
    for idx in ranking:
        rid = idx_to_rid[idx]
        if rid not in consumidos:
            recomendaciones.append((rid, float(scores[idx])))
        if len(recomendaciones) >= k:
            break
    
    return recomendaciones

# Demostración con el mismo usuario U0001
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

# Guardar modelo
with open(os.path.join(MODELOS_DIR, 'lightfm_model.pkl'), 'wb') as f:
    pickle.dump(modelo_cf, f)

# Guardar matrices y dataset (necesarios para pasos posteriores)
artefactos_cf = {
    'dataset': dataset,
    'train_interactions': train_interactions,
    'test_interactions': test_interactions,
    'train_weights': train_weights,
    'test_weights': test_weights,
    'user_features_matrix': user_features_matrix,
    'item_features_matrix': item_features_matrix,
    'user_id_map': user_id_map,
    'item_id_map': item_id_map,
    'historico_train_precision': historico_train_precision,
    'historico_test_precision': historico_test_precision,
    'historico_train_recall': historico_train_recall,
    'historico_test_recall': historico_test_recall,
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
import seaborn as sns
from sklearn.manifold import TSNE

plt.style.use(ESTILO_MATPLOTLIB)

# --- Gráfica 13: Curva de entrenamiento ---
# Muestra cómo las métricas evolucionan con las épocas.
# Si test sube junto con train → el modelo generaliza bien.
# Si test baja mientras train sube → overfitting.

fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_GRANDE)

# Precision@10
epochs_p = [e for e, _ in historico_train_precision]
train_p = [p for _, p in historico_train_precision]
test_p = [p for _, p in historico_test_precision]

axes[0].plot(epochs_p, train_p, 'o-', color='#2196F3', linewidth=2, label='Entrenamiento')
axes[0].plot(epochs_p, test_p, 's-', color='#FF9800', linewidth=2, label='Prueba')
axes[0].set_xlabel('Época')
axes[0].set_ylabel('Precision@10')
axes[0].set_title('Precision@10 durante Entrenamiento')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Recall@10
epochs_r = [e for e, _ in historico_train_recall]
train_r = [r for _, r in historico_train_recall]
test_r = [r for _, r in historico_test_recall]

axes[1].plot(epochs_r, train_r, 'o-', color='#2196F3', linewidth=2, label='Entrenamiento')
axes[1].plot(epochs_r, test_r, 's-', color='#FF9800', linewidth=2, label='Prueba')
axes[1].set_xlabel('Época')
axes[1].set_ylabel('Recall@10')
axes[1].set_title('Recall@10 durante Entrenamiento')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.suptitle('Curva de Aprendizaje del Modelo LightFM', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '13_curva_entrenamiento_lightfm.png'), dpi=DPI)
plt.close()
print("    13_curva_entrenamiento_lightfm.png")


# --- Gráfica 14: Proyección t-SNE de embeddings de ítems de LightFM ---
# LightFM también genera embeddings (los factores latentes).
# Podemos visualizarlos y comparar con los de SBERT.

# Obtener embeddings de ítems de LightFM
item_embeddings_cf = modelo_cf.item_embeddings  # Shape: (n_items, 64)

print(f"   Embeddings de ítems LightFM: {item_embeddings_cf.shape}")
print("   Calculando t-SNE...")

tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
embeddings_cf_2d = tsne.fit_transform(item_embeddings_cf)

# Mapear cada recurso a su tema (usando el orden del dataset de LightFM)
idx_to_rid = {v: k for k, v in item_id_map.items()}
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

# Métricas finales
final_p_train = historico_train_precision[-1][1]
final_p_test = historico_test_precision[-1][1]
final_r_train = historico_train_recall[-1][1]
final_r_test = historico_test_recall[-1][1]

print(f"\n{'=' * 70}")
print("PASO 4 COMPLETADO: Modelo de filtrado colaborativo")
print(f"{'=' * 70}")
print(f"  Modelo: LightFM ({LIGHTFM_COMPONENTES} componentes, loss={LIGHTFM_LOSS})")
print(f"  Épocas: {LIGHTFM_EPOCHS}")
print(f"  Métricas finales (época {LIGHTFM_EPOCHS}):")
print(f"    Precision@10 — Train: {final_p_train:.4f} | Test: {final_p_test:.4f}")
print(f"    Recall@10    — Train: {final_r_train:.4f} | Test: {final_r_test:.4f}")
print(f"  Archivos generados:")
print(f"    - modelos/lightfm_model.pkl")
print(f"    - modelos/lightfm_artefactos.pkl")
print(f"    - graficas/13_curva_entrenamiento_lightfm.png")
print(f"    - graficas/14_embeddings_lightfm_tsne.png")
print(f"{'=' * 70}")