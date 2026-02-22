"""
=============================================================================
paso5_modelo_hibrido.py — Modelo híbrido y optimización de α
=============================================================================

PROPÓSITO:
    Combinar los modelos SBERT basado en contenido y LightFM
    filtrado colaborativo en un modelo híbrido. La combinación se hace
    mediante una media ponderada de los scores:
    
        score_híbrido = α × score_contenido + (1 - α) × score_colaborativo
    
    Donde α ∈ [0, 1] controla el balance:
        α = 1.0 → 100% contenido, 0% colaborativo
        α = 0.5 → 50% cada uno
        α = 0.0 → 0% contenido, 100% colaborativo
    
    El valor óptimo de α se encuentra mediante validación cruzada de 5 folds
    sobre el conjunto de entrenamiento.

¿POR QUÉ NORMALIZAR LOS SCORES?
    Los scores de SBERT (similitud del coseno) están en [−1, 1] y los de
    LightFM (WARP) son valores negativos grandes (~−29). No se pueden
    sumar directamente. Se aplica normalización min-max a ambos para
    llevarlos al rango [0, 1] antes de combinarlos.

SALIDAS:
    - modelos/alpha_optimo.pkl
    - graficas/15_optimizacion_alpha.png
    - graficas/16_comparacion_3_modelos_demo.png

EJECUCIÓN:
    python paso5_modelo_hibrido.py
=============================================================================
"""

import pandas as pd
import numpy as np
import os
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

import faiss
from lightfm import LightFM

from config import (
    fijar_semillas, SEED,
    RUTA_USUARIOS, RUTA_RECURSOS,
    MODELOS_DIR, GRAFICAS_DIR,
    SBERT_DIMENSIONES, ALPHA_VALORES, N_FOLDS_CV,
    UMBRAL_RELEVANCIA, K_VALORES,
    ESTILO_MATPLOTLIB, FIGSIZE_NORMAL, FIGSIZE_GRANDE, DPI,
    COLORES_MODELOS
)

fijar_semillas(SEED)

print("=" * 70)
print("PASO 5: Modelo híbrido y optimización de α")
print("=" * 70)

# =============================================================================
# PASO 5.1: Cargar todos los artefactos necesarios
# =============================================================================
print("\n Cargando artefactos de los pasos anteriores...")

df_usuarios = pd.read_csv(RUTA_USUARIOS)
df_recursos = pd.read_csv(RUTA_RECURSOS)
df_train = pd.read_csv(os.path.join(MODELOS_DIR, 'train.csv'))
df_test = pd.read_csv(os.path.join(MODELOS_DIR, 'test.csv'))

# Embeddings SBERT y perfiles de usuario (Paso 3)
embeddings_recursos = np.load(os.path.join(MODELOS_DIR, 'embeddings_recursos.npy'))
perfiles_usuario = np.load(os.path.join(MODELOS_DIR, 'perfiles_usuario.npy'))
index_faiss = faiss.read_index(os.path.join(MODELOS_DIR, 'faiss_index.bin'))

with open(os.path.join(MODELOS_DIR, 'mapeos.pkl'), 'rb') as f:
    mapeos = pickle.load(f)
resource_to_idx = mapeos['resource_to_idx']
user_to_idx_cb = mapeos['user_to_idx']

# Modelo LightFM y artefactos (Paso 4)
with open(os.path.join(MODELOS_DIR, 'lightfm_model.pkl'), 'rb') as f:
    modelo_cf = pickle.load(f)

with open(os.path.join(MODELOS_DIR, 'lightfm_artefactos.pkl'), 'rb') as f:
    artefactos_cf = pickle.load(f)

user_id_map = artefactos_cf['user_id_map']
item_id_map = artefactos_cf['item_id_map']
idx_to_rid = artefactos_cf['idx_to_rid']
user_features_matrix = artefactos_cf['user_features_matrix']
item_features_matrix = artefactos_cf['item_features_matrix']
n_items = artefactos_cf['n_items']

print(f"    Embeddings SBERT: {embeddings_recursos.shape}")
print(f"    Perfiles usuario: {perfiles_usuario.shape}")
print(f"    Índice FAISS: {index_faiss.ntotal} vectores")
print(f"    Modelo LightFM: {modelo_cf.no_components} componentes")
print(f"    Entrenamiento: {len(df_train):,} | Prueba: {len(df_test):,}")


# =============================================================================
# PASO 5.2: Funciones de scoring para cada modelo
# =============================================================================
# Necesitamos funciones que devuelvan un score para CADA recurso dado un
# usuario, no solo el top K. Esto permite combinar los scores.
# =============================================================================

def scores_contenido(user_id):
    """
    Calcula score basado en contenido para TODOS los recursos.
    Score = similitud del coseno entre el perfil del usuario y cada recurso.
    Retorna array de shape (n_recursos,) con scores en [−1, 1].
    """
    idx_u = user_to_idx_cb[user_id]
    perfil = perfiles_usuario[idx_u:idx_u+1]  # (1, 384)
    
    # Producto interno con todos los embeddings (= coseno si normalizados)
    scores = np.dot(embeddings_recursos, perfil.T).flatten()  # (500,)
    return scores


def scores_colaborativo(user_id):
    """
    Calcula score colaborativo para TODOS los recursos.
    Retorna array de shape (n_recursos,) con scores de LightFM.
    """
    uid_interno = user_id_map[user_id]
    
    scores_raw = modelo_cf.predict(
        user_ids=uid_interno,
        item_ids=np.arange(n_items),
        user_features=user_features_matrix,
        item_features=item_features_matrix,
    )
    
    # Reordenar para que el índice corresponda a df_recursos
    # LightFM usa su propio mapeo de ítems, hay que alinearlo
    scores = np.zeros(len(df_recursos))
    for lightfm_idx in range(n_items):
        rid = idx_to_rid[lightfm_idx]
        if rid in resource_to_idx:
            recurso_idx = resource_to_idx[rid]
            scores[recurso_idx] = scores_raw[lightfm_idx]
    
    return scores


def normalizar_minmax(scores):
    """
    Normalización min-max: lleva los scores al rango [0, 1].
    Fórmula: (x - min) / (max - min)
    """
    min_s = scores.min()
    max_s = scores.max()
    if max_s - min_s == 0:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)


def scores_hibrido(user_id, alpha):
    """
    Calcula scores híbridos para todos los recursos.
    
    score_híbrido = α × CB_normalizado + (1 - α) × CF_normalizado
    """
    s_cb = normalizar_minmax(scores_contenido(user_id))
    s_cf = normalizar_minmax(scores_colaborativo(user_id))
    return alpha * s_cb + (1 - alpha) * s_cf


def recomendar_hibrido(user_id, alpha, k=10):
    """
    Genera top K recomendaciones híbridas, excluyendo recursos consumidos.
    """
    scores = scores_hibrido(user_id, alpha)
    consumidos = set(df_train[df_train['user_id'] == user_id]['resource_id'])
    
    ranking = np.argsort(-scores)
    recomendaciones = []
    for idx in ranking:
        rid = df_recursos.iloc[idx]['resource_id']
        if rid not in consumidos:
            recomendaciones.append((rid, float(scores[idx])))
        if len(recomendaciones) >= k:
            break
    
    return recomendaciones


# =============================================================================
# PASO 5.3: Optimización de α mediante validación cruzada
# =============================================================================
# Probamos α ∈ {0.0, 0.1, 0.2, ..., 1.0} y evaluamos cada valor con
# Precision@10 sobre una muestra del conjunto de entrenamiento usando
# validación cruzada de 5 folds.
#
# ¿Por qué sobre entrenamiento y no sobre test?
# Porque el test se reserva para la evaluación final. Si usáramos test
# para optimizar α, estaríamos "haciendo trampa" (data leakage).
# =============================================================================
print(f"\n Optimizando α mediante validación cruzada ({N_FOLDS_CV} folds)...")
print(f"   Valores de α a probar: {ALPHA_VALORES}")

# Preparar datos para CV: usar una muestra de usuarios del entrenamiento
usuarios_train = df_train['user_id'].unique()
np.random.seed(SEED)
usuarios_cv = np.random.choice(usuarios_train, size=min(300, len(usuarios_train)), replace=False)
np.random.shuffle(usuarios_cv)

# Dividir en folds
folds = np.array_split(usuarios_cv, N_FOLDS_CV)

resultados_alpha = {}

print(f"\n   {'α':>4s} | {'P@10 (CV)':>10s} | {'Std':>8s}")
print(f"   {'-'*4} | {'-'*10} | {'-'*8}")

inicio_total = time.time()

for alpha in ALPHA_VALORES:
    scores_folds = []
    
    for fold_idx in range(N_FOLDS_CV):
        # Usuarios de validación en este fold
        usuarios_val = folds[fold_idx]
        
        precisions = []
        for uid in usuarios_val:
            # Recursos relevantes: los que el usuario calificó ≥ 4 en entrenamiento
            # (simulamos evaluación sobre datos conocidos)
            inter_usuario = df_train[df_train['user_id'] == uid]
            relevantes = set(
                inter_usuario[inter_usuario['rating'] >= UMBRAL_RELEVANCIA]['resource_id']
            )
            if len(relevantes) == 0:
                continue
            
            # Generar recomendaciones híbridas
            # Nota: no excluimos consumidos aquí porque estamos evaluando
            # la capacidad del modelo de re-rankear los propios ítems del usuario
            scores = scores_hibrido(uid, alpha)
            top_k_indices = np.argsort(-scores)[:10]
            top_k_rids = set(df_recursos.iloc[top_k_indices]['resource_id'])
            
            hits = len(top_k_rids & relevantes)
            precisions.append(hits / 10)
        
        if precisions:
            scores_folds.append(np.mean(precisions))
    
    mean_score = np.mean(scores_folds)
    std_score = np.std(scores_folds)
    resultados_alpha[alpha] = (mean_score, std_score)
    
    print(f"   {alpha:4.1f} | {mean_score:10.4f} | {std_score:8.4f}")

tiempo_cv = time.time() - inicio_total
print(f"\n   Tiempo de validación cruzada: {tiempo_cv:.1f} segundos")

# Encontrar el mejor α
mejor_alpha = max(resultados_alpha.keys(), key=lambda a: resultados_alpha[a][0])
mejor_score = resultados_alpha[mejor_alpha][0]

print(f"\n    Mejor α = {mejor_alpha} con P@10 = {mejor_score:.4f}")

# Guardar resultado
with open(os.path.join(MODELOS_DIR, 'alpha_optimo.pkl'), 'wb') as f:
    pickle.dump({
        'alpha_optimo': mejor_alpha,
        'resultados_alpha': resultados_alpha,
    }, f)
print(f"    Guardado en modelos/alpha_optimo.pkl")


# =============================================================================
# PASO 5.4: Demostración del modelo híbrido con α óptimo
# =============================================================================
print(f"\n Demostración del modelo híbrido (α = {mejor_alpha}):")

usuario_demo = "U0001"
info_usuario = df_usuarios[df_usuarios['user_id'] == usuario_demo].iloc[0]
print(f"\n   Usuario: {usuario_demo}")
print(f"   Carrera: {info_usuario['carrera']}")
print(f"   Intereses: {info_usuario['intereses']}")

# Recomendaciones de los 3 modelos para comparar
recs_cb = []
s_cb = scores_contenido(usuario_demo)
consumidos = set(df_train[df_train['user_id'] == usuario_demo]['resource_id'])
for idx in np.argsort(-s_cb):
    rid = df_recursos.iloc[idx]['resource_id']
    if rid not in consumidos:
        recs_cb.append((rid, float(s_cb[idx])))
    if len(recs_cb) >= 10:
        break

recs_cf = []
s_cf = scores_colaborativo(usuario_demo)
for idx in np.argsort(-s_cf):
    rid = df_recursos.iloc[idx]['resource_id']
    if rid not in consumidos:
        recs_cf.append((rid, float(s_cf[idx])))
    if len(recs_cf) >= 10:
        break

recs_hybrid = recomendar_hibrido(usuario_demo, mejor_alpha, k=10)

print(f"\n   {'Pos':>3s} | {'Contenido (SBERT)':^30s} | {'Colaborativo (LightFM)':^30s} | {'Híbrido (α={:.1f})':^30s}".format(mejor_alpha))
print(f"   {'-'*3} | {'-'*30} | {'-'*30} | {'-'*30}")

for i in range(10):
    rid_cb = recs_cb[i][0] if i < len(recs_cb) else "—"
    tema_cb = df_recursos[df_recursos['resource_id'] == rid_cb]['tema'].values[0] if rid_cb != "—" else "—"
    
    rid_cf = recs_cf[i][0] if i < len(recs_cf) else "—"
    tema_cf = df_recursos[df_recursos['resource_id'] == rid_cf]['tema'].values[0] if rid_cf != "—" else "—"
    
    rid_h = recs_hybrid[i][0] if i < len(recs_hybrid) else "—"
    tema_h = df_recursos[df_recursos['resource_id'] == rid_h]['tema'].values[0] if rid_h != "—" else "—"
    
    print(f"   {i+1:3d} | {tema_cb:<30s} | {tema_cf:<30s} | {tema_h:<30s}")

# Contar temas únicos en cada lista
temas_cb = set(df_recursos[df_recursos['resource_id'].isin([r[0] for r in recs_cb])]['tema'])
temas_cf = set(df_recursos[df_recursos['resource_id'].isin([r[0] for r in recs_cf])]['tema'])
temas_h = set(df_recursos[df_recursos['resource_id'].isin([r[0] for r in recs_hybrid])]['tema'])

print(f"\n   Diversidad temática:")
print(f"   Contenido:     {len(temas_cb)} temas — {temas_cb}")
print(f"   Colaborativo:  {len(temas_cf)} temas — {temas_cf}")
print(f"   Híbrido:       {len(temas_h)} temas — {temas_h}")


# =============================================================================
# PASO 5.5: Visualizaciones
# =============================================================================
print("\n Generando visualizaciones...")

import matplotlib.pyplot as plt
plt.style.use(ESTILO_MATPLOTLIB)

# --- Gráfica 15: Optimización de α ---
fig, ax = plt.subplots(figsize=FIGSIZE_NORMAL)

alphas = sorted(resultados_alpha.keys())
means = [resultados_alpha[a][0] for a in alphas]
stds = [resultados_alpha[a][1] for a in alphas]

ax.errorbar(alphas, means, yerr=stds, fmt='o-', color='#4CAF50',
            linewidth=2, markersize=8, capsize=5, capthick=2,
            ecolor='#81C784', label='P@10 ± σ (CV 5-fold)')

# Marcar el mejor α
ax.axvline(mejor_alpha, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'α óptimo = {mejor_alpha}')
ax.scatter([mejor_alpha], [mejor_score], color='red', s=150, zorder=5, marker='*')

ax.set_xlabel('α (peso del componente basado en contenido)')
ax.set_ylabel('Precision@10')
ax.set_title('Optimización del Parámetro α mediante Validación Cruzada')
ax.set_xticks(ALPHA_VALORES)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Anotaciones en los extremos
ax.annotate('100% Colaborativo', xy=(0.0, means[0]), xytext=(0.05, means[0] - 0.005),
            fontsize=9, color='gray')
ax.annotate('100% Contenido', xy=(1.0, means[-1]), xytext=(0.75, means[-1] + 0.005),
            fontsize=9, color='gray')

plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '15_optimizacion_alpha.png'), dpi=DPI)
plt.close()
print("    15_optimizacion_alpha.png")


# --- Gráfica 16: Comparación visual de recomendaciones de los 3 modelos ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

modelos_data = [
    ("Basado en Contenido\n(SBERT)", recs_cb, COLORES_MODELOS['Basado en Contenido']),
    ("Filtrado Colaborativo\n(LightFM)", recs_cf, COLORES_MODELOS['Filtrado Colaborativo']),
    (f"Híbrido\n(α = {mejor_alpha})", recs_hybrid, COLORES_MODELOS['Híbrido']),
]

for ax, (titulo, recs, color) in zip(axes, modelos_data):
    temas_recs = [df_recursos[df_recursos['resource_id'] == r[0]]['tema'].values[0] for r in recs]
    tema_counts = pd.Series(temas_recs).value_counts()
    
    ax.barh(range(len(tema_counts)), tema_counts.values, color=color, alpha=0.8, edgecolor='white')
    ax.set_yticks(range(len(tema_counts)))
    ax.set_yticklabels(tema_counts.index, fontsize=9)
    ax.set_xlabel('Recursos recomendados')
    ax.set_title(titulo, fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    for i, v in enumerate(tema_counts.values):
        ax.text(v + 0.1, i, str(v), va='center', fontweight='bold')

plt.suptitle(f'Distribución Temática de Top 10 Recomendaciones para U0001',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '16_comparacion_3_modelos_demo.png'), dpi=DPI)
plt.close()
print("    16_comparacion_3_modelos_demo.png")


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print(f"\n{'=' * 70}")
print("PASO 5 COMPLETADO: Modelo híbrido")
print(f"{'=' * 70}")
print(f"  α óptimo: {mejor_alpha}")
print(f"  Mejor P@10 (CV): {mejor_score:.4f}")
print(f"  Fórmula: score = {mejor_alpha} × CB + {1-mejor_alpha:.1f} × CF")
print(f"  Tiempo CV: {tiempo_cv:.1f} segundos")
print(f"  Archivos generados:")
print(f"    - modelos/alpha_optimo.pkl")
print(f"    - graficas/15_optimizacion_alpha.png")
print(f"    - graficas/16_comparacion_3_modelos_demo.png")
print(f"{'=' * 70}")
