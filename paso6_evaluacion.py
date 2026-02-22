"""
=============================================================================
paso6_evaluacion.py — Evaluación comparativa y contraste de hipótesis
=============================================================================

PROPÓSITO:
    Evaluar formalmente los tres modelos (contenido, colaborativo, híbrido)
    en dos escenarios:
    
    Escenario 1 — General: todos los usuarios del conjunto de prueba
    Escenario 2 — Cold-start: usuarios con pocas interacciones en entrenamiento
    
    Métricas: Precision@K, Recall@K, NDCG@K para K ∈ {5, 10, 20}
    
    Pruebas estadísticas: Wilcoxon signed-rank test (α_sig = 0.05)
    para contrastar las tres hipótesis:
    
    H₁: El modelo híbrido supera al mejor individual en ≥15%
    H₂: SBERT supera a un baseline TF-IDF en CB
    H₃: El híbrido supera al CF puro en cold-start

SALIDAS:
    - graficas/17_comparacion_metricas_general.png
    - graficas/18_comparacion_metricas_coldstart.png
    - graficas/19_mejora_porcentual_hibrido.png
    - graficas/20_distribucion_ndcg_por_modelo.png
    - graficas/21_hipotesis_resumen.png

EJECUCIÓN:
    python paso6_evaluacion.py
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
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    fijar_semillas, SEED,
    RUTA_USUARIOS, RUTA_RECURSOS,
    MODELOS_DIR, GRAFICAS_DIR,
    SBERT_DIMENSIONES, K_VALORES, UMBRAL_RELEVANCIA,
    UMBRAL_COLD_START, ALPHA_SIGNIFICANCIA,
    ESTILO_MATPLOTLIB, FIGSIZE_NORMAL, FIGSIZE_GRANDE, DPI,
    COLORES_MODELOS
)

fijar_semillas(SEED)

print("=" * 70)
print("PASO 6: Evaluación comparativa y contraste de hipótesis")
print("=" * 70)

# =============================================================================
# PASO 6.1: Cargar artefactos
# =============================================================================
print("\n Cargando artefactos...")

df_usuarios = pd.read_csv(RUTA_USUARIOS)
df_recursos = pd.read_csv(RUTA_RECURSOS)
df_train = pd.read_csv(os.path.join(MODELOS_DIR, 'train.csv'))
df_test = pd.read_csv(os.path.join(MODELOS_DIR, 'test.csv'))

# SBERT
embeddings_recursos = np.load(os.path.join(MODELOS_DIR, 'embeddings_recursos.npy'))
perfiles_usuario = np.load(os.path.join(MODELOS_DIR, 'perfiles_usuario.npy'))

with open(os.path.join(MODELOS_DIR, 'mapeos.pkl'), 'rb') as f:
    mapeos = pickle.load(f)
resource_to_idx = mapeos['resource_to_idx']
user_to_idx_cb = mapeos['user_to_idx']

# LightFM
with open(os.path.join(MODELOS_DIR, 'lightfm_model.pkl'), 'rb') as f:
    modelo_cf = pickle.load(f)

with open(os.path.join(MODELOS_DIR, 'lightfm_artefactos.pkl'), 'rb') as f:
    art = pickle.load(f)

user_id_map = art['user_id_map']
item_id_map = art['item_id_map']
idx_to_rid = art['idx_to_rid']
user_features_matrix = art['user_features_matrix']
item_features_matrix = art['item_features_matrix']
n_items = art['n_items']

# Alpha óptimo
with open(os.path.join(MODELOS_DIR, 'alpha_optimo.pkl'), 'rb') as f:
    alpha_data = pickle.load(f)
ALPHA_OPTIMO = alpha_data['alpha_optimo']

print(f"    Todos los artefactos cargados")
print(f"   α óptimo: {ALPHA_OPTIMO}")


# =============================================================================
# PASO 6.2: Funciones de scoring (mismas del paso 5 + TF-IDF baseline)
# =============================================================================

def scores_contenido_sbert(user_id):
    idx_u = user_to_idx_cb[user_id]
    perfil = perfiles_usuario[idx_u:idx_u+1]
    return np.dot(embeddings_recursos, perfil.T).flatten()

def scores_colaborativo(user_id):
    uid_interno = user_id_map[user_id]
    scores_raw = modelo_cf.predict(
        user_ids=uid_interno,
        item_ids=np.arange(n_items),
        user_features=user_features_matrix,
        item_features=item_features_matrix,
    )
    scores = np.zeros(len(df_recursos))
    for lightfm_idx in range(n_items):
        rid = idx_to_rid[lightfm_idx]
        if rid in resource_to_idx:
            scores[resource_to_idx[rid]] = scores_raw[lightfm_idx]
    return scores

def normalizar_minmax(scores):
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s == 0:
        return np.zeros_like(scores)
    return (scores - min_s) / (max_s - min_s)

def scores_hibrido(user_id, alpha=ALPHA_OPTIMO):
    s_cb = normalizar_minmax(scores_contenido_sbert(user_id))
    s_cf = normalizar_minmax(scores_colaborativo(user_id))
    return alpha * s_cb + (1 - alpha) * s_cf


# --- Baseline TF-IDF (para H₂) ---
print("\n Construyendo baseline TF-IDF para contraste de H₂...")

textos_recursos = []
for _, r in df_recursos.iterrows():
    textos_recursos.append(f"{r['titulo']}. {r['descripcion']}. Tema: {r['tema']}")

vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
tfidf_matrix = vectorizer.fit_transform(textos_recursos)  # (500, features)

# Construir perfiles TF-IDF de usuario (promedio ponderado, igual que SBERT)
tfidf_dense = tfidf_matrix.toarray()  # (500, n_features)
perfiles_tfidf = np.zeros((len(df_usuarios), tfidf_dense.shape[1]))

for uid in df_usuarios['user_id']:
    idx_u = user_to_idx_cb[uid]
    inter = df_train[df_train['user_id'] == uid]
    if len(inter) == 0:
        continue
    embeddings_list = []
    ratings_list = []
    for _, row in inter.iterrows():
        if row['resource_id'] in resource_to_idx:
            embeddings_list.append(tfidf_dense[resource_to_idx[row['resource_id']]])
            ratings_list.append(row['rating'])
    if embeddings_list:
        perfil = np.average(embeddings_list, axis=0, weights=ratings_list)
        norma = np.linalg.norm(perfil)
        if norma > 0:
            perfil = perfil / norma
        perfiles_tfidf[idx_u] = perfil

print(f"    TF-IDF baseline: {tfidf_dense.shape}")

def scores_contenido_tfidf(user_id):
    idx_u = user_to_idx_cb[user_id]
    perfil = perfiles_tfidf[idx_u:idx_u+1]
    # Normalizar recursos TF-IDF para similitud del coseno
    norms = np.linalg.norm(tfidf_dense, axis=1, keepdims=True)
    norms[norms == 0] = 1
    tfidf_norm = tfidf_dense / norms
    return np.dot(tfidf_norm, perfil.T).flatten()


# =============================================================================
# PASO 6.3: Métricas de evaluación
# =============================================================================

def precision_at_k(recomendados, relevantes, k):
    """Proporción de ítems relevantes en las top K recomendaciones."""
    top_k = recomendados[:k]
    hits = len(set(top_k) & set(relevantes))
    return hits / k

def recall_at_k(recomendados, relevantes, k):
    """Proporción de ítems relevantes recuperados en las top K."""
    if len(relevantes) == 0:
        return 0.0
    top_k = recomendados[:k]
    hits = len(set(top_k) & set(relevantes))
    return hits / len(relevantes)

def ndcg_at_k(recomendados, relevantes, k):
    """
    Normalized Discounted Cumulative Gain.
    Mide no solo SI aparecen ítems relevantes, sino EN QUÉ POSICIÓN.
    Un ítem relevante en posición 1 vale más que uno en posición 10.
    """
    top_k = recomendados[:k]
    
    # DCG: sum of 1/log2(pos+1) for relevant items
    dcg = 0.0
    for i, rid in enumerate(top_k):
        if rid in relevantes:
            dcg += 1.0 / np.log2(i + 2)  # +2 porque posiciones empiezan en 1
    
    # IDCG: DCG ideal (todos los relevantes al inicio)
    n_rel = min(len(relevantes), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n_rel))
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def generar_ranking(scores, consumidos_set):
    """Genera lista ordenada de resource_ids excluyendo consumidos."""
    ranking_indices = np.argsort(-scores)
    ranking = []
    for idx in ranking_indices:
        rid = df_recursos.iloc[idx]['resource_id']
        if rid not in consumidos_set:
            ranking.append(rid)
    return ranking


# =============================================================================
# PASO 6.4: Evaluación completa
# =============================================================================
print(f"\n Evaluando los modelos...")

# Identificar usuarios para cada escenario
inter_por_usuario_train = df_train.groupby('user_id').size()

# Escenario 1: todos los usuarios con interacciones relevantes en test
usuarios_test = df_test['user_id'].unique()

# Escenario 2: usuarios cold-start (< UMBRAL_COLD_START interacciones en train)
usuarios_coldstart = []
for uid in usuarios_test:
    n_inter = inter_por_usuario_train.get(uid, 0)
    if n_inter < UMBRAL_COLD_START:
        usuarios_coldstart.append(uid)

# Si no hay usuarios con <5 interacciones, usar el percentil inferior
if len(usuarios_coldstart) < 20:
    print(f"     Solo {len(usuarios_coldstart)} usuarios con <{UMBRAL_COLD_START} interacciones")
    print(f"   Usando percentil 10 inferior como cold-start simulado...")
    p10 = inter_por_usuario_train.quantile(0.10)
    usuarios_coldstart = [
        uid for uid in usuarios_test
        if inter_por_usuario_train.get(uid, 0) <= p10
    ]

print(f"   Escenario 1 (General): {len(usuarios_test)} usuarios")
print(f"   Escenario 2 (Cold-start): {len(usuarios_coldstart)} usuarios")
print(f"   Umbral de relevancia: rating ≥ {UMBRAL_RELEVANCIA}")
print(f"   Valores de K: {K_VALORES}")

# Definir modelos a evaluar
modelos = {
    'CB-SBERT': scores_contenido_sbert,
    'CB-TFIDF': scores_contenido_tfidf,
    'CF-LightFM': scores_colaborativo,
    'Híbrido': lambda uid: scores_hibrido(uid, ALPHA_OPTIMO),
}

def evaluar_escenario(usuarios, nombre_escenario):
    """Evalúa todos los modelos sobre un conjunto de usuarios."""
    print(f"\n    Escenario: {nombre_escenario} ({len(usuarios)} usuarios)")
    
    # Almacenar métricas por usuario (para pruebas estadísticas)
    resultados_por_usuario = {modelo: {k: {'p': [], 'r': [], 'ndcg': []} 
                              for k in K_VALORES} for modelo in modelos}
    
    n_evaluados = 0
    for uid in usuarios:
        # Recursos relevantes en test (rating >= umbral)
        relevantes = set(
            df_test[(df_test['user_id'] == uid) & 
                    (df_test['rating'] >= UMBRAL_RELEVANCIA)]['resource_id']
        )
        if len(relevantes) == 0:
            continue
        
        consumidos = set(df_train[df_train['user_id'] == uid]['resource_id'])
        n_evaluados += 1
        
        for nombre_modelo, func_scores in modelos.items():
            scores = func_scores(uid)
            ranking = generar_ranking(scores, consumidos)
            
            for k in K_VALORES:
                p = precision_at_k(ranking, relevantes, k)
                r = recall_at_k(ranking, relevantes, k)
                n = ndcg_at_k(ranking, relevantes, k)
                resultados_por_usuario[nombre_modelo][k]['p'].append(p)
                resultados_por_usuario[nombre_modelo][k]['r'].append(r)
                resultados_por_usuario[nombre_modelo][k]['ndcg'].append(n)
    
    print(f"   Usuarios evaluados (con ítems relevantes en test): {n_evaluados}")
    
    # Calcular promedios
    resumen = {}
    for modelo in modelos:
        resumen[modelo] = {}
        for k in K_VALORES:
            resumen[modelo][k] = {
                'P@K': np.mean(resultados_por_usuario[modelo][k]['p']),
                'R@K': np.mean(resultados_por_usuario[modelo][k]['r']),
                'NDCG@K': np.mean(resultados_por_usuario[modelo][k]['ndcg']),
                'P@K_std': np.std(resultados_por_usuario[modelo][k]['p']),
                'R@K_std': np.std(resultados_por_usuario[modelo][k]['r']),
                'NDCG@K_std': np.std(resultados_por_usuario[modelo][k]['ndcg']),
            }
    
    return resumen, resultados_por_usuario, n_evaluados


# Ejecutar evaluación
inicio = time.time()
resumen_general, por_usuario_general, n_general = evaluar_escenario(usuarios_test, "General")
resumen_coldstart, por_usuario_coldstart, n_coldstart = evaluar_escenario(usuarios_coldstart, "Cold-start")
tiempo_eval = time.time() - inicio
print(f"\n     Tiempo de evaluación: {tiempo_eval:.1f} segundos")


# =============================================================================
# PASO 6.5: Mostrar resultados en tablas
# =============================================================================
print(f"\n{'=' * 70}")
print("RESULTADOS — ESCENARIO 1: GENERAL")
print(f"{'=' * 70}")

print(f"\n   {'Modelo':<15s}", end="")
for k in K_VALORES:
    print(f" | {'P@'+str(k):>7s}  {'R@'+str(k):>7s}  {'NDCG@'+str(k):>8s}", end="")
print()
print(f"   {'-'*15}", end="")
for _ in K_VALORES:
    print(f" | {'-'*7}  {'-'*7}  {'-'*8}", end="")
print()

for modelo in modelos:
    print(f"   {modelo:<15s}", end="")
    for k in K_VALORES:
        r = resumen_general[modelo][k]
        print(f" | {r['P@K']:7.4f}  {r['R@K']:7.4f}  {r['NDCG@K']:8.4f}", end="")
    print()

print(f"\n{'=' * 70}")
print("RESULTADOS — ESCENARIO 2: COLD-START")
print(f"{'=' * 70}")

print(f"\n   {'Modelo':<15s}", end="")
for k in K_VALORES:
    print(f" | {'P@'+str(k):>7s}  {'R@'+str(k):>7s}  {'NDCG@'+str(k):>8s}", end="")
print()
print(f"   {'-'*15}", end="")
for _ in K_VALORES:
    print(f" | {'-'*7}  {'-'*7}  {'-'*8}", end="")
print()

for modelo in modelos:
    print(f"   {modelo:<15s}", end="")
    for k in K_VALORES:
        r = resumen_coldstart[modelo][k]
        print(f" | {r['P@K']:7.4f}  {r['R@K']:7.4f}  {r['NDCG@K']:8.4f}", end="")
    print()


# =============================================================================
# PASO 6.6: Contraste de hipótesis
# =============================================================================
print(f"\n{'=' * 70}")
print("CONTRASTE DE HIPÓTESIS")
print(f"{'=' * 70}")

K_REF = 10  # K de referencia para las pruebas

# --- H₁: Híbrido ≥ 15% mejor que el mejor individual ---
print(f"\n H₁: El modelo híbrido supera al mejor modelo individual en ≥15% (NDCG@{K_REF})")

ndcg_hibrido = por_usuario_general['Híbrido'][K_REF]['ndcg']
ndcg_cb = por_usuario_general['CB-SBERT'][K_REF]['ndcg']
ndcg_cf = por_usuario_general['CF-LightFM'][K_REF]['ndcg']

mean_hibrido = np.mean(ndcg_hibrido)
mean_cb = np.mean(ndcg_cb)
mean_cf = np.mean(ndcg_cf)
mejor_individual = max(mean_cb, mean_cf)
mejor_nombre = "CB-SBERT" if mean_cb >= mean_cf else "CF-LightFM"
mejora_h1 = (mean_hibrido - mejor_individual) / mejor_individual * 100

# Test de Wilcoxon: ¿el híbrido es significativamente mejor?
ndcg_mejor = ndcg_cb if mean_cb >= mean_cf else ndcg_cf
stat_h1, p_value_h1 = stats.wilcoxon(ndcg_hibrido, ndcg_mejor, alternative='greater')

print(f"   NDCG@{K_REF} Híbrido:          {mean_hibrido:.4f}")
print(f"   NDCG@{K_REF} Mejor individual:  {mejor_individual:.4f} ({mejor_nombre})")
print(f"   Mejora porcentual:          {mejora_h1:.1f}%")
print(f"   Wilcoxon stat:              {stat_h1:.1f}")
print(f"   p-value:                    {p_value_h1:.6f}")
print(f"   Significativo (p < 0.05):   {'SÍ ' if p_value_h1 < ALPHA_SIGNIFICANCIA else 'NO '}")
print(f"   Mejora ≥ 15%:               {'SÍ ' if mejora_h1 >= 15 else 'NO '}")
h1_aceptada = p_value_h1 < ALPHA_SIGNIFICANCIA and mejora_h1 >= 15

# --- H₂: SBERT > TF-IDF ---
print(f"\n H₂: SBERT supera a TF-IDF como representación de contenido (NDCG@{K_REF})")

ndcg_sbert = por_usuario_general['CB-SBERT'][K_REF]['ndcg']
ndcg_tfidf = por_usuario_general['CB-TFIDF'][K_REF]['ndcg']

mean_sbert = np.mean(ndcg_sbert)
mean_tfidf = np.mean(ndcg_tfidf)
mejora_h2 = (mean_sbert - mean_tfidf) / mean_tfidf * 100 if mean_tfidf > 0 else float('inf')

stat_h2, p_value_h2 = stats.wilcoxon(ndcg_sbert, ndcg_tfidf, alternative='greater')

print(f"   NDCG@{K_REF} CB-SBERT:  {mean_sbert:.4f}")
print(f"   NDCG@{K_REF} CB-TFIDF:  {mean_tfidf:.4f}")
print(f"   Mejora porcentual:  {mejora_h2:.1f}%")
print(f"   Wilcoxon stat:      {stat_h2:.1f}")
print(f"   p-value:            {p_value_h2:.6f}")
print(f"   Significativo:      {'SÍ ' if p_value_h2 < ALPHA_SIGNIFICANCIA else 'NO '}")
h2_aceptada = p_value_h2 < ALPHA_SIGNIFICANCIA

# --- H₃: Híbrido > CF en cold-start ---
print(f"\n H₃: El híbrido supera al CF puro en escenario cold-start (NDCG@{K_REF})")

ndcg_hibrido_cs = por_usuario_coldstart['Híbrido'][K_REF]['ndcg']
ndcg_cf_cs = por_usuario_coldstart['CF-LightFM'][K_REF]['ndcg']

mean_hibrido_cs = np.mean(ndcg_hibrido_cs)
mean_cf_cs = np.mean(ndcg_cf_cs)
mejora_h3 = (mean_hibrido_cs - mean_cf_cs) / mean_cf_cs * 100 if mean_cf_cs > 0 else float('inf')

if len(ndcg_hibrido_cs) >= 10:
    stat_h3, p_value_h3 = stats.wilcoxon(ndcg_hibrido_cs, ndcg_cf_cs, alternative='greater')
else:
    # Muestra muy pequeña: usar Mann-Whitney
    stat_h3, p_value_h3 = stats.mannwhitneyu(ndcg_hibrido_cs, ndcg_cf_cs, alternative='greater')

print(f"   NDCG@{K_REF} Híbrido (cold-start):  {mean_hibrido_cs:.4f}")
print(f"   NDCG@{K_REF} CF (cold-start):        {mean_cf_cs:.4f}")
print(f"   Mejora porcentual:              {mejora_h3:.1f}%")
print(f"   Wilcoxon stat:                  {stat_h3:.1f}")
print(f"   p-value:                        {p_value_h3:.6f}")
print(f"   Significativo:                  {'SÍ ' if p_value_h3 < ALPHA_SIGNIFICANCIA else 'NO '}")
h3_aceptada = p_value_h3 < ALPHA_SIGNIFICANCIA


# =============================================================================
# PASO 6.7: Guardar resultados
# =============================================================================
resultados_completos = {
    'resumen_general': resumen_general,
    'resumen_coldstart': resumen_coldstart,
    'por_usuario_general': por_usuario_general,
    'por_usuario_coldstart': por_usuario_coldstart,
    'n_general': n_general,
    'n_coldstart': n_coldstart,
    'hipotesis': {
        'H1': {'aceptada': h1_aceptada, 'mejora': mejora_h1, 'p_value': p_value_h1, 'stat': stat_h1},
        'H2': {'aceptada': h2_aceptada, 'mejora': mejora_h2, 'p_value': p_value_h2, 'stat': stat_h2},
        'H3': {'aceptada': h3_aceptada, 'mejora': mejora_h3, 'p_value': p_value_h3, 'stat': stat_h3},
    }
}

with open(os.path.join(MODELOS_DIR, 'resultados_evaluacion.pkl'), 'wb') as f:
    pickle.dump(resultados_completos, f)
print(f"\n Resultados guardados en modelos/resultados_evaluacion.pkl")


# =============================================================================
# PASO 6.8: Visualizaciones
# =============================================================================
print("\n Generando visualizaciones...")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 11
plt.style.use(ESTILO_MATPLOTLIB)

colores = {
    'CB-SBERT': COLORES_MODELOS['Basado en Contenido'],
    'CB-TFIDF': '#9E9E9E',  # Gris para el baseline
    'CF-LightFM': COLORES_MODELOS['Filtrado Colaborativo'],
    'Híbrido': COLORES_MODELOS['Híbrido'],
}

modelos_principales = ['CB-SBERT', 'CF-LightFM', 'Híbrido']

# --- Gráfica 17: Comparación de métricas - Escenario General ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
metricas_nombres = ['P@K', 'R@K', 'NDCG@K']
metricas_labels = ['Precision@K', 'Recall@K', 'NDCG@K']

for ax, metrica, label in zip(axes, metricas_nombres, metricas_labels):
    x = np.arange(len(K_VALORES))
    width = 0.25
    
    for i, modelo in enumerate(modelos_principales):
        valores = [resumen_general[modelo][k][metrica] for k in K_VALORES]
        stds = [resumen_general[modelo][k][metrica+'_std'] for k in K_VALORES]
        bars = ax.bar(x + i*width, valores, width, label=modelo,
                      color=colores[modelo], alpha=0.85, edgecolor='white',
                      yerr=stds, capsize=3)
        # Etiquetas sobre las barras
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('K')
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(k) for k in K_VALORES])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Escenario 1: Evaluación General — Comparación de Modelos', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '17_comparacion_metricas_general.png'), dpi=DPI)
plt.close()
print("    17_comparacion_metricas_general.png")


# --- Gráfica 18: Comparación de métricas - Escenario Cold-start ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for ax, metrica, label in zip(axes, metricas_nombres, metricas_labels):
    x = np.arange(len(K_VALORES))
    width = 0.25
    
    for i, modelo in enumerate(modelos_principales):
        valores = [resumen_coldstart[modelo][k][metrica] for k in K_VALORES]
        stds = [resumen_coldstart[modelo][k][metrica+'_std'] for k in K_VALORES]
        bars = ax.bar(x + i*width, valores, width, label=modelo,
                      color=colores[modelo], alpha=0.85, edgecolor='white',
                      yerr=stds, capsize=3)
        for bar, val in zip(bars, valores):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('K')
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.set_xticks(x + width)
    ax.set_xticklabels([str(k) for k in K_VALORES])
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)

plt.suptitle('Escenario 2: Cold-Start — Comparación de Modelos', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '18_comparacion_metricas_coldstart.png'), dpi=DPI)
plt.close()
print("    18_comparacion_metricas_coldstart.png")


# --- Gráfica 19: Mejora porcentual del híbrido ---
fig, ax = plt.subplots(figsize=FIGSIZE_NORMAL)

mejoras_cb = [(resumen_general['Híbrido'][k]['NDCG@K'] - resumen_general['CB-SBERT'][k]['NDCG@K']) 
              / resumen_general['CB-SBERT'][k]['NDCG@K'] * 100 for k in K_VALORES]
mejoras_cf = [(resumen_general['Híbrido'][k]['NDCG@K'] - resumen_general['CF-LightFM'][k]['NDCG@K']) 
              / resumen_general['CF-LightFM'][k]['NDCG@K'] * 100 for k in K_VALORES]

x = np.arange(len(K_VALORES))
width = 0.35

bars1 = ax.bar(x - width/2, mejoras_cb, width, label='vs CB-SBERT',
               color=COLORES_MODELOS['Basado en Contenido'], alpha=0.8)
bars2 = ax.bar(x + width/2, mejoras_cf, width, label='vs CF-LightFM',
               color=COLORES_MODELOS['Filtrado Colaborativo'], alpha=0.8)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

ax.axhline(y=15, color='red', linestyle='--', linewidth=1.5, label='Umbral H₁ (15%)')
ax.set_xlabel('K')
ax.set_ylabel('Mejora porcentual (%)')
ax.set_title('Mejora del Modelo Híbrido sobre Modelos Individuales (NDCG@K)')
ax.set_xticks(x)
ax.set_xticklabels([f'K={k}' for k in K_VALORES])
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '19_mejora_porcentual_hibrido.png'), dpi=DPI)
plt.close()
print("    19_mejora_porcentual_hibrido.png")


# --- Gráfica 20: Distribución de NDCG@10 por modelo (violin/box) ---
fig, ax = plt.subplots(figsize=FIGSIZE_NORMAL)

data_violin = []
labels_violin = []
colors_violin = []
for modelo in modelos_principales:
    data_violin.append(por_usuario_general[modelo][K_REF]['ndcg'])
    labels_violin.append(modelo)
    colors_violin.append(colores[modelo])

bp = ax.boxplot(data_violin, labels=labels_violin, patch_artist=True, widths=0.6)
for patch, color in zip(bp['boxes'], colors_violin):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Agregar medias
means = [np.mean(d) for d in data_violin]
ax.scatter(range(1, len(means)+1), means, color='red', s=80, zorder=5, marker='D', label='Media')

for i, m in enumerate(means):
    ax.text(i+1.3, m, f'{m:.4f}', va='center', fontsize=10, fontweight='bold')

ax.set_ylabel(f'NDCG@{K_REF}')
ax.set_title(f'Distribución de NDCG@{K_REF} por Modelo (Escenario General)')
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '20_distribucion_ndcg_por_modelo.png'), dpi=DPI)
plt.close()
print("    20_distribucion_ndcg_por_modelo.png")


# --- Gráfica 21: Resumen de hipótesis ---
fig, ax = plt.subplots(figsize=(12, 5))

hipotesis = ['H₁: Híbrido ≥15%\nmejor que individual', 
             'H₂: SBERT > TF-IDF\nen contenido',
             'H₃: Híbrido > CF\nen cold-start']
resultados_h = [h1_aceptada, h2_aceptada, h3_aceptada]
p_values = [p_value_h1, p_value_h2, p_value_h3]
mejoras = [mejora_h1, mejora_h2, mejora_h3]

bar_colors = ['#4CAF50' if r else '#F44336' for r in resultados_h]
bars = ax.barh(range(len(hipotesis)), mejoras, color=bar_colors, alpha=0.8, edgecolor='white', height=0.6)

for i, (bar, p, aceptada) in enumerate(zip(bars, p_values, resultados_h)):
    estado = " Aceptada" if aceptada else " No aceptada"
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{mejoras[i]:.1f}% | p={p:.4f} | {estado}',
            va='center', fontsize=11, fontweight='bold')

ax.set_yticks(range(len(hipotesis)))
ax.set_yticklabels(hipotesis, fontsize=11)
ax.set_xlabel('Mejora porcentual (%)')
ax.set_title('Resumen del Contraste de Hipótesis', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(axis='x', alpha=0.3)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '21_hipotesis_resumen.png'), dpi=DPI)
plt.close()
print("    21_hipotesis_resumen.png")


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print(f"\n{'=' * 70}")
print("PASO 6 COMPLETADO: Evaluación y contraste de hipótesis")
print(f"{'=' * 70}")
print(f"  Usuarios evaluados (general):    {n_general}")
print(f"  Usuarios evaluados (cold-start): {n_coldstart}")
print(f"  Tiempo de evaluación: {tiempo_eval:.1f} segundos")
print(f"\n  RESUMEN DE HIPÓTESIS:")
print(f"  H₁ (Híbrido ≥15% mejor): {'ACEPTADA ' if h1_aceptada else 'NO ACEPTADA '} (mejora: {mejora_h1:.1f}%, p={p_value_h1:.4f})")
print(f"  H₂ (SBERT > TF-IDF):     {'ACEPTADA ' if h2_aceptada else 'NO ACEPTADA '} (mejora: {mejora_h2:.1f}%, p={p_value_h2:.4f})")
print(f"  H₃ (Híbrido > CF cold):  {'ACEPTADA ' if h3_aceptada else 'NO ACEPTADA '} (mejora: {mejora_h3:.1f}%, p={p_value_h3:.4f})")
print(f"\n  Gráficas generadas: 17-21")
print(f"{'=' * 70}")
