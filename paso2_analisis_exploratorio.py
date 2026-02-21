"""
=============================================================================
paso2_analisis_exploratorio.py — Análisis exploratorio y gráficas descriptivas
=============================================================================

PROPÓSITO:
    Generar las gráficas estadísticas. Incluye:
    
    - Gráficas de datos numéricos: histogramas, box plots
    - Gráficas de datos categóricos: barras, pastel
    - Detección visual de patrones: heatmap de densidad, scatter
    - Tablas de frecuencia y medidas de tendencia central


SALIDAS:
    - graficas/01_distribucion_ratings.png
    - graficas/02_distribucion_semestres.png
    - graficas/03_distribucion_carreras.png
    - graficas/04_distribucion_temas.png
    - graficas/05_boxplot_ratings_por_carrera.png
    - graficas/06_interacciones_por_usuario.png
    - graficas/07_heatmap_carrera_tema.png
    - graficas/08_ratings_alineados_vs_aleatorios.png
    - graficas/09_evolucion_temporal.png
    - graficas/10_matriz_densidad.png
    - Imprime tablas de frecuencia y medidas centrales en consola

EJECUCIÓN:
    python paso2_analisis_exploratorio.py
    (requiere haber ejecutado paso1_generar_datos.py primero)
=============================================================================
"""
# Importar constantes necesarias
from config import N_USUARIOS, N_RECURSOS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from config import (
    fijar_semillas, SEED, GRAFICAS_DIR, INTERESES_POR_CARRERA,
    RUTA_USUARIOS, RUTA_RECURSOS, RUTA_INTERACCIONES,
    ESTILO_MATPLOTLIB, FIGSIZE_NORMAL, FIGSIZE_GRANDE, DPI
)
import os

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
fijar_semillas(SEED)
plt.style.use(ESTILO_MATPLOTLIB)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

print("=" * 70)
print("PASO 2: Análisis exploratorio de datos")
print("=" * 70)

# Cargar datos
print("\n Cargando datos...")
df_usuarios = pd.read_csv(RUTA_USUARIOS)
df_recursos = pd.read_csv(RUTA_RECURSOS)
df_interacciones = pd.read_csv(RUTA_INTERACCIONES)
df_interacciones['timestamp'] = pd.to_datetime(df_interacciones['timestamp'])

print(f"   Usuarios: {len(df_usuarios):,}")
print(f"   Recursos: {len(df_recursos):,}")
print(f"   Interacciones: {len(df_interacciones):,}")


# =============================================================================
# TABLAS DE FRECUENCIA
# =============================================================================
# La plantilla pide "explicar las tablas de frecuencia empleadas según
# sea la variable utilizada"

print(f"\n{'=' * 70}")
print("TABLAS DE FRECUENCIA")
print(f"{'=' * 70}")

# Tabla de frecuencia: Temas de recursos
print("\n Tabla de frecuencia — Temas de recursos educativos:")
freq_temas = df_recursos['tema'].value_counts().reset_index()
freq_temas.columns = ['Tema', 'Frecuencia']
freq_temas['Porcentaje'] = (freq_temas['Frecuencia'] / len(df_recursos) * 100).round(1)
freq_temas['% Acumulado'] = freq_temas['Porcentaje'].cumsum().round(1)
print(freq_temas.to_string(index=False))

# Tabla de frecuencia: Ratings
print("\n Tabla de frecuencia — Distribución de ratings:")
freq_ratings = df_interacciones['rating'].value_counts().sort_index().reset_index()
freq_ratings.columns = ['Rating', 'Frecuencia']
freq_ratings['Porcentaje'] = (freq_ratings['Frecuencia'] / len(df_interacciones) * 100).round(1)
freq_ratings['% Acumulado'] = freq_ratings['Porcentaje'].cumsum().round(1)
print(freq_ratings.to_string(index=False))

# Tabla de frecuencia: Carreras
print("\n Tabla de frecuencia — Distribución por carrera:")
freq_carreras = df_usuarios['carrera'].value_counts().reset_index()
freq_carreras.columns = ['Carrera', 'Frecuencia']
freq_carreras['Porcentaje'] = (freq_carreras['Frecuencia'] / len(df_usuarios) * 100).round(1)
print(freq_carreras.to_string(index=False))


# =============================================================================
# MEDIDAS DE TENDENCIA CENTRAL Y DISPERSIÓN
# =============================================================================

print(f"\n{'=' * 70}")
print("MEDIDAS DE TENDENCIA CENTRAL Y DISPERSIÓN")
print(f"{'=' * 70}")

# Rating
print("\n Variable: Rating (escala 1-5)")
print(f"   Media:            {df_interacciones['rating'].mean():.2f}")
print(f"   Mediana:          {df_interacciones['rating'].median():.0f}")
print(f"   Moda:             {df_interacciones['rating'].mode()[0]}")
print(f"   Desviación est.:  {df_interacciones['rating'].std():.2f}")
print(f"   Varianza:         {df_interacciones['rating'].var():.2f}")
print(f"   Rango:            {df_interacciones['rating'].min()} - {df_interacciones['rating'].max()}")
print(f"   Rango intercuartil (IQR): {df_interacciones['rating'].quantile(0.75) - df_interacciones['rating'].quantile(0.25):.2f}")
print(f"   Asimetría (skew): {df_interacciones['rating'].skew():.3f}")
print(f"   Curtosis:         {df_interacciones['rating'].kurtosis():.3f}")

# Semestre
print("\n Variable: Semestre (1-9)")
print(f"   Media:            {df_usuarios['semestre'].mean():.2f}")
print(f"   Mediana:          {df_usuarios['semestre'].median():.0f}")
print(f"   Desviación est.:  {df_usuarios['semestre'].std():.2f}")

# Interacciones por usuario
inter_por_usuario = df_interacciones.groupby('user_id').size()
print("\n Variable: Interacciones por usuario")
print(f"   Media:            {inter_por_usuario.mean():.1f}")
print(f"   Mediana:          {inter_por_usuario.median():.0f}")
print(f"   Desviación est.:  {inter_por_usuario.std():.1f}")
print(f"   Mínimo:           {inter_por_usuario.min()}")
print(f"   Máximo:           {inter_por_usuario.max()}")

# Interacciones por recurso
inter_por_recurso = df_interacciones.groupby('resource_id').size()
print("\n Variable: Interacciones por recurso")
print(f"   Media:            {inter_por_recurso.mean():.1f}")
print(f"   Mediana:          {inter_por_recurso.median():.0f}")
print(f"   Desviación est.:  {inter_por_recurso.std():.1f}")
print(f"   Mínimo:           {inter_por_recurso.min()}")
print(f"   Máximo:           {inter_por_recurso.max()}")


# =============================================================================
# GRÁFICA 1: Distribución de ratings (histograma + KDE)
# =============================================================================
# ¿Qué muestra? Cómo se distribuyen las calificaciones que los usuarios
# dan a los recursos. Esperamos ver un sesgo hacia ratings altos (3-4)
# porque el 70% de las interacciones son con recursos de interés.

print(f"\n Generando gráficas...")

fig, ax = plt.subplots(figsize=FIGSIZE_NORMAL)
bars = ax.bar(
    freq_ratings['Rating'], freq_ratings['Frecuencia'],
    color=['#ef5350', '#ff7043', '#ffca28', '#66bb6a', '#42a5f5'],
    edgecolor='white', linewidth=1.5
)
# Agregar etiquetas sobre cada barra
for bar, pct in zip(bars, freq_ratings['Porcentaje']):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
            f'{pct}%', ha='center', va='bottom', fontweight='bold')

ax.set_xlabel('Rating')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución de Ratings de Interacciones')
ax.set_xticks([1, 2, 3, 4, 5])
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '01_distribucion_ratings.png'), dpi=DPI)
plt.close()
print("    01_distribucion_ratings.png")


# =============================================================================
# GRÁFICA 2: Distribución de semestres (histograma)
# =============================================================================
# ¿Qué muestra? La pirámide de matrícula — más estudiantes en semestres
# intermedios (3-6), menos en primero y último.

fig, ax = plt.subplots(figsize=FIGSIZE_NORMAL)
semestre_counts = df_usuarios['semestre'].value_counts().sort_index()
ax.bar(semestre_counts.index, semestre_counts.values, color='#5c6bc0', edgecolor='white')
for x, y in zip(semestre_counts.index, semestre_counts.values):
    ax.text(x, y + 2, str(y), ha='center', va='bottom', fontweight='bold')
ax.set_xlabel('Semestre')
ax.set_ylabel('Número de Estudiantes')
ax.set_title('Distribución de Estudiantes por Semestre')
ax.set_xticks(range(1, 10))
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '02_distribucion_semestres.png'), dpi=DPI)
plt.close()
print("    02_distribucion_semestres.png")


# =============================================================================
# GRÁFICA 3: Distribución por carrera (barras horizontales)
# =============================================================================
# ¿Qué muestra? La composición de la población estudiantil por programa
# académico, verificando que coincide con los porcentajes diseñados.

fig, ax = plt.subplots(figsize=FIGSIZE_NORMAL)
carrera_counts = df_usuarios['carrera'].value_counts()
colors = ['#26a69a', '#42a5f5', '#ab47bc', '#ef5350']
bars = ax.barh(carrera_counts.index, carrera_counts.values, color=colors, edgecolor='white')
for bar, val in zip(bars, carrera_counts.values):
    ax.text(val + 5, bar.get_y() + bar.get_height()/2.,
            f'{val} ({val/len(df_usuarios)*100:.0f}%)',
            ha='left', va='center', fontweight='bold')
ax.set_xlabel('Número de Estudiantes')
ax.set_title('Distribución de Estudiantes por Carrera')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '03_distribucion_carreras.png'), dpi=DPI)
plt.close()
print("    03_distribucion_carreras.png")


# =============================================================================
# GRÁFICA 4: Distribución de temas de recursos (barras)
# =============================================================================
# ¿Qué muestra? Cuántos recursos hay por tema. Esperamos distribución
# aproximadamente uniforme ya que fue diseñada así.

fig, ax = plt.subplots(figsize=FIGSIZE_GRANDE)
tema_counts = df_recursos['tema'].value_counts()
ax.bar(range(len(tema_counts)), tema_counts.values, color='#7e57c2', edgecolor='white')
ax.set_xticks(range(len(tema_counts)))
ax.set_xticklabels(tema_counts.index, rotation=45, ha='right', fontsize=10)
for i, v in enumerate(tema_counts.values):
    ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)
ax.set_ylabel('Número de Recursos')
ax.set_title('Distribución de Recursos por Tema')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '04_distribucion_temas.png'), dpi=DPI)
plt.close()
print("    04_distribucion_temas.png")


# =============================================================================
# GRÁFICA 5: Box plot de ratings por carrera
# =============================================================================
# ¿Qué muestra? Si hay diferencias en cómo los estudiantes de diferentes
# carreras califican los recursos. Permite ver mediana, IQR y outliers.

# Merge para obtener carrera de cada interacción
df_inter_users = df_interacciones.merge(df_usuarios[['user_id', 'carrera']], on='user_id')

fig, ax = plt.subplots(figsize=FIGSIZE_NORMAL)
carreras_unicas = sorted(df_inter_users['carrera'].unique())
data_boxplot = [df_inter_users[df_inter_users['carrera'] == c]['rating'].values for c in carreras_unicas]
bp = ax.boxplot(data_boxplot, labels=carreras_unicas, patch_artist=True)
colors_box = ['#26a69a', '#42a5f5', '#ab47bc', '#ef5350']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel('Rating')
ax.set_title('Distribución de Ratings por Carrera')
ax.set_xticklabels(carreras_unicas, rotation=15, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '05_boxplot_ratings_por_carrera.png'), dpi=DPI)
plt.close()
print("    05_boxplot_ratings_por_carrera.png")


# =============================================================================
# GRÁFICA 6: Distribución de interacciones por usuario (histograma)
# =============================================================================
# ¿Qué muestra? Cuántas interacciones tiene cada usuario. Permite
# identificar usuarios muy activos y usuarios con pocas interacciones
# (candidatos a cold-start).

fig, ax = plt.subplots(figsize=FIGSIZE_NORMAL)
ax.hist(inter_por_usuario.values, bins=30, color='#00897b', edgecolor='white', alpha=0.8)
ax.axvline(inter_por_usuario.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Media = {inter_por_usuario.mean():.1f}')
ax.axvline(inter_por_usuario.median(), color='orange', linestyle='--', linewidth=2,
           label=f'Mediana = {inter_por_usuario.median():.0f}')
ax.set_xlabel('Número de Interacciones')
ax.set_ylabel('Número de Usuarios')
ax.set_title('Distribución de Interacciones por Usuario')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '06_interacciones_por_usuario.png'), dpi=DPI)
plt.close()
print("    06_interacciones_por_usuario.png")


# =============================================================================
# GRÁFICA 7: Heatmap carrera × tema (patrón de interacción)
# =============================================================================
# ¿Qué muestra? La intensidad de interacciones entre cada carrera y cada
# tema. Esperamos ver "bloques calientes" donde la carrera se alinea con
# sus temas de interés (el sesgo del 70%).
# ESTA ES UNA DE LAS GRÁFICAS MÁS IMPORTANTES: demuestra visualmente
# que el sesgo de interacción funciona correctamente.

df_inter_recursos = df_inter_users.merge(df_recursos[['resource_id', 'tema']], on='resource_id')

# Tabla cruzada: cuántas interacciones por carrera × tema
cross_tab = pd.crosstab(df_inter_recursos['carrera'], df_inter_recursos['tema'])
# Normalizar por fila (porcentaje de interacciones de cada carrera por tema)
cross_tab_norm = cross_tab.div(cross_tab.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(14, 6))
sns.heatmap(cross_tab_norm, annot=True, fmt='.1f', cmap='YlOrRd',
            ax=ax, linewidths=0.5, cbar_kws={'label': '% de Interacciones'})
ax.set_title('Distribución de Interacciones: Carrera × Tema (%)')
ax.set_xlabel('Tema del Recurso')
ax.set_ylabel('Carrera del Estudiante')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '07_heatmap_carrera_tema.png'), dpi=DPI)
plt.close()
print("    07_heatmap_carrera_tema.png")


# =============================================================================
# GRÁFICA 8: Comparación ratings alineados vs. aleatorios
# =============================================================================
# ¿Qué muestra? Que las interacciones donde el tema coincide con los
# intereses del usuario tienen ratings significativamente más altos.
# Esto valida el sesgo diseñado y es relevante para justificar H₁.

def es_interaccion_alineada(row, df_usuarios_dict):
    """Determina si una interacción es alineada con los intereses del usuario."""
    intereses = df_usuarios_dict.get(row['user_id'], [])
    return row['tema'] in intereses

# Crear diccionario de intereses para búsqueda rápida
intereses_dict = {}
for _, u in df_usuarios.iterrows():
    intereses_dict[u['user_id']] = u['intereses'].split('|')

df_inter_recursos['alineada'] = df_inter_recursos.apply(
    lambda row: es_interaccion_alineada(row, intereses_dict), axis=1
)

fig, ax = plt.subplots(figsize=FIGSIZE_NORMAL)
data_alineada = df_inter_recursos[df_inter_recursos['alineada']]['rating']
data_aleatoria = df_inter_recursos[~df_inter_recursos['alineada']]['rating']

positions = [1, 2]
bp = ax.boxplot([data_alineada.values, data_aleatoria.values],
                labels=['Alineada\n(tema = interés)', 'Exploración\n(tema ≠ interés)'],
                patch_artist=True, widths=0.6)
bp['boxes'][0].set_facecolor('#66bb6a')
bp['boxes'][1].set_facecolor('#ef5350')
for box in bp['boxes']:
    box.set_alpha(0.7)

ax.text(1, data_alineada.mean() + 0.15,
        f'μ = {data_alineada.mean():.2f}\nn = {len(data_alineada):,}',
        ha='center', fontsize=11, fontweight='bold')
ax.text(2, data_aleatoria.mean() + 0.15,
        f'μ = {data_aleatoria.mean():.2f}\nn = {len(data_aleatoria):,}',
        ha='center', fontsize=11, fontweight='bold')

ax.set_ylabel('Rating')
ax.set_title('Comparación de Ratings: Interacciones Alineadas vs. Exploración')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '08_ratings_alineados_vs_aleatorios.png'), dpi=DPI)
plt.close()
print("    08_ratings_alineados_vs_aleatorios.png")


# =============================================================================
# GRÁFICA 9: Evolución temporal de interacciones
# =============================================================================
# ¿Qué muestra? Cómo se distribuyen las interacciones en el tiempo.
# Importante para la partición temporal (80/20).

fig, ax = plt.subplots(figsize=FIGSIZE_GRANDE)
df_interacciones['mes'] = df_interacciones['timestamp'].dt.to_period('M')
temporal = df_interacciones.groupby('mes').size()

ax.plot(range(len(temporal)), temporal.values, color='#5c6bc0', linewidth=2)
ax.fill_between(range(len(temporal)), temporal.values, alpha=0.3, color='#5c6bc0')

# Marcar el punto de corte 80/20
corte_idx = int(len(temporal) * 0.8)
ax.axvline(corte_idx, color='red', linestyle='--', linewidth=2,
           label=f'Corte 80/20 (entrenamiento | prueba)')
ax.legend(fontsize=11)

# Etiquetas de tiempo simplificadas
tick_positions = list(range(0, len(temporal), 6))
tick_labels = [str(temporal.index[i]) for i in tick_positions if i < len(temporal)]
ax.set_xticks(tick_positions[:len(tick_labels)])
ax.set_xticklabels(tick_labels, rotation=45, ha='right')

ax.set_xlabel('Mes')
ax.set_ylabel('Número de Interacciones')
ax.set_title('Evolución Temporal de Interacciones')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '09_evolucion_temporal.png'), dpi=DPI)
plt.close()
print("    09_evolucion_temporal.png")


# =============================================================================
# GRÁFICA 10: Densidad de la matriz de interacciones
# =============================================================================
# ¿Qué muestra? Una visualización de la matriz usuario × recurso.
# Los puntos representan interacciones existentes. Permite ver la
# dispersidad (sparsity) de los datos — uno de los problemas clave
# que el sistema híbrido busca resolver.

fig, ax = plt.subplots(figsize=(10, 8))

# Tomar una muestra para visualizar (la matriz completa es muy grande)
muestra_users = sorted(np.random.choice(df_usuarios['user_id'].unique(), 100, replace=False))
muestra_inter = df_interacciones[df_interacciones['user_id'].isin(muestra_users)]

# Crear mapeos para los ejes
user_map = {uid: i for i, uid in enumerate(muestra_users)}
resource_ids = sorted(df_recursos['resource_id'].unique())
resource_map = {rid: i for i, rid in enumerate(resource_ids)}

x = muestra_inter['resource_id'].map(resource_map).dropna()
y = muestra_inter['user_id'].map(user_map).dropna()

# Solo mantener los índices que existen en ambos
mask = x.index.intersection(y.index)
x = x.loc[mask]
y = y.loc[mask]

ax.scatter(x, y, s=1, alpha=0.5, color='#5c6bc0')
ax.set_xlabel(f'Recursos ({len(resource_ids)} totales)')
ax.set_ylabel(f'Usuarios (muestra de {len(muestra_users)})')
ax.set_title(f'Matriz de Interacciones (Densidad: {len(df_interacciones)/(N_USUARIOS*N_RECURSOS)*100:.1f}%)')
plt.tight_layout()
plt.savefig(os.path.join(GRAFICAS_DIR, '10_matriz_densidad.png'), dpi=DPI)
plt.close()
print("    10_matriz_densidad.png")


# =============================================================================
# RESUMEN FINAL
# =============================================================================
print(f"\n{'=' * 70}")
print("ANÁLISIS EXPLORATORIO COMPLETADO")
print(f"{'=' * 70}")
print(f"  10 gráficas guardadas en: {GRAFICAS_DIR}/")
print(f"  Tablas de frecuencia impresas en consola")
print(f"  Medidas centrales y de dispersión calculadas")
print(f"\n  Hallazgos principales:")
print(f"  - Rating promedio: {df_interacciones['rating'].mean():.2f} (sesgo positivo por interacciones alineadas)")
print(f"  - Interacciones alineadas: {df_inter_recursos['alineada'].sum():,} ({df_inter_recursos['alineada'].mean()*100:.1f}%)")
print(f"  - Interacciones por usuario: {inter_por_usuario.mean():.0f} ± {inter_por_usuario.std():.0f}")
print(f"  - Densidad de la matriz: {len(df_interacciones)/(len(df_usuarios)*len(df_recursos))*100:.1f}%")

n_cold_start = (inter_por_usuario < 5).sum()
print(f"  - Usuarios cold-start (<5 interacciones): {n_cold_start} ({n_cold_start/len(df_usuarios)*100:.1f}%)")
print(f"{'=' * 70}")
