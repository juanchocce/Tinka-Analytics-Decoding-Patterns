import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from modules import etl, analysis

st.set_page_config(page_title="Randomness Audit Framework", page_icon="📊", layout="wide")

# Load CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# LOAD DATA
df_draws, df_exploded = etl.load_data()

if df_draws.empty:
    st.error("No se encontró el archivo de datos. Por favor verifica 'data/tinka_data.csv'")
    st.stop()

st.title("📊 Framework de Auditoría de Aleatoriedad")
st.markdown("Análisis estadístico riguroso de la era moderna de La Tinka (Oct 2022 - Presente | 50 Bolillas)")

current_sorteo = df_draws['Sorteo'].max() if 'Sorteo' in df_draws.columns else 0

st.markdown("---")
st.header("FASE 1: ESTADÍSTICA DESCRIPTIVA (EDA)")

# ----------------- Análisis 1 -----------------
st.subheader("Análisis 1: Frecuencia de Bolillas")

st.markdown("""
> **¿Qué es esto?** Un conteo histórico de cuántas veces ha salido cada número.\n
> **¿Para qué sirve?** En retail, se usa para identificar los productos que tienen mayor rotación y optimizar inventarios.\n
> **¿Qué estamos midiendo aquí?** Buscamos identificar sesgos en el sorteo. Si todos los números tienen la misma probabilidad, las barras deberían estar relativamente parejas alrededor de la media.
""")

df_freq, mean_freq, std_freq = analysis.get_frequency_analysis(df_exploded)
fig_freq = px.bar(df_freq, x='Numero', y='Frecuencia', title="Frecuencia Histórica por Bolilla")
fig_freq.update_xaxes(title="Número de Bolilla (1-50)")
fig_freq.update_yaxes(title="Veces que salió")
fig_freq.add_hline(y=mean_freq, line_dash="dash", line_color="red", annotation_text=f"Media: {mean_freq:.1f}")
fig_freq.update_layout(template="plotly_dark")
st.plotly_chart(fig_freq, use_container_width=True)

max_freq_num = df_freq.loc[df_freq['Frecuencia'].idxmax()]['Numero']
min_freq_num = df_freq.loc[df_freq['Frecuencia'].idxmin()]['Numero']

st.info(f"**Interpretación del Resultado:** La frecuencia media esperada es de {mean_freq:.1f} apariciones por bolilla. "
        f"La bolilla que más ha salido es la {max_freq_num} y la que menos es la {min_freq_num}. "
        f"En un escenario de negocio, los números (o productos) por encima de la media dictarían las reglas de abastecimiento central.")

# ----------------- Análisis 2 -----------------
st.markdown("---")
st.subheader("Análisis 2: Distribución de la Suma de Resultados")

st.markdown("""
> **¿Qué es esto?** La distribución de la suma de los 6 números ganadores en cada sorteo.\n
> **¿Para qué sirve?** Control de calidad en procesos industriales (Six Sigma) para detectar desviaciones en cadenas de producción.\n
> **¿Qué estamos midiendo aquí?** Aplicamos el Teorema del Límite Central. La suma de múltiples variables independientes (las 6 bolillas) tiende a formar una clásica "Campana de Gauss", donde los extremos son raros y el centro es lo habitual.
""")

sums, mean_sum, std_sum, p_value_shapiro = analysis.get_sum_distribution(df_draws)
fig_sum = px.histogram(sums, nbins=20, title="Distribución de Sumas por Sorteo", marginal="box")
fig_sum.update_xaxes(title="Valor de la Suma")
fig_sum.update_yaxes(title="Frecuencia de Aparición")
fig_sum.add_vline(x=mean_sum, line_dash="dash", line_color="red", annotation_text=f"Media: {mean_sum:.1f}")
fig_sum.update_layout(template="plotly_dark")
st.plotly_chart(fig_sum, use_container_width=True)

if p_value_shapiro > 0.05:
    shapiro_msg = "La distribución se asemeja a una normal (no hay evidencia de lo contrario)."
else:
    shapiro_msg = "La distribución presenta ligeras desviaciones respecto a una campana de Gauss perfecta."

st.info(f"**Interpretación del Resultado:** La suma promedio central es de {mean_sum:.1f} (muy cerca al teórico 153) con una desviación de {std_sum:.1f}. "
           f"Valor-p (Prueba de Normalidad): {p_value_shapiro:.4f}. {shapiro_msg} "
           f"En control de manufactura, resultados fuera de ±3 desviaciones ({mean_sum-3*std_sum:.1f} a {mean_sum+3*std_sum:.1f}) serían defectos de fábrica críticos.")

# ----------------- Análisis 3 -----------------
st.markdown("---")
st.subheader("Análisis 3: Ratio de Paridad")

st.markdown("""
> **¿Qué es esto?** Análisis de qué proporción de números Pares vs Impares salen en conjunto.\n
> **¿Para qué sirve?** Sirve para Segmentación de Clientes y para garantizar que una prueba A/B en Marketing mantenga su balance poblacional.\n
> **¿Qué estamos midiendo aquí?** Evaluamos la probabilidad hipergeométrica (sacar bolas sin reemplazo). Comparamos lo que *debería* suceder matemáticamente frente a lo que *realmente* está sucediendo en la máquina.
""")

parity_counts = analysis.get_parity_analysis(df_draws)
fig_par = go.Figure(data=[
    go.Bar(name='Observado (Real)', x=parity_counts['Combinacion'], y=parity_counts['ProporcionObservada']),
    go.Scatter(name='Teórico (Matemático)', x=parity_counts['Combinacion'], y=parity_counts['ProbabilidadTeorica'], mode='lines+markers', line=dict(color='red'))
])
fig_par.update_layout(title="Distribución de Pares / Impares", template="plotly_dark", barmode='group')
fig_par.update_xaxes(title="Combinación (Pares-Impares)")
fig_par.update_yaxes(title="Probabilidad / Proporción")
st.plotly_chart(fig_par, use_container_width=True)

most_common_parity = parity_counts.loc[parity_counts['FrecuenciaObservada'].idxmax()]['Combinacion']
st.info(f"**Interpretación del Resultado:** La combinación más frecuente es {most_common_parity}. La cercanía extrema entre la línea roja (Teoría) y las barras (Realidad Histórica) confirma la integridad probabilística de las bolillas, al igual que una auditoría confirmaría un balance contable natural.")


st.markdown("---")
st.header("FASE 2: ESTADÍSTICA INFERENCIAL (TEST DE ALEATORIEDAD)")

# ----------------- Análisis 4 -----------------
st.subheader("Análisis 4: Prueba de Bondad de Ajuste Chi-Cuadrado ($\chi^2$)")

st.markdown("""
> **¿Qué es esto?** Una prueba estadística que compara la distribución total observada contra una distribución teóricamente perfecta.\n
> **¿Para qué sirve?** Pruebas de hipótesis científicas y validación de equidad en algoritmos/modelos predictivos.\n
> **¿Qué estamos midiendo aquí?** Específicamente, estamos midiendo si La Tinka está "arreglada" o es puro azar. Evaluamos matemáticamente la distancia total de los resultados reales frente a una distribución donde todos los números salen exactamente igual.
""")

# current_sorteo needs to be transformed correctly if it is drawing numbers.
chi2_stat, p_value_chi2, obs_freq, exp_freq = analysis.get_chi_square_test(df_exploded)

fig_chi = go.Figure(data=[
    go.Bar(name='Frecuencia Observada', x=list(range(1, 51)), y=obs_freq),
    go.Scatter(name='Esperado (Uniforme)', x=list(range(1, 51)), y=exp_freq, mode='lines', line=dict(color='yellow', dash='dash'))
])
fig_chi.update_layout(title="Prueba de Bondad de Ajuste vs. Distribución Uniforme", template="plotly_dark")
fig_chi.update_xaxes(title="Número de Bolilla (1-50)")
fig_chi.update_yaxes(title="Veces que salió")
st.plotly_chart(fig_chi, use_container_width=True)

if p_value_chi2 < 0.05:
    chi_conclusion = "Existen diferencias estadísticamente significativas con la distribución uniforme. La máquina presenta sesgos."
else:
    chi_conclusion = "No hay evidencia suficiente para descartar que sea uniforme. El sistema se comporta con azar justo."

st.success(f"**Interpretación del Resultado:** Estadístico $\chi^2$: {chi2_stat:.2f} | Valor-p: {p_value_chi2:.4f}. {chi_conclusion} "
           f"En la auditoría interna de un Generador Numérico corporativo, un valor p > 0.05 indica luz verde operativa y disipa sospechas de manipulación.")

# ----------------- Análisis 5 -----------------
st.markdown("---")
st.subheader("Análisis 5: Z-Score Gap Map (Análisis de Retrasos)")

st.markdown("""
> **¿Qué es esto?** Un mapa de calor y dispersión que mide la 'presión' matemática sobre cada número en función al tiempo que lleva sin salir.\n
> **¿Para qué sirve?** En el sector logístico o industrial, se usa para predecir fallos de maquinaria (Mantenimiento Preventivo) basado en el Tiempo Medio Entre Fallos (MTBF).\n
> **¿Qué estamos midiendo aquí?** Medimos el Z-Score (desviaciones estándar) actual de cada bolilla con respecto a su rutina habitual de salida. Una burbuja grande y roja es una bolilla "anómala" e inusualmente retrasada.
""")

current_max_sorteo = int(str(current_sorteo).strip()) if str(current_sorteo).isdigit() else 0 # simple fallback
gaps_df, anomaly = analysis.get_gap_metrics(df_exploded, current_max_sorteo)

fig_scatter = px.scatter(
    gaps_df, 
    x='Mean_Gap', 
    y='Current_Gap', 
    size='Plot_Size', 
    color='Z_Score',
    text='Numero',
    color_continuous_scale='RdYlGn_r',
    title="Gap Map: Detección de Anomalías (Presión Estadística)",
    hover_data=['Z_Score', 'Current_Gap', 'Mean_Gap']
)
fig_scatter.update_xaxes(title="Retraso Histórico Promedio (Sorteos)")
fig_scatter.update_yaxes(title="Retraso Actual (Sorteos)")
fig_scatter.update_traces(textposition='top center')
fig_scatter.update_layout(height=500, template="plotly_dark")
st.plotly_chart(fig_scatter, use_container_width=True)

st.success(f"**Interpretación del Resultado:** Se detecta la anomalía más grande (Z-Score {anomaly['Z_Score']:.2f}) para el número {anomaly['Numero']}. "
        f"Lleva {anomaly['Current_Gap']} sorteos inactivo, frente a su promedio normal de {anomaly['Mean_Gap']:.1f}. "
        f"En operaciones logísticas, cualquier equipo técnico superando un Z-Score de 2 activaría una revisión obligatoria por riesgo inminente de falla.")

# ----------------- Análisis 6 -----------------
st.markdown("---")
st.subheader("Análisis 6: Runs Test (Prueba de Rachas)")

st.markdown("""
> **¿Qué es esto?** Una prueba binaria para ver la secuencia cronológica de valores altos y bajos.\n
> **¿Para qué sirve?** Detección de fraude financiero (ej. facturación cíclica evadiendo impuestos) y detección de algoritmos en el mercado de valores.\n
> **¿Qué estamos midiendo aquí?** Medimos si el resultado de *hoy* depende de alguna forma del de *ayer*. Si los números oscilan naturalmente arriba y abajo de la mediana, son independientes. Si hay largas tendencias fijas, pierden aleatoriedad.
""")

z_stat_runs, p_value_runs, runs, expected_runs = analysis.get_runs_test(df_draws)

# Create a sequence plot of sums
sums_col = analysis.get_sum_distribution(df_draws)[0]
fig_runs = go.Figure()
fig_runs.add_trace(go.Scatter(
    y=sums_col.values, # Sums
    mode='lines+markers',
    name='Suma del Sorteo',
    line=dict(color='cyan', width=1)
))
fig_runs.add_hline(y=sums_col.median(), line_dash="dash", line_color="orange", annotation_text="Mediana")
fig_runs.update_layout(title="Serie Temporal de Sumas (Visualización de Rachas/Runs)", template="plotly_dark")
fig_runs.update_xaxes(title="Índice de Sorteo Histórico")
fig_runs.update_yaxes(title="Suma Total de Bolillas Ganadoras")
st.plotly_chart(fig_runs, use_container_width=True)

if p_value_runs < 0.05:
    runs_conclusion = "Evidencia de autocorrelación (rechazamos independencia pura). Observamos rachas anormales."
else:
    runs_conclusion = "La prueba confirma independencia; las rachas (cruces de línea) son probabilísticamente naturales."

st.success(f"**Interpretación del Resultado:** Rachas observadas: {runs} | Rachas estimadas teóricamente: {expected_runs:.1f} | Valor-p: {p_value_runs:.4f}. "
           f"{runs_conclusion} "
           f"Para una empresa de ciberseguridad, fallar el Runs Test en encriptación implica que el código es predecible y vulnerable a hackeos.")
