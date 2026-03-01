import streamlit as st

st.set_page_config(page_title="Informe Ejecutivo", page_icon="📝", layout="wide")

# Load CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("📝 Informe de Conclusiones ")
st.markdown("### Resumen Ejecutivo para Gerencia y C-Level")

st.markdown("""
Este portafolio demostró la aplicación práctica de Matemática Estadística, Machine Learning y Simulaciones Cuantitativas a un entorno ruidoso (La Lotería). 

Las siguientes conclusiones sintetizan la auditoría realizada y su paralelismo con escenarios de negocios reales:
""")

# ----------------- SECCIÓN 1 -----------------
st.header("Sección 1: Auditoría de Integridad")
st.success("**Pregunta Clave:** ¿Es confiable el sorteo de La Tinka o presenta sesgos manipulados?")
st.markdown("""
**Respuesta:** Sí, es confiable. El análisis inferencial (Fase 2) dictamina a favor de un sistema de Azar Puro.

*   **Evidencia:** La prueba *Chi-Cuadrado de Bondad de Ajuste* rechazó la existencia de desbalances (los números salen de manera equitativa a largo plazo).
*   **Independencia:** El *Runs Test (Prueba de Rachas)* confirmó que los resultados no están correlacionados (la máquina no tiene memoria).
*   **Paralelismo de Negocio:** Una auditoría externa a un Generador de Números Aleatorios corporativo (RNG) o un algoritmo de sorteos bancarios utilizaría exactamente este *framework* para garantizar transparencia frente a reguladores.
""")

st.markdown("---")
# ----------------- SECCIÓN 2 -----------------
st.header("Sección 2: Capacidad Predictiva")
st.warning("**Pregunta Clave:** ¿Puede la Inteligencia Artificial predecir el esquema y garantizar victorias?")
st.markdown("""
**Respuesta:** No. El algoritmo predictivo revela el límite de la IA frente al caos entrópico.

*   **Evidencia XGBoost:** El modelo presentó una métrica AUC cercana al 0.50 (equivalente a lanzar una moneda), demostrando que variables temporales o retrasos (*Z-Scores*) no otorgan un *Edge* predictivo en loterías.
*   **Evidencia Deep Learning (LSTM):** Se evidenció el *Overfitting* al memorizar datos de entrenamiento fallando en la fase de validación futura.
*   **Paralelismo de Negocio:** La Ciencia de Datos seria exige **honestidad brutal**. Un portafolio que prometa predecir la lotería ignora el Análisis de Ruido Matemático. Sin embargo, estas mismas arquitecturas predecirían con alta precisión el Riesgo Crediticio dado que allí *sí* existen variables correlacionadas.
""")

st.markdown("---")
# ----------------- SECCIÓN 3 -----------------
st.header("Sección 3: Gestión de Riesgo y Operatividad")
st.error("**Pregunta Clave:** Estratégicamente, ¿cuál es el comportamiento financiero óptimo si se decide participar?")
st.markdown("""
**Respuesta:** La esperanza matemática es inherentemente negativa, dictando una abstención de inversión (*Kelly = 0%*).

*   **Evidencia (Criterio de Kelly & Monte Carlo):** Las simulaciones vectorizadas probaron una erosión sistemática de capital (Risk of Ruin extremo) en largas ventanas de tiempo.
*   **El Único Camino:** Si, por razones externas, se *debe* participar, se comprobó que las métricas y los promedios ('Números Calientes') son Sesgos Cognitivos estériles probados por los *Tests A/B*.
*   **Paralelismo de Negocio:** En Trading o Venture Capital, conocer esta ecuación (Kelly) salva carteras de quiebras silenciosas, protegiendo a la empresa cuando el algoritmo no posee ventajas numéricas reales en el nicho.
""")

st.markdown("---")
st.info("""
### Stack Tecnológico Desplegado en el Proyecto
*   **Backend Analítico:** `Python`, `SciPy` (Estadística de Bajo Nivel), `NumPy` (Vectorización para rapidez extrema computacional).
*   **Machine Learning:** `XGBoost`, `Scikit-Learn` (Métricas de Confusión y AUC ROC).
*   **Presentación Interactiva:** `Streamlit`, `Plotly Express` & `Graph Objects` (Visualizaciones Dark Mode Inmersivas).
*   **API Demo (Producción Simulada):** `FastAPI` y `Uvicorn` (Endpoint parametrizado con `Pydantic`).
""")
