---

# 📊 Tinka-Analytics: Decoding Stochastic Patterns 🎰

---

## 📋 Descripción del Proyecto

Este proyecto realiza un **Análisis Exploratorio de Datos (EDA)** y un estudio probabilístico avanzado de "La Tinka" (Perú). El objetivo es desmitificar el azar mediante la identificación de patrones de variabilidad, ciclos de latencia y anomalías estadísticas en los sorteos históricos (específicamente en la era moderna de 6/50 bolillas).

> **Nota:** Este es un proyecto de investigación estadística y formación en Data Science. No garantiza ganancias, pero optimiza la exposición al riesgo mediante la matemática. 🧠

---

<img width="1366" height="768" alt="1" src="https://github.com/user-attachments/assets/6eaed7f1-1271-4854-9503-2b944ea9091e" />

---
## 🚀 Características Principales

### 1. 🔍 Análisis de Gaps (Latencia)

Cálculo del intervalo de sorteos que tarda cada número en reaparecer. Identificamos:

* **Hot Numbers:** Números con ciclos de aparición cortos.
* **Overdue Numbers:** Números con un *Z-Score* de retraso elevado, indicando una "presión" estadística por reaparecer.

### 2. 🔗 Matrices de Co-ocurrencia (Clustering)

Utilizamos técnicas de **Market Basket Analysis** para encontrar "bolillas amigas".

* ¿Qué números tienden a salir juntos con un ?
* Identificación de pares y ternas con alta frecuencia histórica.

### 3. 📉 Cadenas de Markov y Entropía

* **Entropía de Shannon:** Medimos el grado de aleatoriedad del sistema para detectar "bolsones" de baja entropía.
* **Transición de Estados:** Análisis de si un sorteo de suma "Baja" suele ser seguido por uno de suma "Alta" (Reversión a la media).

### 4. 🎲 Simulación de Monte Carlo

Ejecución de **10,000 simulaciones** para validar estrategias de "Jugadas Múltiples" (9-12 números) y determinar el punto de saturación del Retorno de Inversión (ROI).

---

## 🛠️ Tech Stack

* **Lenguaje:** Python 3.x 🐍
* **Librerías:**
* `Pandas`: Manipulación y limpieza de datos.
* `NumPy` & `SciPy`: Cálculos estadísticos y probabilísticos.
* `Matplotlib` & `Seaborn`: Visualización avanzada (Heatmaps, Scatter plots).



---

## 📐 Fundamentos Matemáticos

El proyecto se basa en la fórmula de combinatoria para el cálculo de probabilidades hipergeométricas:

Y la medición de la incertidumbre mediante la Entropía de Shannon:

---

## 📈 Visualizaciones Impactantes

El script genera automáticamente:

1. **Heatmap de Frecuencia Anual:** Para detectar cambios de comportamiento en la máquina de sorteo.
2. **Scatter Plot de Presión:** Relación entre el ciclo medio y el retraso actual.
3. **Matriz de Co-ocurrencia:** Red de relaciones entre números.

---

## 💻 Cómo Ejecutarlo

1. Clona el repositorio:
```bash
git clone https://github.com/juanchocce/Tinka-Analytics-Decoding-Patterns.git

```


2. Sube el archivo `tinka_data.csv` a la raíz del proyecto.
3. Abre el notebook en **Google Colab** o ejecuta localmente:
```bash
python tinka_analysis.py

```



---

## 🙋‍♂️ Autor

**Juan Chocce**

* *Systems Engineer & Data Science Enthusiast* 🚀
* [LinkedIn](https://www.linkedin.com/in/juanchocce/) | [Portfolio Website](https://juanchocce.github.io/)

---

*“En Dios confiamos, todos los demás deben traer datos”. – W. Edwards Deming* 📊

---

