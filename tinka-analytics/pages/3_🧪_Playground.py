import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from modules import simulation, etl, analysis

st.set_page_config(page_title="Simulación & Negocio", page_icon="🧪", layout="wide")

# Load CSS
with open('assets/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

df_draws, df_exploded = etl.load_data()

st.title("🧪 Laboratorio de Simulación y Valor Esperado (FASE 4)")
st.markdown("Cálculo avanzado de riesgo matemático usando simulaciones de Monte Carlo y dimensionamiento de posición óptima (Criterio de Kelly).")

tab1, tab2, tab3 = st.tabs(["Monte Carlo (Simulador de Mercado)", "Gestión de Riesgo (Kelly)", "A/B Testing (Mitigación de Sesgos)"])

# ----------------- TAB 1: MONTE CARLO -----------------
with tab1:
    st.header("Simulador de Mercado (Vectorización Monte Carlo)")

    st.markdown("""
    > **¿Qué es esto?** Una técnica probabilística que corre miles de escenarios futuros virtuales mediante fuerza bruta matemática.\n
    > **¿Para qué sirve?** El estándar de oro en bancos de inversión (Valor en Riesgo / VaR) y predicción física (aerodinámica).\n
    > **¿Qué estamos midiendo aquí?** Testeamos las consecuencias lógicas a largo plazo de jugar tu combinación frente a la máquina 10,000 veces consecutivas, evidenciando el devastador poder de la Esperanza Matemática.
    """)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Configurar Inversión Simular")
        user_input = st.text_input("Ingresa los números del ticket (separados por comas)", "5, 12, 23, 34, 45, 50")
        n_sims = st.slider("Total Compra de Sorteos Paralelos", min_value=1000, max_value=50000, value=10000, step=1000)
        
        try:
            user_list = [int(x.strip()) for x in user_input.split(',')]
            valid_input = len(user_list) >= 6 and len(user_list) <= 15 and all(1 <= x <= 50 for x in user_list)
        except:
            valid_input = False
            
        if not valid_input:
            st.error("Por favor ingresa entre 6 y 15 números válidos del 1 al 50.")
            
        run_btn = st.button("Ejecutar Simulación (Test Stress)", disabled=not valid_input, type="primary")

    with col2:
        if run_btn and valid_input:
            with st.spinner(f"Estresando algoritmo... {n_sims:,} escenarios paralelos..."):
                hit_counts, roi_percent, unique_matches, total_revenue = simulation.run_simulation(user_list, n_simulations=n_sims)
                
            col_m1, col_m2 = st.columns(2)
            n_played = len(user_list)
            cost_table = {6:5, 7:35, 8:140, 9:420, 10:1050, 11:2310, 12:4620, 13:8580, 14:15015, 15:25025}
            cost_total = n_sims * cost_table.get(n_played, 5)
            
            col_m1.metric("Impacto Capital (Total Gastado)", f"S/ {cost_total:,.2f}")
            col_m2.metric("Liquidación (Prizes Total Bruto)", f"S/ {total_revenue:,.2f}")
            
            st.metric("Constante ROI Final Observado", f"{roi_percent:.2f}%", delta=f"{roi_percent:.2f}%")
            
            hits_df = pd.DataFrame(list(hit_counts.items()), columns=['Aciertos', 'Frecuencia']).sort_values('Aciertos')
            fig_hits = px.bar(hits_df, x='Aciertos', y='Frecuencia', title='Aislación de Volatilidad a Aciertos Positivos (3+)')
            fig_hits.update_xaxes(title="Número de Sorteos Simulados (Categoría de Ganancia)")
            fig_hits.update_yaxes(title="Capital Acumulado (Hits Totales Obtenidos)")
            fig_hits.update_layout(template='plotly_dark')
            st.plotly_chart(fig_hits, use_container_width=True)
            
            st.warning("**Interpretación del Resultado (Esperanza de Negocio Matemática):** Como se demuestra en nanosegundos vectoriales con NumPy computacional, la esperanza corporativa es crónicamente negativa. **Todo modelo de Negocio con un ROI general estático intrínseco de caída libre llevará forzosamente a la Bancarrota (Risk of Ruin de 100%).** Un asesor Financiero descarta la inversión sin dudarlo ni un minuto más.")

# ----------------- TAB 2: KELLY -----------------
with tab2:
    st.header("Cálculo Óptimo: Criterio de Kelly")

    st.markdown("""
    > **¿Qué es esto?** La mítica ecuación de gestión monetaria que Warren Buffett y Ed Thorp aplican en Wall Street. Calcula agresividad contra quiebra dictaminando matemáticamente qué porcentaje fractal del patrimonio arriesgar dada una ventaja (*edge*).\n
    > **¿Para qué sirve?** El freno de emergencia algorítmico natural. Previene que un inversionista pierda todo el 'Bankroll' (Fondo). Maximiza retorno a largo plazo geométrico en mercados de criptomonedas/renta variable.\n
    > **¿Qué estamos midiendo aquí?** Evidenciamos que como la lotería de por sí NO tiene un *Edge* positivo a favor de la persona, Kelly recomendaría $f^* = 0$. Simulamos un escenario de lotería ficticia superior favorable del 55% para demostrar que un porcentaje de Bankroll calculado dinámicamente humilla al monto fijo ciego.
    """)
    
    col_k1, col_k2, col_k3 = st.columns(3)
    p_win = col_k1.slider("Probabilidad de Target Ganador Ficticio (p)", 0.01, 0.99, 0.55)
    payout = col_k2.number_input("Multiplicador Payout Liquidez (b)", 0.1, 5.0, 1.0)
    capital_inicial = col_k3.number_input("Capital Privado Inicial ($)", 100, 10000, 1000)
    
    f_star = simulation.get_kelly_criterion(p_win, payout)
    st.metric("Porcentaje Patrimonial a Arriesgar por Periodo (Kelly f*)", f"{f_star*100:.2f}%")
    
    if st.button("Estresar Evolución de Capital Dinámico (100 Ciclos)"):
        with st.spinner("Integrando función logarítmica..."):
            cap_kelly = simulation.simulate_capital_growth(capital_inicial, 100, p_win, payout, 'kelly')
            cap_fixed = simulation.simulate_capital_growth(capital_inicial, 100, p_win, payout, 'fixed')
            
        fig_cap = go.Figure()
        fig_cap.add_trace(go.Scatter(y=cap_kelly, mode='lines', name='Política Kelly Dinámica Computada'))
        fig_cap.add_trace(go.Scatter(y=cap_fixed, mode='lines', name='Política Retrato Ciego Fija (5% Estático)'))
        fig_cap.update_layout(title='Evolución de Trazado Financiero al Pasar Eventos Positivos Simulados', template='plotly_dark')
        fig_cap.update_xaxes(title="Tiempo (Paso Iterativo Comercial)")
        fig_cap.update_yaxes(title="Crecimiento del Bankroll Corporativo Acumulado")
        st.plotly_chart(fig_cap, use_container_width=True)
        
        st.success("**Interpretación del Resultado (Defensa a Largo Plazo):** Si un CEO o algoritmo cuantitativo predictivo (XGBoost) halla un nicho exitoso, aplicar un multiplicador como Posición Kelly (línea azul dinámica) absorbe choques de varianza minimizando destrucción general, y escala multiplicándose exponencialmente.")

# ----------------- TAB 3: A/B TESTING -----------------
with tab3:
    st.header("Evaluación Científica: Sesgos vs Experimento (A/B Test)")

    st.markdown("""
    > **¿Qué es esto?** Enfrentar dos paradigmas rivales simultáneamente con la misma fuente de muestra limpia aleatoria (Test Ciego de Variables Ocultas).\n
    > **¿Para qué sirve?** El núcleo fundacional empírico de Farmacéuticas, Marketing UX de Tecnológicas (BOTÓN Lleno de Color "A" vs BOTÓN Plástico Gris "B").\n
    > **¿Qué estamos midiendo aquí?** Desafíamos la intuición humana general ("Debo marcar los *Números Calientes* que salen bastante") comprobando que la media arrastra ambos resultados a lo mismo a través de 500 épocas en tiempo real.
    """)
    
    if st.button("Forzar Experimento Doble Ciego Iterativo (500 Muestras)"):
        with st.spinner("Calculando divergencias..."):
            df_freq, _, _ = analysis.get_frequency_analysis(df_exploded)
            ab_results = simulation.run_ab_test_simulator(df_freq, 500)
            
            fig_ab = go.Figure()
            fig_ab.add_trace(go.Scatter(x=ab_results['Draw'], y=ab_results['Hot_Strategy_Hits'].rolling(20).mean(), name='A: Top Frecuencia de Data (Selección Algorítmica)'))
            fig_ab.add_trace(go.Scatter(x=ab_results['Draw'], y=ab_results['Random_Strategy_Hits'].rolling(20).mean(), name='B: Selección Cuántica Neutra Pura'))
            
            fig_ab.update_layout(title='Hits Convergentes Medidos por Iteración Temporal (Media Móvil Exponencial: 20 Sorteos)', template='plotly_dark')
            fig_ab.update_xaxes(title="Punto Cronológico Simulado")
            fig_ab.update_yaxes(title="Aciertos Promedio Promediados")
            st.plotly_chart(fig_ab, use_container_width=True)
            
            hit_mean_hot = ab_results['Hot_Strategy_Hits'].mean()
            hit_mean_rand = ab_results['Random_Strategy_Hits'].mean()
            
            st.metric("Diferencial de Control de Varianza", f"{abs(hit_mean_hot - hit_mean_rand):.4f} micro-aciertos de diferencial máximo")
            
            st.info("**Interpretación del Resultado (Data Driven Anti-Sesgos):** El gráfico empata brutalmente. Esta confluencia matemática refuta frontalmente el instinto humano ilusorio. Validar las pruebas A/B elimina el *Confirmation Bias* natural de equipos de ventas subjetivos, y fuerza un ambiente puramente numérico (Data-Driven Decisions) para enviar la versión ganadora a los cuarteles de Producción.")
