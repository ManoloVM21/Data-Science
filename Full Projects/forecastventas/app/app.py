import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Obtener la ruta raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Ventas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stMetric {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #667eea;
        color: white;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #764ba2;
    }
    </style>
""", unsafe_allow_html=True)

# Función para cargar datos y modelo
@st.cache_resource
def cargar_recursos():
    try:
        modelo_path = BASE_DIR / 'models' / 'modelo_final.joblib'
        data_path = BASE_DIR / 'data' / 'processed' / 'inferencia_df_transformado.csv'
        
        modelo = joblib.load(modelo_path)
        df = pd.read_csv(data_path)
        df['fecha'] = pd.to_datetime(df['fecha'])
        return modelo, df
    except Exception as e:
        st.error(f"❌ Error cargando recursos: {str(e)}")
        return None, None

# Función para recalcular variables de precio
def recalcular_precios(df_prod, descuento_adj, escenario_comp):
    df_sim = df_prod.copy()
    
    # Recalcular precio_venta con el descuento ajustado
    descuento_original = ((df_sim['precio_base'] - df_sim['precio_venta']) / df_sim['precio_base']) * 100
    descuento_nuevo = descuento_original + descuento_adj
    descuento_nuevo = descuento_nuevo.clip(0, 100)
    df_sim['precio_venta'] = df_sim['precio_base'] * (1 - descuento_nuevo / 100)
    df_sim['descuento_porcentaje'] = descuento_nuevo
    
    # Ajustar precios de competencia según escenario
    if escenario_comp == "Competencia -5%":
        ajuste = 0.95
    elif escenario_comp == "Competencia +5%":
        ajuste = 1.05
    else:
        ajuste = 1.0
    
    for col in ['Amazon', 'Decathlon', 'Deporvillage']:
        if col in df_sim.columns:
            df_sim[col] = df_sim[col] * ajuste
    
    # Recalcular precio_competencia (promedio de las 3 columnas)
    comp_cols = [c for c in ['Amazon', 'Decathlon', 'Deporvillage'] if c in df_sim.columns]
    if comp_cols:
        df_sim['precio_competencia'] = df_sim[comp_cols].mean(axis=1)
    
    # Recalcular ratio_precio
    df_sim['ratio_precio'] = df_sim['precio_venta'] / df_sim['precio_competencia']
    
    return df_sim

# Función para hacer predicciones recursivas
def predecir_recursivo(modelo, df_sim):
    df_pred = df_sim.copy().sort_values('fecha').reset_index(drop=True)
    predicciones = []
    
    # Obtener nombres de features que el modelo espera
    feature_cols = [col for col in modelo.feature_names_in_ if col in df_pred.columns]
    
    # Nombres de las columnas de lag
    lag_cols = [f'unidades_vendidas_lag_{i}' for i in range(1, 8)]
    
    for idx in range(len(df_pred)):
        # Preparar features para predicción
        X = df_pred.loc[[idx], feature_cols]
        
        # Predecir
        pred = modelo.predict(X)[0]
        pred = max(0, pred)  # Asegurar que no sea negativa
        predicciones.append(pred)
        
        # Actualizar lags para el siguiente día (si no es el último día)
        if idx < len(df_pred) - 1:
            # Desplazar lags: lag_7←lag_6, lag_6←lag_5, ..., lag_2←lag_1
            for i in range(7, 1, -1):
                lag_actual = f'unidades_vendidas_lag_{i}'
                lag_anterior = f'unidades_vendidas_lag_{i-1}'
                if lag_actual in df_pred.columns and lag_anterior in df_pred.columns:
                    df_pred.loc[idx + 1, lag_actual] = df_pred.loc[idx, lag_anterior]
            
            # lag_1 = predicción actual
            if 'unidades_vendidas_lag_1' in df_pred.columns:
                df_pred.loc[idx + 1, 'unidades_vendidas_lag_1'] = pred
            
            # Actualizar media móvil
            if 'unidades_vendidas_ma7' in df_pred.columns:
                ultimas_7 = predicciones[-7:] if len(predicciones) >= 7 else predicciones
                df_pred.loc[idx + 1, 'unidades_vendidas_ma7'] = np.mean(ultimas_7)
    
    df_pred['unidades_predichas'] = predicciones
    df_pred['ingresos_predichos'] = df_pred['unidades_predichas'] * df_pred['precio_venta']
    
    return df_pred

# Función para crear gráfico de predicción
def crear_grafico_prediccion(df_resultado):
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Configurar estilo de seaborn
    sns.set_style("whitegrid")
    
    # Línea de predicción
    dias = df_resultado['dia_mes'].values
    unidades = df_resultado['unidades_predichas'].values
    
    sns.lineplot(x=dias, y=unidades, marker='o', linewidth=2.5, 
                 color='#667eea', markersize=6, ax=ax)
    
    # Marcar Black Friday (día 28)
    bf_idx = df_resultado[df_resultado['dia_mes'] == 28].index
    if len(bf_idx) > 0:
        bf_unidades = df_resultado.loc[bf_idx[0], 'unidades_predichas']
        ax.axvline(x=28, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.plot(28, bf_unidades, 'ro', markersize=15, zorder=5)
        ax.annotate('🛍️ Black Friday', xy=(28, bf_unidades), 
                   xytext=(28, bf_unidades * 1.15),
                   fontsize=12, fontweight='bold', color='red',
                   ha='center',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    ax.set_xlabel('Día del Mes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
    ax.set_title('Predicción de Ventas Diarias - Noviembre 2025', 
                fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 31)
    
    plt.tight_layout()
    return fig

# Cargar recursos
modelo, df_inferencia = cargar_recursos()

if modelo is None or df_inferencia is None:
    st.stop()

# Obtener lista de productos
productos = sorted(df_inferencia['nombre'].unique())

# ============= SIDEBAR =============
st.sidebar.title("🎯 Controles de Simulación")
st.sidebar.markdown("---")

# Selector de producto
producto_seleccionado = st.sidebar.selectbox(
    "📦 Seleccionar Producto:",
    productos,
    index=0
)

# Slider de descuento
descuento_ajuste = st.sidebar.slider(
    "💰 Ajuste de Descuento:",
    min_value=-50,
    max_value=50,
    value=0,
    step=5,
    format="%d%%",
    help="Ajusta el descuento sobre el descuento actual del producto"
)

# Selector de escenario de competencia
st.sidebar.markdown("### 🏪 Escenario de Competencia:")
escenario_competencia = st.sidebar.radio(
    "",
    ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
    index=0,
    help="Simula cambios en los precios de la competencia"
)

st.sidebar.markdown("---")

# Botón de simulación
simular = st.sidebar.button("🚀 Simular Ventas", type="primary")

# ============= ZONA PRINCIPAL =============
st.title("📊 Dashboard de Simulación de Ventas")
st.markdown(f"### 🛍️ Producto: **{producto_seleccionado}** | 📅 Noviembre 2025")
st.markdown("---")

if simular:
    with st.spinner("⏳ Ejecutando predicciones recursivas..."):
        # Filtrar datos del producto
        df_producto = df_inferencia[df_inferencia['nombre'] == producto_seleccionado].copy()
        
        if len(df_producto) == 0:
            st.error("❌ No hay datos para este producto")
            st.stop()
        
        # Recalcular precios según controles
        df_simulado = recalcular_precios(df_producto, descuento_ajuste, escenario_competencia)
        
        # Hacer predicciones recursivas
        df_resultado = predecir_recursivo(modelo, df_simulado)
        
        # Calcular KPIs
        unidades_totales = df_resultado['unidades_predichas'].sum()
        ingresos_totales = df_resultado['ingresos_predichos'].sum()
        precio_promedio = df_resultado['precio_venta'].mean()
        descuento_promedio = df_resultado['descuento_porcentaje'].mean()
        
        # ============= KPIs =============
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📦 Unidades Totales",
                value=f"{int(unidades_totales):,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="💰 Ingresos Proyectados",
                value=f"€{ingresos_totales:,.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="🏷️ Precio Promedio",
                value=f"€{precio_promedio:.2f}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="🎯 Descuento Promedio",
                value=f"{descuento_promedio:.1f}%",
                delta=None
            )
        
        st.markdown("---")
        
        # ============= GRÁFICO =============
        st.subheader("📈 Predicción de Ventas Diarias")
        fig = crear_grafico_prediccion(df_resultado)
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # ============= TABLA DETALLADA =============
        st.subheader("📋 Detalle Diario de Predicciones")
        
        # Preparar datos para la tabla
        df_tabla = df_resultado[['fecha', 'nombre_dia', 'precio_venta', 'precio_competencia', 
                                'descuento_porcentaje', 'unidades_predichas', 'ingresos_predichos']].copy()
        df_tabla['fecha'] = df_tabla['fecha'].dt.strftime('%d/%m/%Y')
        df_tabla.columns = ['Fecha', 'Día', 'Precio Venta (€)', 'Precio Comp. (€)', 
                           'Descuento (%)', 'Unidades', 'Ingresos (€)']
        
        # Formatear números
        df_tabla['Precio Venta (€)'] = df_tabla['Precio Venta (€)'].apply(lambda x: f"€{x:.2f}")
        df_tabla['Precio Comp. (€)'] = df_tabla['Precio Comp. (€)'].apply(lambda x: f"€{x:.2f}")
        df_tabla['Descuento (%)'] = df_tabla['Descuento (%)'].apply(lambda x: f"{x:.1f}%")
        df_tabla['Unidades'] = df_tabla['Unidades'].apply(lambda x: f"{int(x):,}")
        df_tabla['Ingresos (€)'] = df_tabla['Ingresos (€)'].apply(lambda x: f"€{x:,.2f}")
        
        # Agregar emoji para Black Friday
        df_tabla.loc[27, 'Día'] = '🛍️ ' + df_tabla.loc[27, 'Día'] + ' (Black Friday)'
        
        st.dataframe(df_tabla, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # ============= COMPARATIVA DE ESCENARIOS =============
        st.subheader("📊 Comparativa de Escenarios de Competencia")
        st.markdown("*Manteniendo el descuento seleccionado, variando solo precios de competencia*")
        
        # Calcular los 3 escenarios
        escenarios = ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
        resultados_escenarios = {}
        
        for escenario in escenarios:
            df_esc = recalcular_precios(df_producto, descuento_ajuste, escenario)
            df_pred_esc = predecir_recursivo(modelo, df_esc)
            resultados_escenarios[escenario] = {
                'unidades': df_pred_esc['unidades_predichas'].sum(),
                'ingresos': df_pred_esc['ingresos_predichos'].sum()
            }
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 📉 Competencia -5%")
            st.metric("Unidades", f"{int(resultados_escenarios['Competencia -5%']['unidades']):,}")
            st.metric("Ingresos", f"€{resultados_escenarios['Competencia -5%']['ingresos']:,.2f}")
        
        with col2:
            st.markdown("### ➡️ Actual (0%)")
            st.metric("Unidades", f"{int(resultados_escenarios['Actual (0%)']['unidades']):,}")
            st.metric("Ingresos", f"€{resultados_escenarios['Actual (0%)']['ingresos']:,.2f}")
        
        with col3:
            st.markdown("### 📈 Competencia +5%")
            st.metric("Unidades", f"{int(resultados_escenarios['Competencia +5%']['unidades']):,}")
            st.metric("Ingresos", f"€{resultados_escenarios['Competencia +5%']['ingresos']:,.2f}")
        
        st.success("✅ Simulación completada exitosamente")

else:
    # Mostrar mensaje inicial
    st.info("👈 Configura los parámetros en el panel lateral y presiona **'Simular Ventas'** para ver las predicciones")
    
    # Mostrar información del dataset
    st.markdown("### 📊 Información del Dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total de Productos", len(productos))
        st.metric("Días a Predecir", "30 días")
    with col2:
        st.metric("Mes de Predicción", "Noviembre 2025")
        st.metric("Modelo", "HistGradientBoostingRegressor")