import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Dashboard de Vendas", layout="wide")

# Título
st.title("📊 Dashboard de Vendas + Previsão")
st.markdown("Análise de vendas e previsão para os próximos 7 dias")

# Sidebar - upload de dados
with st.sidebar:
    st.header("📁 Dados")
    
    # Opção: usar dados de exemplo ou upload
    opcao = st.radio("Fonte dos dados:", ["Usar exemplo", "Upload CSV"])
    
    if opcao == "Upload CSV":
        arquivo = st.file_uploader("Escolha o arquivo", type=['csv'])
        if arquivo:
            df = pd.read_csv(arquivo)
        else:
            st.stop()
    else:
        # Gera dados de exemplo realistas
        np.random.seed(42)
        datas = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
        vendas = 1000 + np.cumsum(np.random.randn(len(datas)) * 50) + np.sin(np.arange(len(datas)) * 2 * np.pi / 30) * 200
        vendas = np.maximum(vendas, 100)  # Mínimo 100
        
        df = pd.DataFrame({
            'data': datas,
            'vendas': vendas.astype(int),
            'categoria': np.random.choice(['Online', 'Loja Física', 'Parceiros'], len(datas))
        })
        df['data'] = pd.to_datetime(df['data'])

# Prepara dados
df['data'] = pd.to_datetime(df['data'])
df = df.sort_values('data')

# Métricas principais
col1, col2, col3, col4 = st.columns(4)

total_vendas = df['vendas'].sum()
media_diaria = df['vendas'].mean()
maior_venda = df['vendas'].max()
dias_analisados = len(df)

col1.metric("💰 Total Vendas", f"R${total_vendas:,.0f}")
col2.metric("📈 Média Diária", f"R${media_diaria:,.0f}")
col3.metric("🏆 Maior Venda", f"R${maior_venda:,.0f}")
col4.metric("📅 Dias Analisados", dias_analisados)

st.divider()

# Abas
tab_analise, tab_previsao = st.tabs(["📊 Análise", "🔮 Previsão"])

with tab_analise:
    # Gráfico de vendas ao longo do tempo
    st.subheader("Evolução das Vendas")
    
    fig = px.line(df, x='data', y='vendas', title='Vendas Diárias',
                  labels={'vendas': 'Vendas (R$)', 'data': 'Data'})
    fig.update_traces(line_color='#667eea')
    st.plotly_chart(fig, use_container_width=True)
    
    # Vendas por categoria
    col_cat1, col_cat2 = st.columns(2)
    
    with col_cat1:
        st.subheader("Por Categoria")
        vendas_cat = df.groupby('categoria')['vendas'].sum().reset_index()
        fig_pie = px.pie(vendas_cat, values='vendas', names='categoria')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_cat2:
        st.subheader("Top 10 Dias")
        top_dias = df.nlargest(10, 'vendas')[['data', 'vendas']]
        top_dias['data'] = top_dias['data'].dt.strftime('%d/%m/%Y')
        st.dataframe(top_dias, use_container_width=True, hide_index=True)

with tab_previsao:
    st.subheader("🔮 Previsão dos Próximos 7 Dias")
    
    # Prepara dados para ML
    df_ml = df.copy()
    df_ml['dias_numero'] = (df_ml['data'] - df_ml['data'].min()).dt.days
    
    X = df_ml[['dias_numero']]
    y = df_ml['vendas']
    
    # Treina modelo simples (Regressão Linear)
    modelo = LinearRegression()
    modelo.fit(X, y)
    
    # Gera previsões para próximos 7 dias
    ultimo_dia = df_ml['dias_numero'].max()
    dias_futuros = np.array([[ultimo_dia + i] for i in range(1, 8)])
    previsoes = modelo.predict(dias_futuros)
    
    # Cria dataframe de previsão
    datas_futuras = [df['data'].max() + timedelta(days=i) for i in range(1, 8)]
    df_previsao = pd.DataFrame({
        'data': datas_futuras,
        'vendas_previstas': previsoes.astype(int),
        'tipo': 'Previsão'
    })
    
    # Junta dados históricos + previsão
    df_historico = df[['data', 'vendas']].copy()
    df_historico['tipo'] = 'Real'
    df_historico = df_historico.rename(columns={'vendas': 'valor'})
    df_previsao_plot = df_previsao.rename(columns={'vendas_previstas': 'valor'})
    
    df_completo = pd.concat([df_historico.tail(30), df_previsao_plot])  # Últimos 30 dias + previsão
    
    # Gráfico
    fig_prev = go.Figure()
    
    # Dados históricos
    fig_prev.add_trace(go.Scatter(
        x=df_historico.tail(30)['data'],
        y=df_historico.tail(30)['valor'],
        mode='lines',
        name='Real',
        line=dict(color='#667eea')
    ))
    
    # Previsão
    fig_prev.add_trace(go.Scatter(
        x=df_previsao['data'],
        y=df_previsao['vendas_previstas'],
        mode='lines+markers',
        name='Previsão',
        line=dict(color='#e74c3c', dash='dash')
    ))
    
    fig_prev.update_layout(
        title='Vendas: Real vs Previsão',
        xaxis_title='Data',
        yaxis_title='Vendas (R$)',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_prev, use_container_width=True)
    
    # Tabela de previsão
    st.subheader("📋 Detalhamento da Previsão")
    df_previsao['data'] = df_previsao['data'].dt.strftime('%d/%m/%Y (%a)')
    df_previsao['vendas_previstas'] = df_previsao['vendas_previstas'].apply(lambda x: f"R${x:,.0f}")
    df_previsao = df_previsao.rename(columns={
        'data': 'Data',
        'vendas_previstas': 'Previsão de Vendas',
        'tipo': 'Tipo'
    })
    st.dataframe(df_previsao[['Data', 'Previsão de Vendas']], use_container_width=True, hide_index=True)
    
    # Insights automáticos
    media_prevista = previsoes.mean()
    tendencia = "📈 Crescente" if previsoes[-1] > previsoes[0] else "📉 Decrescente"
    
    col_ins1, col_ins2 = st.columns(2)
    with col_ins1:
        st.metric("Média Prevista (7 dias)", f"R${media_prevista:,.0f}")
    with col_ins2:
        st.metric("Tendência", tendencia)
    
    st.info(f"💡 **Insight:** Com base nos últimos {dias_analisados} dias, espera-se faturamento de **R${previsoes.sum():,.0f}** na próxima semana.")

st.divider()
st.caption("🚀 Feito com Python + Streamlit | Machine Learning: Regressão Linear")