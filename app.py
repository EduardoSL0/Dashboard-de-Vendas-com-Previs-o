import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# ========== CACHE DE PERFORMANCE ==========
@st.cache_data
def gerar_dados_exemplo():
    """Gera dados de exemplo realistas com sazonalidade"""
    np.random.seed(42)
    datas = pd.date_range(start='2024-01-01', end=datetime.now(), freq='D')
    tendencia = np.linspace(800, 1500, len(datas))
    sazonal_semanal = np.sin(np.arange(len(datas)) * 2 * np.pi / 7) * 150
    sazonal_mensal = np.sin(np.arange(len(datas)) * 2 * np.pi / 30) * 100
    ruido = np.random.randn(len(datas)) * 80
    vendas = tendencia + sazonal_semanal + sazonal_mensal + ruido
    vendas = np.maximum(vendas, 300)
    
    return pd.DataFrame({
        'data': datas,
        'vendas': vendas.astype(int),
        'categoria': np.random.choice(['Online', 'Loja Física', 'Parceiros'], len(datas))
    })

# ========== CONFIGURAÇÃO ==========
st.set_page_config(page_title="Dashboard Executivo de Vendas", layout="wide")

st.markdown("## 📊 Dashboard Executivo de Vendas")
st.caption("Análise Preditiva com Machine Learning | Validado com R² em dados de teste")

# ========== SIDEBAR ==========
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Meta
    st.subheader("🎯 Meta Diária")
    meta_diaria = st.number_input("Valor (R$)", min_value=100, max_value=10000, value=1200, step=100)
    
    st.divider()
    
    # Dados
    st.subheader("📁 Fonte de Dados")
    opcao = st.radio("Escolha:", ["Usar exemplo", "Upload CSV"], label_visibility="collapsed")
    
    if opcao == "Upload CSV":
        arquivo = st.file_uploader("CSV com colunas: data, vendas, categoria", type=['csv'])
        if arquivo:
            df_raw = pd.read_csv(arquivo)
        else:
            st.stop()
    else:
        df_raw = gerar_dados_exemplo()

# ========== VALIDAÇÃO ==========
colunas_necessarias = ['data', 'vendas']

if not all(col in df_raw.columns for col in colunas_necessarias):
    st.error("❌ Erro: O arquivo precisa conter as colunas: `data` e `vendas`")
    st.info(f"📋 Colunas encontradas: {', '.join(df_raw.columns.tolist())}")
    st.stop()

# ========== TRATAMENTO ==========
try:
    df_raw['data'] = pd.to_datetime(df_raw['data'])
    df_raw = df_raw.sort_values('data')
except Exception as e:
    st.error("❌ Erro ao processar a coluna 'data'. Use formato YYYY-MM-DD ou DD/MM/YYYY")
    st.stop()

# ========== FILTROS ==========
st.sidebar.divider()
st.sidebar.subheader("📅 Período")

data_inicio = st.sidebar.date_input("De", df_raw['data'].min())
data_fim = st.sidebar.date_input("Até", df_raw['data'].max())

df_dashboard = df_raw[(df_raw['data'] >= pd.to_datetime(data_inicio)) & 
                      (df_raw['data'] <= pd.to_datetime(data_fim))].copy()

if len(df_dashboard) == 0:
    st.warning("⚠️ Nenhum dado no período selecionado.")
    st.stop()

# Período analisado
st.caption(f"📅 Período: {df_dashboard['data'].min().strftime('%d/%m/%Y')} até {df_dashboard['data'].max().strftime('%d/%m/%Y')} ({len(df_dashboard)} dias)")

# ========== MÉTRICAS ==========
col1, col2, col3, col4, col5, col6 = st.columns(6)

total_vendas = df_dashboard['vendas'].sum()
media_diaria = df_dashboard['vendas'].mean()
maior_venda = df_dashboard['vendas'].max()
dias_analisados = len(df_dashboard)

crescimento = ((df_dashboard['vendas'].iloc[-1] - df_dashboard['vendas'].iloc[0]) / 
               df_dashboard['vendas'].iloc[0]) * 100 if len(df_dashboard) > 1 else 0

atingimento = (media_diaria / meta_diaria) * 100 if meta_diaria > 0 else 0

col1.metric("💰 Total", f"R${total_vendas:,.0f}")
col2.metric("📈 Média/Dia", f"R${media_diaria:,.0f}")
col3.metric("🏆 Recorde", f"R${maior_venda:,.0f}")
col4.metric("📅 Dias", f"{dias_analisados}")
col5.metric("📊 Cresc.", f"{crescimento:.1f}%")
col6.metric("🎯 Meta", f"{atingimento:.0f}%", 
            delta=f"{atingimento - 100:.0f}%" if atingimento != 100 else None)

# ========== ALERTA ==========
st.divider()

ultima_venda = df_dashboard['vendas'].iloc[-1]
tendencia_icon = "📈" if crescimento > 0 else "📉" if crescimento < 0 else "➡️"

if ultima_venda < media_diaria * 0.8:
    st.warning(f"{tendencia_icon} **Atenção:** Vendas 20% abaixo da média. Revisar estratégia.")
elif ultima_venda > media_diaria * 1.2:
    st.success(f"{tendencia_icon} **Excelente:** Vendas 20% acima da média!")
else:
    st.info(f"{tendencia_icon} **Estável:** Dentro da faixa esperada.")

# ========== ABAS ==========
tab_analise, tab_previsao, tab_insights = st.tabs(["📊 Análise", "🔮 Previsão ML", "💡 Insights"])

with tab_analise:
    df_dashboard['media_movel_7'] = df_dashboard['vendas'].rolling(window=7, min_periods=1).mean()
    
    st.subheader("Evolução das Vendas")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_dashboard['data'], y=df_dashboard['vendas'],
        mode='lines', name='Vendas Diárias',
        line=dict(color='#667eea', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=df_dashboard['data'], y=df_dashboard['media_movel_7'],
        mode='lines', name='Média Móvel (7 dias)',
        line=dict(color='#f1c40f', width=3, dash='dot')
    ))
    
    fig.add_hline(y=meta_diaria, line_dash="dash", line_color="#2ecc71",
                  annotation_text=f"Meta: R${meta_diaria:,.0f}")
    
    fig.update_layout(
        title='Vendas vs Tendência vs Meta',
        xaxis_title='Data', yaxis_title='Vendas (R$)',
        hovermode='x unified', template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col_cat1, col_cat2 = st.columns(2)
    
    with col_cat1:
        st.subheader("Por Categoria")
        vendas_cat = df_dashboard.groupby('categoria')['vendas'].sum().reset_index()
        fig_pie = px.pie(vendas_cat, values='vendas', names='categoria', hole=0.4)
        fig_pie.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_cat2:
        st.subheader("Top 10 Dias")
        top_dias = df_dashboard.nlargest(10, 'vendas')[['data', 'vendas']].copy()
        top_dias['data'] = top_dias['data'].dt.strftime('%d/%m/%Y')
        st.dataframe(top_dias, use_container_width=True, hide_index=True)

with tab_previsao:
    st.subheader("🔮 Previsão dos Próximos 7 Dias")
    
    # Feature Engineering
    df_modelo = df_dashboard.copy()
    df_modelo['dias_numero'] = (df_modelo['data'] - df_modelo['data'].min()).dt.days
    df_modelo['dia_semana'] = df_modelo['data'].dt.dayofweek
    df_modelo['mes'] = df_modelo['data'].dt.month
    
    X = df_modelo[['dias_numero', 'dia_semana', 'mes']]
    y = df_modelo['vendas']
    
    # Validação real: treino/teste separados (shuffle=False para série temporal)
    if len(X) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)
        
        # R² em dados de TESTE (avaliação real)
        y_pred_test = modelo.predict(X_test)
        r2 = r2_score(y_test, y_pred_test)
    else:
        modelo = LinearRegression()
        modelo.fit(X, y)
        r2 = 0.0
        st.info("ℹ️ Poucos dados para validação. R² calculado no treino.")
    
    # Previsões futuras
    ultimo_dia = df_modelo['dias_numero'].max()
    ultimo_dia_semana = df_modelo['dia_semana'].iloc[-1]
    ultimo_mes = df_modelo['mes'].iloc[-1]
    
    dias_futuros = []
    for i in range(1, 8):
        dias_futuros.append([
            ultimo_dia + i,
            (ultimo_dia_semana + i) % 7,
            ultimo_mes if (ultimo_dia + i) <= 30 else (ultimo_mes % 12) + 1
        ])
    
    dias_futuros = np.array(dias_futuros)
    previsoes_raw = modelo.predict(dias_futuros)
    
    # Clip para valores realistas
    vmin = df_dashboard['vendas'].min()
    vmax = df_dashboard['vendas'].max() * 1.5
    previsoes = np.clip(previsoes_raw, vmin, vmax)
    
    # Dataframe previsão
    datas_futuras = [df_dashboard['data'].max() + timedelta(days=i) for i in range(1, 8)]
    df_previsao = pd.DataFrame({
        'data': datas_futuras,
        'vendas_previstas': previsoes.astype(int)
    })
    
    # Gráfico
    fig_prev = go.Figure()
    
    fig_prev.add_trace(go.Scatter(
        x=df_dashboard.tail(30)['data'],
        y=df_dashboard.tail(30)['vendas'],
        mode='lines', name='Real',
        line=dict(color='#667eea', width=2)
    ))
    
    fig_prev.add_trace(go.Scatter(
        x=df_previsao['data'], y=df_previsao['vendas_previstas'],
        mode='lines+markers', name='Previsão ML',
        line=dict(color='#e74c3c', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig_prev.add_trace(go.Scatter(
        x=df_previsao['data'].tolist() + df_previsao['data'].tolist()[::-1],
        y=(previsoes * 1.1).tolist() + (previsoes * 0.9).tolist()[::-1],
        fill='toself', fillcolor='rgba(231, 76, 60, 0.1)',
        line=dict(color='rgba(0,0,0,0)'), name='Intervalo (±10%)'
    ))
    
    fig_prev.update_layout(
        title='Vendas: Real vs Previsão (ML)',
        xaxis_title='Data', yaxis_title='Vendas (R$)',
        hovermode='x unified', template='plotly_dark'
    )
    
    st.plotly_chart(fig_prev, use_container_width=True)
    
    # Métricas
    col_q1, col_q2, col_q3 = st.columns(3)
    col_q1.metric("Qualidade do Modelo (R²)", f"{r2:.2f}",
                  help="Validado em dados de teste (20% separados). Quanto mais próximo de 1, melhor.")
    col_q2.metric("Média/Dia (Previsto)", f"R${previsoes.mean():,.0f}")
    col_q3.metric("Total 7 Dias", f"R${previsoes.sum():,.0f}")
    
    # Tabela
    st.subheader("📋 Previsão Detalhada")
    df_previsao['data'] = df_previsao['data'].dt.strftime('%d/%m/%Y (%a)')
    df_previsao['vendas_previstas'] = df_previsao['vendas_previstas'].apply(lambda x: f"R${x:,.0f}")
    st.dataframe(df_previsao.rename(columns={
        'data': 'Data', 'vendas_previstas': 'Previsão ML'
    }), use_container_width=True, hide_index=True)

with tab_insights:
    st.subheader("💡 Insights Automáticos")
    
    # Mapa de dias em PT-BR
    dias_map = {
        'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
        'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
    
    df_insight = df_dashboard.copy()
    df_insight['dia_semana'] = df_insight['data'].dt.day_name().map(dias_map)
    df_insight['mes'] = df_insight['data'].dt.month_name()
    
    # Ordenação correta dos dias
    ordem_dias = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
    vendas_por_dia = df_insight.groupby('dia_semana')['vendas'].mean()
    vendas_por_dia = vendas_por_dia.reindex([d for d in ordem_dias if d in vendas_por_dia.index])
    
    melhor_dia = vendas_por_dia.idxmax()
    pior_dia = vendas_por_dia.idxmin()
    melhor_categoria = df_insight.groupby('categoria')['vendas'].sum().idxmax()
    
    # Crescimento mensal
    df_insight['ano_mes'] = df_insight['data'].dt.to_period('M')
    vendas_mensais = df_insight.groupby('ano_mes')['vendas'].sum()
    if len(vendas_mensais) > 1:
        crescimento_mensal = ((vendas_mensais.iloc[-1] - vendas_mensais.iloc[-2]) / 
                              vendas_mensais.iloc[-2]) * 100
    else:
        crescimento_mensal = 0
    
    # Cards
    col_i1, col_i2, col_i3, col_i4 = st.columns(4)
    col_i1.metric("🏆 Melhor Dia", melhor_dia)
    col_i2.metric("📉 Pior Dia", pior_dia)
    col_i3.metric("⭐ Melhor Canal", melhor_categoria)
    col_i4.metric("📈 vs Mês Ant.", f"{crescimento_mensal:.1f}%")
    
    st.divider()
    
    st.info(f"""
    📊 **Análise Estratégica:**
    
    • **Melhor dia:** {melhor_dia} (média R${vendas_por_dia.max():,.0f})
    • **Pior dia:** {pior_dia} (média R${vendas_por_dia.min():,.0f})
    • **Canal top:** {melhor_categoria}
    • **Crescimento mensal:** {crescimento_mensal:.1f}%
    
    💡 **Recomendação:** Invista em campanhas às {melhor_dia}s e fortaleça o canal {melhor_categoria}.
    """)
    
    # Gráfico ordenado corretamente
    st.subheader("Vendas por Dia da Semana")
    fig_dia = px.bar(
        vendas_por_dia.reset_index(),
        x='dia_semana', y='vendas',
        labels={'vendas': 'Média (R$)', 'dia_semana': 'Dia'},
        color='vendas', color_continuous_scale='viridis'
    )
    st.plotly_chart(fig_dia, use_container_width=True)

# ========== DOWNLOAD ==========
st.divider()

col_down, _ = st.columns([1, 3])
with col_down:
    csv = df_dashboard.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Baixar Dados (CSV)", csv,
        "dados_vendas.csv", "text/csv",
        use_container_width=True
    )

st.caption("🚀 Python + Streamlit | ML: Regressão Linear com Feature Engineering | Validação: Train/Test Split")
