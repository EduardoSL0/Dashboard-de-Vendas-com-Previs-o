# 📊 Dashboard Executivo de Vendas com Previsão ML

Aplicação profissional que analisa vendas históricas e prevê faturamento dos próximos 7 dias usando Machine Learning com validação real em dados de teste.

---

## 🎯 Problema que Resolve

Empresas precisam prever vendas para:
- **Planejar estoque** e evitar ruptura ou excesso
- **Alocar equipe** nos dias de maior demanda
- **Calcular investimento** em marketing e campanhas

Este dashboard transforma dados brutos em **decisões acionáveis** com previsões validadas.

---

## 🚀 Funcionalidades

| Funcionalidade | Descrição |
|:--|:--|
| 📈 **Análise Temporal** | Vendas diárias + média móvel de 7 dias + meta ajustável |
| 🥧 **Análise por Categoria** | Distribuição entre Online, Loja Física e Parceiros |
| 🔮 **Previsão ML** | Projeção para próximos 7 dias com Regressão Linear |
| ✅ **Validação Real** | R² calculado em dados de teste (20% separados, shuffle=False) |
| 🧠 **Feature Engineering** | Tendência + sazonalidade semanal + sazonalidade mensal |
| 💡 **Insights Automáticos** | Melhor/pior dia, canal top, recomendações estratégicas |
| 📅 **Filtros Interativos** | Período customizável + meta diária ajustável |
| 🚨 **Alertas Inteligentes** | Vendas acima/abaixo da média com recomendação |
| 📥 **Exportar Dados** | Download dos dados filtrados em CSV |

---

## 🛠️ Tecnologias

| Camada | Tecnologia |
|:--|:--|
| Linguagem | Python 3.10+ |
| Interface Web | Streamlit |
| Visualização | Plotly |
| Machine Learning | Scikit-learn (Regressão Linear) |
| Manipulação de Dados | Pandas, NumPy |
| Validação de Modelo | Train/Test Split temporal |

---

## 💡 Destaques Técnicos

### 🔧 Feature Engineering
O modelo utiliza 3 variáveis para capturar padrões complexos:

```python
# Tendência temporal (crescimento ao longo do tempo)
df['dias_numero'] = (df['data'] - df['data'].min()).dt.days

# Sazonalidade semanal (segunda vs sexta)
df['dia_semana'] = df['data'].dt.dayofweek

# Sazonalidade mensal (início vs fim do mês)
df['mes'] = df['data'].dt.month
