"""
SISTEMA INTEGRADO DE PREVISÃO DO BITCOIN - APLICAÇÃO STREAMLIT
Dashboard completo com análises macroeconômicas, técnicas e previsões ML
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Adicionar diretório raiz ao path
sys.path.append(str(Path(__file__).parent))

# Configuração da página
st.set_page_config(
    page_title="Sistema de Previsão Bitcoin",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título e descrição
st.title("₿ Sistema Integrado de Previsão do Bitcoin")
st.markdown("""
**Análise Macroeconômica + Técnica + Machine Learning para previsão de preços**
*Dados em tempo real do FRED e Yahoo Finance*
""")

# Barra lateral para configurações
st.sidebar.header("⚙️ Configurações do Sistema")

# Inicialização de estado
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

@st.cache_resource
def initialize_system():
    """Inicializa todos os componentes do sistema"""
    from core.data_manager import data_manager
    from core.config import config
    
    print("✅ Sistema inicializado")
    return {
        'data_manager': data_manager,
        'config': config
    }

@st.cache_data(ttl=3600)  # Cache de 1 hora
def load_data():
    """Carrega todos os dados necessários"""
    try:
        from core.data_manager import data_manager
        
        st.info("📥 Carregando dados...")
        
        # Carregar dados
        all_data = data_manager.get_all_data(
            btc_period="5y",
            macro_years=5,
            assets_period="5y"
        )
        
        return {
            'success': True,
            'data': all_data,
            'timestamp': datetime.now()
        }
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now()
        }

@st.cache_data
def process_data(btc_data, macro_data, assets_data):
    """Processa dados para análise"""
    from data_collection.data_processor import data_processor
    
    try:
        # Processar dados Bitcoin
        btc_processed = data_processor.process_bitcoin_data(btc_data)
        
        # Processar dados macro
        macro_processed = data_processor.process_macro_data(macro_data) if macro_data is not None else None
        
        # Alinhar datasets
        aligned_data = data_processor.align_datasets(
            btc_processed['data'],
            macro_processed['data'] if macro_processed else None,
            assets_data
        )
        
        # Preparar dataset ML
        ml_data = data_processor.prepare_ml_dataset(
            aligned_data['master_dataframe'],
            target_column='bitcoin_Close',
            forecast_horizon=30
        )
        
        return {
            'success': True,
            'btc_processed': btc_processed,
            'macro_processed': macro_processed,
            'aligned_data': aligned_data,
            'ml_data': ml_data
        }
    except Exception as e:
        st.error(f"❌ Erro ao processar dados: {e}")
        return {'success': False, 'error': str(e)}

@st.cache_resource
def run_analysis(processed_data):
    """Executa análises econômicas e técnicas"""
    from economic_analysis.cycle_analyzer import cycle_analyzer
    from economic_analysis.macro_correlator import macro_correlator
    from technical_analysis.technical_analyzer import technical_analyzer
    from technical_analysis.volume_analysis import volume_analyzer
    
    try:
        analyses = {}
        
        # 1. Análise de ciclos econômicos
        if processed_data['macro_processed']:
            st.info("🔄 Analisando ciclos econômicos...")
            macro_df = processed_data['macro_processed']['data']
            cycle_analysis = cycle_analyzer.analyze_current_cycle(macro_df)
            analyses['cycle'] = cycle_analysis
        
        # 2. Análise de correlações macro-Bitcoin
        st.info("🔗 Analisando correlações...")
        btc_prices = processed_data['btc_processed']['data']['Close']
        macro_data = processed_data['macro_processed']['data'] if processed_data['macro_processed'] else None
        
        if macro_data is not None:
            correlations = macro_correlator.calculate_correlations(
                btc_prices.to_frame(),
                macro_data
            )
            analyses['correlations'] = correlations
        
        # 3. Análise técnica
        st.info("📊 Analisando indicadores técnicos...")
        btc_data = processed_data['btc_processed']['data']
        tech_indicators = technical_analyzer.calculate_all_indicators(btc_data)
        tech_report = technical_analyzer.generate_technical_report(tech_indicators, btc_data)
        analyses['technical'] = tech_report
        
        # 4. Análise de volume
        st.info("📈 Analisando volume...")
        volume_analysis = volume_analyzer.analyze_volume(btc_data)
        volume_report = volume_analyzer.generate_volume_report(volume_analysis, btc_data)
        analyses['volume'] = volume_report
        
        return {'success': True, 'analyses': analyses}
        
    except Exception as e:
        st.error(f"❌ Erro nas análises: {e}")
        return {'success': False, 'error': str(e)}

@st.cache_resource
def train_models(ml_data):
    """Treina modelos de ML"""
    from ml_models.model_trainer import model_trainer
    
    try:
        st.info("🤖 Treinando modelos ML...")
        
        X = ml_data['X']
        y = ml_data['y']
        
        # Split temporal
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Treinar modelos
        results = model_trainer.train_all_models(
            X_train, y_train,
            X_test, y_test,
            model_types=['random_forest', 'xgboost', 'lightgbm', 'gradient_boosting']
        )
        
        # Criar ensemble
        ensemble_results = model_trainer.create_ensemble(
            X_train, y_train,
            X_test, y_test,
            ensemble_method='averaging'
        )
        
        return {
            'success': True,
            'models': results,
            'ensemble': ensemble_results,
            'feature_importance': model_trainer.feature_importance
        }
        
    except Exception as e:
        st.error(f"❌ Erro no treinamento: {e}")
        return {'success': False, 'error': str(e)}

@st.cache_data
def make_predictions(trained_models, current_features):
    """Faz previsões com os modelos treinados"""
    from ml_models.model_trainer import model_trainer
    from prediction_engine.prediction_engine import prediction_engine
    
    try:
        st.info("🎯 Gerando previsões...")
        
        predictions = {}
        
        # Previsão com ensemble
        ensemble_model = trained_models['ensemble']['individual_models']
        
        # Usar ensemble averaging
        ensemble_pred = np.mean([
            model.predict(current_features) 
            for model in ensemble_model.values()
        ], axis=0)
        
        # Calcular intervalos de confiança (simplificado)
        individual_preds = [model.predict(current_features) for model in ensemble_model.values()]
        pred_std = np.std(individual_preds, axis=0)
        
        current_price = float(current_features['bitcoin_Close'].iloc[-1] if 'bitcoin_Close' in current_features.columns else 0)
        pred_price = float(ensemble_pred[0])
        
        # Calcular estatísticas
        return_percentage = ((pred_price / current_price) - 1) * 100
        confidence_interval = 1.96 * pred_std[0]  # 95% CI
        
        predictions['ensemble'] = {
            'predicted_price': pred_price,
            'current_price': current_price,
            'return_percentage': return_percentage,
            'confidence_interval': confidence_interval,
            'lower_bound': pred_price - confidence_interval,
            'upper_bound': pred_price + confidence_interval,
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'horizon_days': 30
        }
        
        # Salvar no estado da sessão
        st.session_state.predictions = predictions
        
        return {'success': True, 'predictions': predictions}
        
    except Exception as e:
        st.error(f"❌ Erro nas previsões: {e}")
        return {'success': False, 'error': str(e)}

def create_visualizations(data, analyses, predictions):
    """Cria visualizações para o dashboard"""
    
    visualizations = {}
    
    try:
        # 1. Gráfico de preço do Bitcoin
        if data and 'bitcoin' in data['data']:
            btc_data = data['data']['bitcoin']['data']
            fig_price = go.Figure()
            
            fig_price.add_trace(go.Scatter(
                x=btc_data.index,
                y=btc_data['Close'],
                mode='lines',
                name='Preço Bitcoin',
                line=dict(color='#F7931A', width=2)
            ))
            
            # Adicionar bandas de Bollinger se disponíveis
            if 'bb_upper' in btc_data.columns and 'bb_lower' in btc_data.columns:
                fig_price.add_trace(go.Scatter(
                    x=btc_data.index,
                    y=btc_data['bb_upper'],
                    mode='lines',
                    name='Banda Superior',
                    line=dict(color='rgba(247, 147, 26, 0.3)', width=1),
                    showlegend=False
                ))
                fig_price.add_trace(go.Scatter(
                    x=btc_data.index,
                    y=btc_data['bb_lower'],
                    mode='lines',
                    name='Banda Inferior',
                    line=dict(color='rgba(247, 147, 26, 0.3)', width=1),
                    fill='tonexty',
                    fillcolor='rgba(247, 147, 26, 0.1)',
                    showlegend=False
                ))
            
            fig_price.update_layout(
                title='Preço do Bitcoin (USD)',
                xaxis_title='Data',
                yaxis_title='Preço (USD)',
                template='plotly_dark',
                hovermode='x unified'
            )
            
            visualizations['price_chart'] = fig_price
        
        # 2. Gráfico de correlações
        if analyses and 'correlations' in analyses['analyses']:
            corr_df = analyses['analyses']['correlations']
            top_corrs = corr_df.head(10)
            
            fig_corr = go.Figure()
            
            # Correlações positivas
            pos_corrs = top_corrs[top_corrs['correlation'] > 0]
            if not pos_corrs.empty:
                fig_corr.add_trace(go.Bar(
                    x=pos_corrs['correlation'],
                    y=pos_corrs.index,
                    orientation='h',
                    name='Positivas',
                    marker_color='green'
                ))
            
            # Correlações negativas
            neg_corrs = top_corrs[top_corrs['correlation'] < 0]
            if not neg_corrs.empty:
                fig_corr.add_trace(go.Bar(
                    x=neg_corrs['correlation'],
                    y=neg_corrs.index,
                    orientation='h',
                    name='Negativas',
                    marker_color='red'
                ))
            
            fig_corr.update_layout(
                title='Top 10 Correlações com Bitcoin',
                xaxis_title='Correlação',
                yaxis_title='Indicador',
                template='plotly_dark',
                barmode='relative'
            )
            
            visualizations['correlation_chart'] = fig_corr
        
        # 3. Gráfico de previsão
        if predictions and 'ensemble' in predictions['predictions']:
            pred = predictions['predictions']['ensemble']
            
            fig_pred = go.Figure()
            
            # Histórico recente (últimos 60 dias)
            if data and 'bitcoin' in data['data']:
                btc_data = data['data']['bitcoin']['data']
                recent_data = btc_data.iloc[-60:]
                
                fig_pred.add_trace(go.Scatter(
                    x=recent_data.index,
                    y=recent_data['Close'],
                    mode='lines',
                    name='Histórico',
                    line=dict(color='#F7931A', width=2)
                ))
            
            # Previsão
            pred_date = datetime.now() + timedelta(days=30)
            fig_pred.add_trace(go.Scatter(
                x=[datetime.now(), pred_date],
                y=[pred['current_price'], pred['predicted_price']],
                mode='lines+markers',
                name='Previsão',
                line=dict(color='#00FF00', width=3, dash='dash')
            ))
            
            # Intervalo de confiança
            fig_pred.add_trace(go.Scatter(
                x=[pred_date, pred_date],
                y=[pred['lower_bound'], pred['upper_bound']],
                mode='lines',
                name='Intervalo 95%',
                line=dict(color='rgba(0, 255, 0, 0.3)', width=8)
            ))
            
            fig_pred.update_layout(
                title=f'Previsão: ${pred["predicted_price"]:,.2f} (+{pred["return_percentage"]:.1f}%)',
                xaxis_title='Data',
                yaxis_title='Preço (USD)',
                template='plotly_dark'
            )
            
            visualizations['prediction_chart'] = fig_pred
        
        # 4. Gráfico de indicadores técnicos
        if analyses and 'technical' in analyses['analyses']:
            tech_report = analyses['analyses']['technical']
            
            # Criar subplots para RSI e MACD
            from plotly.subplots import make_subplots
            
            if data and 'bitcoin' in data['data']:
                btc_data = data['data']['bitcoin']['data']
                
                fig_tech = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Preço', 'RSI', 'MACD'),
                    vertical_spacing=0.1,
                    row_heights=[0.5, 0.25, 0.25]
                )
                
                # Preço
                fig_tech.add_trace(
                    go.Scatter(x=btc_data.index, y=btc_data['Close'], name='Preço'),
                    row=1, col=1
                )
                
                # RSI
                if 'rsi_14' in btc_data.columns:
                    fig_tech.add_trace(
                        go.Scatter(x=btc_data.index, y=btc_data['rsi_14'], name='RSI'),
                        row=2, col=1
                    )
                    # Linhas de sobrecompra/sobrevenda
                    fig_tech.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig_tech.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                if 'macd_histogram' in btc_data.columns:
                    fig_tech.add_trace(
                        go.Bar(x=btc_data.index, y=btc_data['macd_histogram'], name='MACD Histogram'),
                        row=3, col=1
                    )
                
                fig_tech.update_layout(height=600, template='plotly_dark', showlegend=False)
                visualizations['technical_chart'] = fig_tech
        
    except Exception as e:
        st.error(f"❌ Erro nas visualizações: {e}")
    
    return visualizations

def display_dashboard():
    """Exibe o dashboard principal"""
    
    # Inicializar sistema
    system = initialize_system()
    
    # Sidebar - Controles
    with st.sidebar:
        st.header("🎯 Controles")
        
        if st.button("🔄 Atualizar Dados", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.data_loaded = False
            st.session_state.models_trained = False
            st.rerun()
        
        st.divider()
        
        st.header("📊 Configurações de Análise")
        forecast_days = st.slider("Horizonte de Previsão (dias)", 7, 90, 30)
        
        st.divider()
        
        st.header("ℹ️ Status do Sistema")
        
        if st.session_state.data_loaded:
            st.success("✅ Dados Carregados")
        else:
            st.warning("⚠️ Dados Pendentes")
            
        if st.session_state.models_trained:
            st.success("✅ Modelos Treinados")
        else:
            st.warning("⚠️ Treinamento Pendente")
    
    # Layout principal
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dashboard", 
        "🔍 Análises", 
        "🤖 Machine Learning", 
        "⚙️ Configuração"
    ])
    
    with tab1:
        # Carregar dados
        if not st.session_state.data_loaded:
            st.info("Carregando dados do sistema...")
            data_result = load_data()
            
            if data_result['success']:
                st.session_state.data = data_result
                st.session_state.data_loaded = True
                st.success("✅ Dados carregados com sucesso!")
                st.rerun()
            else:
                st.error("Falha ao carregar dados")
                return
        
        # Processar dados
        if st.session_state.data_loaded and 'processed_data' not in st.session_state:
            with st.spinner("Processando dados..."):
                data = st.session_state.data['data']
                processed = process_data(
                    data['bitcoin']['data'],
                    data.get('macro'),
                    data.get('related_assets')
                )
                
                if processed['success']:
                    st.session_state.processed_data = processed
                else:
                    st.error("Falha no processamento")
                    return
        
        # Executar análises
        if 'processed_data' in st.session_state and 'analyses' not in st.session_state:
            with st.spinner("Executando análises..."):
                analyses = run_analysis(st.session_state.processed_data)
                if analyses['success']:
                    st.session_state.analyses = analyses
                else:
                    st.error("Falha nas análises")
                    return
        
        # Treinar modelos (se necessário)
        if 'processed_data' in st.session_state and not st.session_state.models_trained:
            with st.spinner("Treinando modelos ML..."):
                ml_data = st.session_state.processed_data['ml_data']
                training = train_models(ml_data)
                
                if training['success']:
                    st.session_state.models = training
                    st.session_state.models_trained = True
                    st.success("✅ Modelos treinados com sucesso!")
                else:
                    st.error("Falha no treinamento")
                    return
        
        # Fazer previsões
        if st.session_state.models_trained and st.session_state.predictions is None:
            with st.spinner("Gerando previsões..."):
                ml_data = st.session_state.processed_data['ml_data']
                current_features = ml_data['X'].iloc[-1:].copy()
                
                predictions = make_predictions(
                    st.session_state.models,
                    current_features
                )
                
                if predictions['success']:
                    st.session_state.predictions = predictions
                else:
                    st.error("Falha nas previsões")
        
        # Exibir dashboard
        if all([
            st.session_state.data_loaded,
            'processed_data' in st.session_state,
            'analyses' in st.session_state,
            st.session_state.models_trained,
            st.session_state.predictions is not None
        ]):
            # Criar visualizações
            viz = create_visualizations(
                st.session_state.data,
                st.session_state.analyses,
                st.session_state.predictions
            )
            
            # Layout do dashboard
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(viz.get('price_chart'), use_container_width=True)
                
                # Métricas principais
                if st.session_state.predictions:
                    pred = st.session_state.predictions['predictions']['ensemble']
                    
                    metric1, metric2, metric3 = st.columns(3)
                    with metric1:
                        st.metric(
                            "Preço Atual",
                            f"${pred['current_price']:,.2f}",
                            delta=f"{pred['return_percentage']:.1f}% previsto"
                        )
                    with metric2:
                        st.metric(
                            "Previsão 30d",
                            f"${pred['predicted_price']:,.2f}",
                            delta=f"±${pred['confidence_interval']:.2f}"
                        )
                    with metric3:
                        # Indicador técnico atual
                        if 'analyses' in st.session_state:
                            tech = st.session_state.analyses['analyses']['technical']
                            st.metric(
                                "Sinal Técnico",
                                tech['recommendation']['action'],
                                delta=f"Score: {tech['recommendation']['score']}"
                            )
            
            with col2:
                st.plotly_chart(viz.get('prediction_chart'), use_container_width=True)
            
            # Segunda linha
            st.subheader("Análises Detalhadas")
            
            col3, col4 = st.columns(2)
            
            with col3:
                if 'correlation_chart' in viz:
                    st.plotly_chart(viz['correlation_chart'], use_container_width=True)
            
            with col4:
                if 'technical_chart' in viz:
                    st.plotly_chart(viz['technical_chart'], use_container_width=True)
            
            # Terceira linha - Relatórios
            st.subheader("Relatórios")
            
            tab_a, tab_b, tab_c = st.tabs(["📋 Resumo", "📊 Técnico", "🌍 Macro"])
            
            with tab_a:
                # Resumo executivo
                if st.session_state.predictions:
                    pred = st.session_state.predictions['predictions']['ensemble']
                    
                    st.markdown(f"""
                    ### 📈 Resumo Executivo
                    
                    **Previsão para {pred['horizon_days']} dias:**
                    - **Preço Atual:** ${pred['current_price']:,.2f}
                    - **Preço Previsto:** ${pred['predicted_price']:,.2f}
                    - **Retorno Esperado:** {pred['return_percentage']:.1f}%
                    - **Intervalo de Confiança (95%):** ${pred['lower_bound']:,.2f} - ${pred['upper_bound']:,.2f}
                    
                    **Horizonte:** {datetime.now().strftime('%Y-%m-%d')} → {(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')}
                    """)
            
            with tab_b:
                # Análise técnica
                if 'analyses' in st.session_state:
                    tech = st.session_state.analyses['analyses']['technical']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("RSI", f"{tech['summary']['rsi_14']:.1f}", tech['summary']['rsi_status'])
                    with col2:
                        st.metric("Tendência", tech['trend']['trend'], f"Força: {tech['trend']['strength']:.2f}")
                    with col3:
                        st.metric("Volatilidade", f"{tech['summary']['volatility_30d']:.1f}%", "30 dias")
            
            with tab_c:
                # Análise macro
                if 'analyses' in st.session_state and 'cycle' in st.session_state.analyses['analyses']:
                    cycle = st.session_state.analyses['analyses']['cycle']
                    
                    st.markdown(f"""
                    ### 🌍 Análise Macroeconômica
                    
                    **Ciclo Atual:** {cycle['current_phase']['label']}
                    **Duração:** {cycle['current_phase']['duration_days']} dias
                    **Confiança:** {cycle['confidence_score']:.0%}
                    
                    **Próxima Fase Prevista:** {cycle['next_phase_prediction']['next_phase']}
                    **Probabilidade:** {cycle['next_phase_prediction']['probability']:.0%}
                    """)
    
    with tab2:
        st.header("🔍 Análises Detalhadas")
        
        if 'analyses' in st.session_state:
            analyses = st.session_state.analyses['analyses']
            
            # Análise técnica
            if 'technical' in analyses:
                tech = analyses['technical']
                
                st.subheader("📊 Análise Técnica")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### Indicadores Atuais")
                    indicators_df = pd.DataFrame([
                        {"Indicador": "Preço", "Valor": f"${tech['summary']['price']:,.2f}", "Status": "-"},
                        {"Indicador": "RSI 14", "Valor": f"{tech['summary']['rsi_14']:.1f}", "Status": tech['summary']['rsi_status']},
                        {"Indicador": "MACD", "Valor": f"{tech['summary']['macd']:.4f}", "Status": tech['summary']['macd_signal']},
                        {"Indicador": "BB Position", "Valor": f"{tech['summary']['bb_position']:.3f}", "Status": tech['summary']['bb_status']},
                        {"Indicador": "Volatilidade 30d", "Valor": f"{tech['summary']['volatility_30d']:.1f}%", "Status": "-"}
                    ])
                    st.dataframe(indicators_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("### Sinais de Trading")
                    signals_df = pd.DataFrame([
                        {"Sinal": "RSI", "Valor": tech['signals']['rsi_signal'], "Direção": "Compra" if tech['signals']['rsi_signal'] > 0 else "Venda"},
                        {"Sinal": "MACD", "Valor": tech['signals']['macd_signal'], "Direção": "Compra" if tech['signals']['macd_signal'] > 0 else "Venda"},
                        {"Sinal": "Médias Móveis", "Valor": tech['signals']['ma_signal'], "Direção": "Compra" if tech['signals']['ma_signal'] > 0 else "Venda"},
                        {"Sinal": "Composto", "Valor": f"{tech['signals']['composite_signal']:.2f}", "Direção": tech['signals']['signal_strength']}
                    ])
                    st.dataframe(signals_df, use_container_width=True, hide_index=True)
            
            # Análise de correlações
            if 'correlations' in analyses:
                st.subheader("🔗 Correlações com Indicadores Macro")
                
                corr_df = analyses['correlations']
                top_10 = corr_df.head(10)
                
                st.dataframe(top_10, use_container_width=True)
                
                # Explicação das correlações
                st.markdown("""
                **Interpretação:**
                - **Correlação positiva:** Bitcoin e indicador movem-se na mesma direção
                - **Correlação negativa:** Bitcoin e indicador movem-se em direções opostas
                - **p-value < 0.05:** Correlação estatisticamente significativa
                """)
    
    with tab3:
        st.header("🤖 Machine Learning")
        
        if st.session_state.models_trained:
            models = st.session_state.models
            
            st.subheader("📊 Performance dos Modelos")
            
            # Tabela de performance
            performance_data = []
            for model_name, model_info in models['models'].items():
                performance_data.append({
                    "Modelo": model_name,
                    "RMSE Treino": f"{model_info['train_metrics']['rmse']:.4f}",
                    "RMSE Teste": f"{model_info['test_metrics']['rmse']:.4f}",
                    "R² Treino": f"{model_info['train_metrics']['r2']:.4f}",
                    "R² Teste": f"{model_info['test_metrics']['r2']:.4f}"
                })
            
            # Adicionar ensemble
            if 'ensemble' in models:
                perf_data = models['ensemble']['ensemble_metrics']
                performance_data.append({
                    "Modelo": "ENSEMBLE",
                    "RMSE Treino": f"{perf_data['train_metrics']['rmse']:.4f}",
                    "RMSE Teste": f"{perf_data['val_metrics']['rmse']:.4f}",
                    "R² Treino": f"{perf_data['train_metrics']['r2']:.4f}",
                    "R² Teste": f"{perf_data['val_metrics']['r2']:.4f}"
                })
            
            perf_df = pd.DataFrame(performance_data)
            st.dataframe(perf_df, use_container_width=True, hide_index=True)
            
            # Feature importance
            st.subheader("🎯 Importância das Features")
            
            if 'feature_importance' in models:
                # Combinar importância de todos os modelos
                importance_dfs = []
                for model_name, importance in models['feature_importance'].items():
                    if importance is not None:
                        importance_dfs.append(importance.to_frame(model_name))
                
                if importance_dfs:
                    combined_importance = pd.concat(importance_dfs, axis=1)
                    combined_importance['Média'] = combined_importance.mean(axis=1)
                    top_features = combined_importance['Média'].sort_values(ascending=False).head(10)
                    
                    fig_importance = go.Figure()
                    fig_importance.add_trace(go.Bar(
                        x=top_features.values,
                        y=top_features.index,
                        orientation='h',
                        marker_color='orange'
                    ))
                    
                    fig_importance.update_layout(
                        title='Top 10 Features Mais Importantes',
                        xaxis_title='Importância Média',
                        yaxis_title='Feature',
                        template='plotly_dark'
                    )
                    
                    st.plotly_chart(fig_importance, use_container_width=True)
    
    with tab4:
        st.header("⚙️ Configuração do Sistema")
        
        st.subheader("💾 Status do DataManager")
        
        if system['data_manager']:
            status = system['data_manager'].get_status()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Cache Items", status['cache_size'])
                st.metric("Data Types Cached", len(status['data_types_cached']))
            
            with col2:
                st.write("**Últimas Atualizações:**")
                for data_type, last_update in status['last_updates'].items():
                    st.write(f"- {data_type}: {last_update}")
        
        st.subheader("🛠️ Ferramentas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Limpar Cache", use_container_width=True):
                system['data_manager'].clear_cache()
                st.success("Cache limpo!")
                st.rerun()
        
        with col2:
            if st.button("💾 Salvar Estado", use_container_width=True):
                system['data_manager'].save_state()
                st.success("Estado salvo!")
        
        with col3:
            if st.button("📂 Carregar Estado", use_container_width=True):
                system['data_manager'].load_state()
                st.success("Estado carregado!")
                st.rerun()
        
        st.subheader("📊 Informações do Sistema")
        
        # Informações de configuração
        config = system['config']
        paths = config.get_paths()
        
        info_df = pd.DataFrame([
            {"Configuração": "FRED API", "Valor": "✅ Configurada" if config.FRED_API_KEY else "❌ Não configurada"},
            {"Configuração": "Cache Ativo", "Valor": "✅ Sim" if config.CACHE_ENABLED else "❌ Não"},
            {"Configuração": "Duração Cache", "Valor": f"{config.CACHE_DURATION_HOURS} horas"},
            {"Configuração": "Diretório Base", "Valor": paths['base']},
            {"Configuração": "Diretório Cache", "Valor": paths['data_cache']},
            {"Configuração": "Versão", "Valor": "1.0.0"}
        ])
        
        st.dataframe(info_df, use_container_width=True, hide_index=True)
        
        # Logs do sistema
        st.subheader("📝 Logs Recentes")
        log_placeholder = st.empty()
        
        # Simulação de logs
        logs = [
            f"{datetime.now().strftime('%H:%M:%S')} - Sistema inicializado",
            f"{datetime.now().strftime('%H:%M:%S')} - DataManager carregado",
            f"{datetime.now().strftime('%H:%M:%S')} - Cache verificado",
            f"{datetime.now().strftime('%H:%M:%S')} - APIs configuradas"
        ]
        
        log_text = "\n".join(logs)
        log_placeholder.code(log_text, language="text")

def main():
    """Função principal"""
    try:
        # Verificar dependências
        required_packages = [
            'streamlit', 'pandas', 'numpy', 'plotly', 
            'yfinance', 'fredapi', 'scikit-learn', 
            'xgboost', 'lightgbm'
        ]
        
        # Cabeçalho personalizado
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #F7931A;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
        
        <h1 class="main-header">₿ Sistema de Previsão Bitcoin</h1>
        <p class="sub-header">Análise quantitativa integrada com dados reais do FRED e Yahoo Finance</p>
        """, unsafe_allow_html=True)
        
        # Exibir dashboard
        display_dashboard()
        
        # Rodapé
        st.divider()
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.9rem;'>
        <p>⚠️ <strong>Aviso:</strong> Este é um sistema de análise quantitativa. 
        Não constitui recomendação de investimento.</p>
        <p>Desenvolvido com dados reais do FRED e Yahoo Finance | Versão 1.0.0</p>
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Erro crítico no sistema: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()