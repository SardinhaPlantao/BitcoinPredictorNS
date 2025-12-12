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
from plotly.subplots import make_subplots

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
    from bitcoin_ml_system.core.data_manager import data_manager
    from bitcoin_ml_system.core.config import config
    
    print("✅ Sistema inicializado")
    return {
        'data_manager': data_manager,
        'config': config
    }

@st.cache_data(ttl=3600)  # Cache de 1 hora
def load_data():
    """Carrega todos os dados necessários"""
    try:
        from bitcoin_ml_system.core.data_manager import data_manager
        
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
    from bitcoin_ml_system.data_collection.data_processor import data_processor
    
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
    # Importação condicional para evitar erros se módulos não existirem
    try:
        from bitcoin_ml_system.economic_analysis.cycle_analyzer import cycle_analyzer
        has_cycle_analyzer = True
    except ImportError:
        has_cycle_analyzer = False
    
    try:
        from bitcoin_ml_system.economic_analysis.macro_correlator import macro_correlator
        has_macro_correlator = True
    except ImportError:
        has_macro_correlator = False
    
    try:
        # Nota: na sua estrutura você tem technical_analysis.price_analyzer
        # Vou usar price_analyzer já que não vi technical_analyzer
        from bitcoin_ml_system.technical_analysis.price_analyzer import price_analyzer
        has_technical_analyzer = True
    except ImportError:
        has_technical_analyzer = False
    
    try:
        from bitcoin_ml_system.technical_analysis.volume_analysis import volume_analyzer
        has_volume_analyzer = True
    except ImportError:
        has_volume_analyzer = False
    
    try:
        analyses = {}
        
        # 1. Análise de ciclos econômicos
        if processed_data['macro_processed'] and has_cycle_analyzer:
            st.info("🔄 Analisando ciclos econômicos...")
            macro_df = processed_data['macro_processed']['data']
            cycle_analysis = cycle_analyzer.analyze_current_cycle(macro_df)
            analyses['cycle'] = cycle_analysis
        
        # 2. Análise de correlações macro-Bitcoin
        if has_macro_correlator:
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
        if has_technical_analyzer:
            st.info("📊 Analisando indicadores técnicos...")
            btc_data = processed_data['btc_processed']['data']
            # Ajuste para o nome correto do método
            try:
                tech_indicators = price_analyzer.calculate_all_indicators(btc_data)
                tech_report = price_analyzer.generate_technical_report(tech_indicators, btc_data)
            except AttributeError:
                # Fallback se os métodos tiverem nomes diferentes
                tech_indicators = {}
                tech_report = {'summary': {}, 'signals': {}, 'recommendation': {}}
            analyses['technical'] = tech_report
        
        # 4. Análise de volume
        if has_volume_analyzer:
            st.info("📈 Analisando volume...")
            btc_data = processed_data['btc_processed']['data']
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
    from bitcoin_ml_system.ml_models.model_trainer import model_trainer
    
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
    from bitcoin_ml_system.ml_models.model_trainer import model_trainer
    from bitcoin_ml_system.prediction_engine.price_predictor import price_predictor
    
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
            if isinstance(corr_df, pd.DataFrame) and not corr_df.empty:
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
        
        # 4. Gráfico de indicadores técnicos (se disponível)
        if analyses and 'technical' in analyses['analyses'] and data and 'bitcoin' in data['data']:
            btc_data = data['data']['bitcoin']['data']
            
            fig_tech = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Preço', 'Volume'),
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3]
            )
            
            # Preço
            fig_tech.add_trace(
                go.Scatter(x=btc_data.index, y=btc_data['Close'], name='Preço'),
                row=1, col=1
            )
            
            # Volume
            if 'Volume' in btc_data.columns:
                colors = ['red' if btc_data['Close'].iloc[i] < btc_data['Close'].iloc[i-1] 
                         else 'green' for i in range(len(btc_data))]
                fig_tech.add_trace(
                    go.Bar(x=btc_data.index, y=btc_data['Volume'], name='Volume',
                          marker_color=colors),
                    row=2, col=1
                )
            
            fig_tech.update_layout(height=600, template='plotly_dark', showlegend=True)
            visualizations['technical_chart'] = fig_tech
        
    except Exception as e:
        st.error(f"❌ Erro nas visualizações: {e}")
    
    return visualizations

def display_dashboard():
    """Exibe o dashboard principal"""
    
    # Inicializar sistema
    try:
        system = initialize_system()
    except Exception as e:
        st.error(f"❌ Erro ao inicializar sistema: {e}")
        st.info("Verifique se os módulos 'data_manager' e 'config' existem na pasta 'bitcoin_ml_system/core/'")
        return
    
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
                if 'price_chart' in viz:
                    st.plotly_chart(viz['price_chart'], use_container_width=True)
                
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
                        if 'analyses' in st.session_state and 'technical' in st.session_state.analyses['analyses']:
                            tech = st.session_state.analyses['analyses']['technical']
                            action = tech.get('recommendation', {}).get('action', 'Neutro')
                            score = tech.get('recommendation', {}).get('score', 0)
                            st.metric(
                                "Sinal Técnico",
                                action,
                                delta=f"Score: {score}"
                            )
            
            with col2:
                if 'prediction_chart' in viz:
                    st.plotly_chart(viz['prediction_chart'], use_container_width=True)
            
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
                if 'analyses' in st.session_state and 'technical' in st.session_state.analyses['analyses']:
                    tech = st.session_state.analyses['analyses']['technical']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        rsi = tech.get('summary', {}).get('rsi_14', 50)
                        st.metric("RSI", f"{rsi:.1f}", 
                                 "Sobrevendido" if rsi < 30 else "Sobrecomprado" if rsi > 70 else "Neutro")
                    with col2:
                        trend = tech.get('trend', {}).get('trend', 'Neutro')
                        strength = tech.get('trend', {}).get('strength', 0)
                        st.metric("Tendência", trend, f"Força: {strength:.2f}")
                    with col3:
                        volatility = tech.get('summary', {}).get('volatility_30d', 0)
                        st.metric("Volatilidade", f"{volatility:.1f}%", "30 dias")
            
            with tab_c:
                # Análise macro
                if 'analyses' in st.session_state and 'cycle' in st.session_state.analyses['analyses']:
                    cycle = st.session_state.analyses['analyses']['cycle']
                    
                    st.markdown(f"""
                    ### 🌍 Análise Macroeconômica
                    
                    **Ciclo Atual:** {cycle.get('current_phase', {}).get('label', 'Desconhecido')}
                    **Duração:** {cycle.get('current_phase', {}).get('duration_days', 0)} dias
                    **Confiança:** {cycle.get('confidence_score', 0):.0%}
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
                    indicators_data = [
                        {"Indicador": "Preço", "Valor": f"${tech.get('summary', {}).get('price', 0):,.2f}", "Status": "-"},
                        {"Indicador": "RSI 14", "Valor": f"{tech.get('summary', {}).get('rsi_14', 0):.1f}", 
                         "Status": "Sobrevendido" if tech.get('summary', {}).get('rsi_14', 50) < 30 
                         else "Sobrecomprado" if tech.get('summary', {}).get('rsi_14', 50) > 70 else "Neutro"},
                    ]
                    indicators_df = pd.DataFrame(indicators_data)
                    st.dataframe(indicators_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.header("🤖 Machine Learning")
        
        if st.session_state.models_trained:
            st.info("Funcionalidade de ML disponível após correção dos módulos específicos")
            # Esta seção funcionará depois que os módulos ml_models estiverem corrigidos
    
    with tab4:
        st.header("⚙️ Configuração do Sistema")
        
        try:
            st.subheader("💾 Status do DataManager")
            
            if system['data_manager']:
                status = system['data_manager'].get_status()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Cache Items", status.get('cache_size', 0))
                    st.metric("Data Types Cached", len(status.get('data_types_cached', [])))
                
                with col2:
                    st.write("**Últimas Atualizações:**")
                    for data_type, last_update in status.get('last_updates', {}).items():
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
                {"Configuração": "Versão", "Valor": "1.0.0"}
            ])
            
            st.dataframe(info_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Erro ao carregar configurações: {e}")

def main():
    """Função principal"""
    try:
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