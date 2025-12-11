"""
DASHBOARD VISUALIZAÇÕES COM PLOTLY
"""

import plotly.graph_objects as go
import pandas as pd
from typing import Dict

class DashboardVisualizer:
    """
    Cria gráficos interativos para o dashboard
    """
    
    def plot_price_forecast(self, historical_prices: list, forecast: Dict) -> go.Figure:
        fig = go.Figure()
        
        # Histórico
        fig.add_trace(go.Scatter(y=historical_prices, name='Histórico'))
        
        # Previsão
        fig.add_trace(go.Scatter(y=[historical_prices[-1], forecast['predicted_price']], name='Previsão', mode='lines+markers'))
        
        fig.update_layout(title='Previsão de Preço Bitcoin')
        return fig