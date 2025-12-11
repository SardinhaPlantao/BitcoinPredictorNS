"""
MOTOR DE PREVISÃO PRINCIPAL DO BITCOIN
Integra análises e ML para gerar previsões
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

from ml_models.model_ensemble import ml_ensemble
from economic_analysis.correlation_analyzer import CorrelationAnalyzer
from technical_analysis.technical_analyzer import TechnicalAnalyzer
from core.data_manager import data_manager

class BitcoinPricePredictor:
    """
    Integra análises para previsão final
    """
    
    def __init__(self):
        self.correlator = CorrelationAnalyzer()
        self.technical = TechnicalAnalyzer()
    
    def generate_prediction(self, horizon_days: int = 30) -> Dict:
        """
        Pipeline completo de previsão
        """
        print("\nGerando Previsão...")
        
        # 1. Obter dados do DataManager
        all_data = data_manager.get_all_data()
        
        # 2. Análise Econômica
        correlations = self.correlator.analyze_correlations(
            all_data['bitcoin']['data']['Close'], 
            all_data['macro']
        )
        
        # 3. Análise Técnica
        tech_summary = self.technical.get_technical_summary(all_data['bitcoin']['data'])
        
        # 4. Features
        from ml_models.feature_engineer import feature_engineer
        features = feature_engineer.create_complete_feature_set(
            all_data['bitcoin']['data'], 
            all_data['macro'],
            all_data['related_assets']
        )
        
        # 5. Previsão ML
        df_ml = feature_engineer.prepare_target(features, horizon_days)
        X = df_ml.drop(['Target_Price', 'Target_Return', 'Target_Up'], axis=1)
        
        pred = ml_ensemble.predict(X.iloc[-1:])
        
        print(f"✅ Previsão: {pred[0]:.2f}")
        
        return {
            'predicted_price': pred[0],
            'current_price': all_data['bitcoin']['data']['Close'].iloc[-1],
            'expected_return': (pred[0] / all_data['bitcoin']['data']['Close'].iloc[-1] - 1) * 100,
            'tech_summary': tech_summary,
            'correlations': correlations
        }