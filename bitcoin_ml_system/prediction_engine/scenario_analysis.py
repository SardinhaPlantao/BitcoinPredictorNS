"""
ANÁLISE DE CENÁRIOS PARA PREVISÕES
Bullish, Bearish, Base Case
"""

import numpy as np
from typing import Dict

class ScenarioAnalyzer:
    """
    Gera cenários de previsão
    """
    
    def generate_scenarios(self, base_prediction: Dict, volatility: float) -> List[Dict]:
        """
        Cria cenários baseados na previsão principal
        """
        current = base_prediction['current_price']
        base_price = base_prediction['predicted_price']
        
        scenarios = [
            {'scenario': 'Bullish', 'price': base_price * (1 + volatility), 'color': 'green'},
            {'scenario': 'Base', 'price': base_price, 'color': 'blue'},
            {'scenario': 'Bearish', 'price': base_price * (1 - volatility), 'color': 'red'}
        ]
        
        return scenarios