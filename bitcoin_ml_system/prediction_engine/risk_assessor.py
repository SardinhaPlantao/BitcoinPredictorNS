"""
AVALIAÇÃO DE RISCO PARA PREVISÕES
"""

import numpy as np
from typing import Dict

class RiskAssessor:
    """
    Avalia riscos da previsão
    """
    
    def assess_risk(self, prediction: Dict, historical_data: pd.DataFrame) -> Dict:
        """
        Calcula métricas de risco
        """
        returns = historical_data['Close'].pct_change().dropna()
        
        var_95 = np.percentile(returns, 5) * 100  # Value at Risk
        es = np.mean(returns[returns < var_95 / 100]) * 100  # Expected Shortfall
        
        return {
            'var_95': var_95,
            'expected_shortfall': es,
            'risk_level': 'High' if abs(var_95) > 5 else 'Medium' if abs(var_95) > 3 else 'Low'
        }