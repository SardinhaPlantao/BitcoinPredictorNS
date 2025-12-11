"""
GERADOR DE RELATÓRIOS AUTOMATIZADOS
"""

import pandas as pd
from typing import Dict

class ReportGenerator:
    """
    Gera relatórios em formato DataFrame
    """
    
    def generate_summary_report(self, prediction: Dict, scenarios: list, risk: Dict) -> pd.DataFrame:
        data = {
            'Métrica': ['Preço Previsto', 'Retorno Esperado', 'VaR 95%', 'Expected Shortfall', 'Nível de Risco'],
            'Valor': [prediction['predicted_price'], prediction['expected_return'], risk['var_95'], risk['expected_shortfall'], risk['risk_level']]
        }
        return pd.DataFrame(data)