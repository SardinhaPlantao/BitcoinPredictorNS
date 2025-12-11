"""
CONFIGURAÇÕES GLOBAIS DO SISTEMA
Centraliza todas as configurações, chaves API e parâmetros
"""

import os
from datetime import datetime
from typing import Dict, List, Any

class SystemConfig:
    """
    Configurações globais do sistema de previsão do Bitcoin
    """
    
    # ========== CONFIGURAÇÕES DE API ==========
    FRED_API_KEY = "523cf7c56d5ab6dc2298c8f4f7ce366a"  # Chave pública do FRED
    
    # ========== CONFIGURAÇÕES DE TEMPO ==========
    DEFAULT_START_DATE = datetime(2015, 1, 1)  # Bitcoin mais maduro pós-2015
    DEFAULT_HISTORICAL_YEARS = 5
    TIMEZONE = "America/New_York"
    
    # ========== CONFIGURAÇÕES DE CACHE ==========
    CACHE_ENABLED = True
    CACHE_DURATION_HOURS = 24  # Cache de 24 horas
    CACHE_DIR = "./data_cache"
    
    # ========== CONFIGURAÇÕES DE DADOS ==========
    
    # Símbolos para coleta
    BTC_SYMBOL = "BTC-USD"
    RELATED_ASSETS = {
        'sp500': '^GSPC',
        'gold': 'GC=F',
        'treasury_10y': '^TNX',
        'treasury_2y': '^FVX',
        'dollar_index': 'DX-Y.NYB',
        'vix': '^VIX'
    }
    
    # Indicadores macroeconômicos do FRED (Série IDs)
    MACRO_INDICATORS = {
        # Política Monetária
        'fed_funds_rate': 'FEDFUNDS',
        'm2_money_supply': 'M2SL',
        'reserve_balances': 'WRESBAL',
        
        # Yield Curve
        'yield_curve_10y_2y': 'T10Y2Y',
        'yield_curve_10y_3m': 'T10Y3M',
        'treasury_10y': 'DGS10',
        'treasury_2y': 'DGS2',
        
        # Inflação
        'cpi_all': 'CPIAUCSL',
        'cpi_core': 'CPILFESL',
        
        # Mercado de Trabalho
        'unemployment_rate': 'UNRATE',
        'nonfarm_payrolls': 'PAYEMS',
        'initial_claims': 'ICSA',
        
        # Atividade Econômica
        'industrial_production': 'INDPRO',
        'retail_sales': 'RETAILSM',
        'gdp': 'GDP',
        
        # Sentimento
        'consumer_sentiment': 'UMCSENT',
        'vix_closing': 'VIXCLS',
        
        # Outros importantes
        'housing_starts': 'HOUST',
        'business_inventories': 'BUSINV',
        'dollar_index_fred': 'DTWEXBGS',
        'leading_index': 'USSLIND',
        'coincident_index': 'USPHCI'
    }
    
    # ========== CONFIGURAÇÕES DE MODELO ==========
    ML_CONFIG = {
        'train_test_split': 0.8,
        'cv_folds': 5,
        'random_state': 42,
        'test_size_months': 6,
        'validation_size_months': 6
    }
    
    # ========== CONFIGURAÇÕES DE ANÁLISE ==========
    ANALYSIS_CONFIG = {
        'rolling_window': 30,
        'correlation_threshold': 0.3,
        'significance_level': 0.05,
        'min_data_points': 100
    }
    
    # ========== PATHS DO SISTEMA ==========
    @staticmethod
    def get_paths() -> Dict[str, str]:
        """Retorna todos os paths do sistema"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return {
            'base': base_dir,
            'data_cache': os.path.join(base_dir, 'data_cache'),
            'models': os.path.join(base_dir, 'models'),
            'logs': os.path.join(base_dir, 'logs'),
            'reports': os.path.join(base_dir, 'reports')
        }
    
    # ========== VALIDAÇÃO ==========
    @classmethod
    def validate(cls) -> bool:
        """Valida se todas as configurações são válidas"""
        try:
            # Verificar se diretórios existem
            paths = cls.get_paths()
            for path_name, path in paths.items():
                if not os.path.exists(path) and path_name != 'base':
                    os.makedirs(path, exist_ok=True)
            
            # Verificar configurações críticas
            assert cls.FRED_API_KEY, "FRED_API_KEY não configurada"
            assert cls.BTC_SYMBOL, "BTC_SYMBOL não configurado"
            assert cls.DEFAULT_HISTORICAL_YEARS > 0, "DEFAULT_HISTORICAL_YEARS deve ser > 0"
            
            print("✅ Configurações validadas com sucesso")
            return True
            
        except Exception as e:
            print(f"❌ Erro na validação das configurações: {e}")
            return False

# Instância global de configuração
config = SystemConfig()