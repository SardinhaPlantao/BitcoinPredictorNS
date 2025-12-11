"""
COLETOR DE DADOS FRED (Federal Reserve Economic Data)
Coleta dados macroeconômicos REAIS
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

try:
    from fredapi import Fred
    FRED_AVAILABLE = True
except ImportError:
    FRED_AVAILABLE = False
    print("⚠️ fredapi não instalado. Execute: pip install fredapi")

from core.config import config
from utils.helpers import resample_time_series

class FREDDataCollector:
    """
    Coleta dados econômicos do Federal Reserve
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa o coletor do FRED
        
        Args:
            api_key: Chave da API do FRED (usa config.FRED_API_KEY se None)
        """
        if not FRED_AVAILABLE:
            raise ImportError("fredapi não está instalado")
        
        self.api_key = api_key or config.FRED_API_KEY
        
        try:
            self.fred = Fred(api_key=self.api_key)
            self.connected = True
            print("✅ FRED API conectada com sucesso")
        except Exception as e:
            self.connected = False
            print(f"❌ Falha ao conectar ao FRED: {e}")
    
    def get_series(self, 
                  series_id: str, 
                  start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None,
                  years_back: int = 10) -> Optional[pd.Series]:
        """
        Obtém uma série específica do FRED
        
        Args:
            series_id: ID da série (ex: GDP, UNRATE)
            start_date: Data inicial (usa years_back se None)
            end_date: Data final (usa hoje se None)
            years_back: Anos de histórico (se start_date não fornecido)
        
        Returns:
            Série temporal ou None se erro
        """
        if not self.connected:
            print("⚠️ FRED não conectado")
            return None
        
        try:
            # Definir datas padrão
            if end_date is None:
                end_date = datetime.now()
            
            if start_date is None:
                start_date = end_date - timedelta(days=years_back * 365)
            
            print(f"📥 FRED: {series_id} ({start_date.date()} a {end_date.date()})")
            
            # Buscar série
            series = self.fred.get_series(
                series_id,
                observation_start=start_date.strftime('%Y-%m-%d'),
                observation_end=end_date.strftime('%Y-%m-%d')
            )
            
            if series.empty:
                print(f"⚠️ Série {series_id} vazia")
                return None
            
            # Converter para frequência diária
            series_daily = series.resample('D').ffill()
            
            print(f"✅ {series_id}: {len(series_daily)} períodos")
            return series_daily
            
        except Exception as e:
            print(f"❌ Erro ao obter série {series_id}: {e}")
            return None
    
    def get_multiple_series(self, 
                          series_ids: List[str],
                          years_back: int = 10) -> Optional[pd.DataFrame]:
        """
        Obtém múltiplas séries simultaneamente
        
        Args:
            series_ids: Lista de IDs de série
            years_back: Anos de histórico
        
        Returns:
            DataFrame com todas as séries ou None
        """
        if not self.connected:
            return None
        
        print(f"\n📥 FRED: Coletando {len(series_ids)} séries ({years_back} anos)")
        
        all_series = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years_back * 365)
        
        for series_id in series_ids:
            series = self.get_series(series_id, start_date, end_date)
            if series is not None:
                all_series[series_id] = series
        
        if not all_series:
            print("❌ Nenhuma série coletada")
            return None
        
        # Criar DataFrame combinado
        df = pd.DataFrame(all_series)
        
        # Preencher valores NaN
        df = df.ffill().bfill()
        
        print(f"✅ {len(df.columns)} séries coletadas, {len(df)} períodos")
        return df
    
    def get_all_macro_indicators(self, 
                               years_back: int = 10) -> Optional[pd.DataFrame]:
        """
        Obtém todos os indicadores macro configurados
        
        Args:
            years_back: Anos de histórico
        
        Returns:
            DataFrame com todos os indicadores macro
        """
        series_ids = list(config.MACRO_INDICATORS.values())
        return self.get_multiple_series(series_ids, years_back)
    
    def get_indicators_by_category(self, 
                                 years_back: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Obtém indicadores agrupados por categoria
        
        Args:
            years_back: Anos de histórico
        
        Returns:
            Dicionário com DataFrames por categoria
        """
        categories = {
            'monetary_policy': ['FEDFUNDS', 'M2SL', 'WRESBAL'],
            'yield_curve': ['T10Y2Y', 'T10Y3M', 'DGS10', 'DGS2'],
            'inflation': ['CPIAUCSL', 'CPILFESL'],
            'labor_market': ['UNRATE', 'PAYEMS', 'ICSA'],
            'economic_activity': ['INDPRO', 'RETAILSM', 'GDP'],
            'sentiment': ['UMCSENT', 'VIXCLS'],
            'other': ['HOUST', 'BUSINV', 'DTWEXBGS', 'USSLIND', 'USPHCI']
        }
        
        results = {}
        
        for category, series_ids in categories.items():
            print(f"\n📥 Coletando categoria: {category}")
            df = self.get_multiple_series(series_ids, years_back)
            if df is not None:
                results[category] = df
        
        return results
    
    def calculate_macro_features(self, 
                               macro_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula features macroeconômicas derivadas
        
        Args:
            macro_data: DataFrame com dados macro brutos
        
        Returns:
            DataFrame com features macro calculadas
        """
        if macro_data.empty:
            return macro_data
        
        features = macro_data.copy()
        
        # 1. Liquidez líquida (simplificada)
        if 'M2SL' in features.columns:
            features['net_liquidity'] = features['M2SL'].pct_change(365) * 100
        
        # 2. Yield curve signals
        if 'T10Y2Y' in features.columns:
            features['yield_curve_inverted'] = (features['T10Y2Y'] < 0).astype(int)
            features['yield_curve_trend'] = features['T10Y2Y'].diff(30)
        
        # 3. Inflação
        if 'CPIAUCSL' in features.columns:
            features['inflation_yoy'] = features['CPIAUCSL'].pct_change(365) * 100
        
        # 4. Desemprego momentum
        if 'UNRATE' in features.columns:
            features['unemployment_change'] = features['UNRATE'].diff(90)
        
        # 5. Produção industrial
        if 'INDPRO' in features.columns:
            features['industrial_production_yoy'] = features['INDPRO'].pct_change(365) * 100
        
        # 6. Difusão index (simplificado)
        # Contar quantos indicadores estão acima da média móvel
        indicator_columns = [col for col in features.columns 
                           if col in config.MACRO_INDICATORS.values()]
        
        if indicator_columns:
            # Calcular se cada indicador está acima da média de 200 dias
            above_ma = pd.DataFrame()
            for col in indicator_columns:
                ma = features[col].rolling(200).mean()
                above_ma[col] = (features[col] > ma).astype(int)
            
            features['diffusion_index'] = above_ma.mean(axis=1) * 100
        
        # 7. Financial conditions index (simplificado)
        components = []
        
        if 'VIXCLS' in features.columns:
            # VIX normalizado
            vix_norm = (features['VIXCLS'] - features['VIXCLS'].rolling(200).mean()) / features['VIXCLS'].rolling(200).std()
            components.append(vix_norm)
        
        if 'T10Y2Y' in features.columns:
            # Yield curve normalizado
            yc_norm = (features['T10Y2Y'] - features['T10Y2Y'].rolling(200).mean()) / features['T10Y2Y'].rolling(200).std()
            components.append(yc_norm)
        
        if components:
            financial_conditions = pd.concat(components, axis=1).mean(axis=1)
            features['financial_conditions_index'] = financial_conditions
        
        return features
    
    def validate_macro_data(self, macro_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida a qualidade dos dados macro
        
        Args:
            macro_data: DataFrame com dados macro
        
        Returns:
            Dicionário com relatório de validação
        """
        report = {
            'valid': True,
            'n_indicators': len(macro_data.columns),
            'n_periods': len(macro_data),
            'period_start': macro_data.index[0].strftime('%Y-%m-%d'),
            'period_end': macro_data.index[-1].strftime('%Y-%m-%d'),
            'issues': [],
            'coverage': {}
        }
        
        if macro_data.empty:
            report['valid'] = False
            report['issues'].append('DataFrame vazio')
            return report
        
        # Verificar cobertura por indicador
        for col in macro_data.columns:
            non_nan = macro_data[col].notna().sum()
            coverage = non_nan / len(macro_data) * 100
            report['coverage'][col] = float(coverage)
            
            if coverage < 50:
                report['issues'].append(f'Baixa cobertura para {col}: {coverage:.1f}%')
        
        # Verificar valores extremos
        for col in macro_data.columns:
            if macro_data[col].dtype in [np.float64, np.int64]:
                q1 = macro_data[col].quantile(0.25)
                q3 = macro_data[col].quantile(0.75)
                iqr = q3 - q1
                
                outliers = ((macro_data[col] < q1 - 3 * iqr) | 
                          (macro_data[col] > q3 + 3 * iqr)).sum()
                
                if outliers > len(macro_data) * 0.05:  # Mais de 5% outliers
                    report['issues'].append(f'Muitos outliers em {col}: {outliers}')
        
        return report

# Instância global do coletor FRED
fred_collector = FREDDataCollector()