"""
PROCESSADOR DE DADOS - Limpeza, transformação e engenharia de features básicas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from core.data_manager import data_manager
from core.config import config
from utils.helpers import calculate_returns, calculate_volatility, align_time_series

class DataProcessor:
    """
    Processa dados brutos para análise
    """
    
    def __init__(self):
        self.data_manager = data_manager
    
    def process_bitcoin_data(self, btc_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Processa dados do Bitcoin: limpeza, feature engineering básico
        
        Args:
            btc_data: DataFrame com dados brutos do Bitcoin
            
        Returns:
            Dicionário com dados processados
        """
        if btc_data is None or btc_data.empty:
            raise ValueError("Dados Bitcoin vazios")
        
        df = btc_data.copy()
        
        # 1. Garantir que temos colunas essenciais
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Coluna {col} não encontrada nos dados Bitcoin")
        
        # 2. Remover outliers extremos
        df = self._remove_price_outliers(df)
        
        # 3. Calcular features básicas
        df['Returns'] = calculate_returns(df['Close'], method='log')
        df['Volatility_30d'] = calculate_volatility(df['Returns'], window=30, annualize=True)
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        
        # 4. Calcular drawdown
        df['Cumulative_Max'] = df['Close'].expanding().max()
        df['Drawdown'] = (df['Close'] - df['Cumulative_Max']) / df['Cumulative_Max']
        
        # 5. Features de tendência
        df['Trend_7d'] = df['Close'].rolling(window=7).mean()
        df['Trend_30d'] = df['Close'].rolling(window=30).mean()
        df['Price_vs_Trend'] = (df['Close'] / df['Trend_30d'] - 1) * 100
        
        # 6. RSI (simplificado)
        df['RSI'] = self._calculate_rsi(df['Close'], period=14)
        
        # 7. MACD (simplificado)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # 8. Remover NaN resultantes dos cálculos
        df = df.fillna(method='bfill').fillna(method='ffill')
        
        # 9. Estatísticas
        stats = {
            'current_price': float(df['Close'].iloc[-1]),
            'price_7d_ago': float(df['Close'].iloc[-7]) if len(df) > 7 else None,
            'price_30d_ago': float(df['Close'].iloc[-30]) if len(df) > 30 else None,
            'max_drawdown': float(df['Drawdown'].min() * 100),
            'volatility_30d': float(df['Volatility_30d'].iloc[-1]),
            'avg_volume': float(df['Volume'].mean()),
            'rsi_current': float(df['RSI'].iloc[-1]),
            'data_points': len(df),
            'date_range': {
                'start': df.index[0].strftime('%Y-%m-%d'),
                'end': df.index[-1].strftime('%Y-%m-%d')
            }
        }
        
        return {
            'data': df,
            'stats': stats,
            'features': list(df.columns)
        }
    
    def _remove_price_outliers(self, df: pd.DataFrame, std_threshold: float = 5.0) -> pd.DataFrame:
        """
        Remove outliers extremos baseados no desvio padrão
        """
        returns = df['Close'].pct_change().dropna()
        
        # Calcular limites
        mean_return = returns.mean()
        std_return = returns.std()
        
        lower_bound = mean_return - std_threshold * std_return
        upper_bound = mean_return + std_threshold * std_return
        
        # Identificar outliers
        outlier_mask = (returns < lower_bound) | (returns > upper_bound)
        
        if outlier_mask.any():
            print(f"⚠️  {outlier_mask.sum()} outliers detectados e removidos")
            
            # Para cada outlier, substituir pelo valor anterior
            outlier_indices = returns[outlier_mask].index
            for idx in outlier_indices:
                if idx in df.index:
                    prev_idx = df.index[df.index.get_loc(idx) - 1]
                    df.loc[idx, 'Close'] = df.loc[prev_idx, 'Close']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI simplificado"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def process_macro_data(self, macro_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Processa dados macroeconômicos do FRED
        
        Args:
            macro_data: DataFrame com dados macro brutos
            
        Returns:
            Dicionário com dados macro processados
        """
        if macro_data is None or macro_data.empty:
            raise ValueError("Dados macro vazios")
        
        df = macro_data.copy()
        
        # 1. Converter para frequência diária (preencher valores)
        df = df.resample('D').ffill()
        
        # 2. Calcular mudanças percentuais para indicadores importantes
        for col in df.columns:
            # Para taxas (como fed funds), usar diferença absoluta
            if 'rate' in col.lower() or 'funds' in col.lower():
                df[f'{col}_change'] = df[col].diff()
            else:
                # Para outros indicadores, usar mudança percentual
                df[f'{col}_pct_change'] = df[col].pct_change()
        
        # 3. Calcular tendências
        for col in df.columns:
            if not any(x in col for x in ['_change', '_pct_change']):
                df[f'{col}_trend_30d'] = df[col].rolling(window=30).mean()
                df[f'{col}_trend_90d'] = df[col].rolling(window=90).mean()
        
        # 4. Criar índice composto de condições econômicas
        # Usar indicadores selecionados (pesos podem ser ajustados)
        indicators_for_index = [
            'T10Y2Y',  # Yield curve (importante para recessões)
            'UNRATE',  # Unemployment
            'CPIAUCSL',  # Inflation
            'INDPRO',   # Industrial production
            'M2SL'      # Money supply
        ]
        
        available_indicators = [ind for ind in indicators_for_index if ind in df.columns]
        
        if available_indicators:
            # Normalizar cada indicador
            normalized_indicators = []
            for indicator in available_indicators:
                series = df[indicator].dropna()
                if len(series) > 0:
                    normalized = (series - series.mean()) / series.std()
                    normalized_indicators.append(normalized)
            
            # Calcular índice composto (média dos normalizados)
            if normalized_indicators:
                composite_index = pd.concat(normalized_indicators, axis=1).mean(axis=1)
                df['economic_conditions_index'] = composite_index
        
        # 5. Remover NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # 6. Estatísticas
        stats = {}
        for col in df.columns:
            if col in config.MACRO_INDICATORS.values():
                stats[col] = {
                    'current': float(df[col].iloc[-1]) if len(df) > 0 else None,
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        return {
            'data': df,
            'stats': stats,
            'indicators_count': len(df.columns),
            'date_range': {
                'start': df.index[0].strftime('%Y-%m-%d') if len(df) > 0 else None,
                'end': df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else None
            }
        }
    
    def align_datasets(self, 
                      btc_data: pd.DataFrame, 
                      macro_data: pd.DataFrame,
                      assets_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Alinha todos os datasets para o mesmo período temporal
        
        Args:
            btc_data: Dados Bitcoin
            macro_data: Dados macro
            assets_data: Dados de ativos relacionados
            
        Returns:
            Dicionário com datasets alinhados
        """
        print("\n🔗 Alinhando datasets temporalmente...")
        
        # 1. Garantir que todos são DataFrames com DatetimeIndex
        datasets = {'bitcoin': btc_data}
        
        if macro_data is not None and not macro_data.empty:
            datasets['macro'] = macro_data
        
        # 2. Adicionar ativos relacionados
        if assets_data is not None:
            for asset_name, asset_df in assets_data.items():
                if not asset_df.empty and 'Close' in asset_df.columns:
                    datasets[asset_name] = asset_df[['Close']].rename(
                        columns={'Close': f'{asset_name}_close'}
                    )
        
        # 3. Encontrar período comum
        common_index = None
        for name, df in datasets.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        if len(common_index) < 30:
            raise ValueError(f"Período comum muito curto: {len(common_index)} dias")
        
        print(f"   Período comum: {len(common_index)} dias "
              f"({common_index[0].strftime('%Y-%m-%d')} a {common_index[-1].strftime('%Y-%m-%d')})")
        
        # 4. Alinhar todos os datasets
        aligned_datasets = {}
        for name, df in datasets.items():
            aligned = df.reindex(common_index)
            aligned = aligned.fillna(method='ffill').fillna(method='bfill')
            aligned_datasets[name] = aligned
        
        # 5. Criar DataFrame mestre combinado
        master_df = pd.DataFrame(index=common_index)
        
        for name, df in aligned_datasets.items():
            for col in df.columns:
                master_df[f'{name}_{col}'] = df[col]
        
        # 6. Remover colunas duplicadas
        master_df = master_df.loc[:, ~master_df.columns.duplicated()]
        
        # 7. Estatísticas do dataset alinhado
        stats = {
            'total_days': len(master_df),
            'total_features': len(master_df.columns),
            'start_date': master_df.index[0].strftime('%Y-%m-%d'),
            'end_date': master_df.index[-1].strftime('%Y-%m-%d'),
            'missing_values': master_df.isna().sum().sum(),
            'datasets_included': list(aligned_datasets.keys())
        }
        
        return {
            'master_dataframe': master_df,
            'aligned_datasets': aligned_datasets,
            'stats': stats,
            'common_index': common_index
        }
    
    def prepare_ml_dataset(self, 
                          master_df: pd.DataFrame,
                          target_column: str = 'bitcoin_Close',
                          forecast_horizon: int = 30) -> Dict[str, Any]:
        """
        Prepara dataset para Machine Learning
        
        Args:
            master_df: DataFrame mestre com todos os dados
            target_column: Coluna alvo para previsão
            forecast_horizon: Horizonte de previsão em dias
            
        Returns:
            Dataset preparado para ML
        """
        print(f"\n🤖 Preparando dataset ML (horizonte: {forecast_horizon} dias)...")
        
        if target_column not in master_df.columns:
            raise ValueError(f"Coluna alvo {target_column} não encontrada")
        
        df = master_df.copy()
        
        # 1. Criar target (preço futuro)
        df['target'] = df[target_column].shift(-forecast_horizon)
        
        # 2. Remover linhas com target NaN (últimos dias)
        df = df.dropna(subset=['target'])
        
        # 3. Features básicas
        # 3.1. Lags do preço Bitcoin
        for lag in [1, 2, 3, 5, 7, 14, 30]:
            df[f'btc_lag_{lag}'] = df[target_column].shift(lag)
        
        # 3.2. Retornos históricos
        for window in [1, 3, 7, 14, 30]:
            df[f'btc_return_{window}d'] = df[target_column].pct_change(window)
        
        # 3.3. Volatilidade
        returns = df[target_column].pct_change()
        for window in [7, 14, 30]:
            df[f'btc_volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(365)
        
        # 3.4. Médias móveis
        for window in [7, 20, 50, 100]:
            df[f'btc_sma_{window}'] = df[target_column].rolling(window).mean()
            df[f'btc_sma_ratio_{window}'] = df[target_column] / df[f'btc_sma_{window}']
        
        # 4. Features de outros ativos (se disponíveis)
        for col in df.columns:
            if '_close' in col and col != target_column:
                asset_name = col.replace('_close', '')
                # Correlação rolling com BTC
                df[f'{asset_name}_corr_30d'] = df[col].rolling(30).corr(df[target_column])
                # Retorno relativo
                df[f'{asset_name}_rel_return'] = df[col].pct_change() - df[target_column].pct_change()
        
        # 5. Features macro (se disponíveis)
        macro_cols = [col for col in df.columns if 'macro_' in col]
        for col in macro_cols:
            macro_name = col.replace('macro_', '')
            # Tendência macro
            df[f'{macro_name}_trend_30d'] = df[col].rolling(30).mean()
            df[f'{macro_name}_trend_ratio'] = df[col] / df[f'{macro_name}_trend_30d']
        
        # 6. Remover NaN resultante das features
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remover colunas com muitos NaN (se houver)
        threshold = 0.1  # Máximo 10% NaN
        valid_cols = df.columns[df.isna().mean() < threshold]
        df = df[valid_cols]
        
        # 7. Separar features e target
        feature_cols = [col for col in df.columns if col not in ['target', target_column]]
        X = df[feature_cols]
        y = df['target']
        
        # 8. Estatísticas do dataset ML
        ml_stats = {
            'total_samples': len(X),
            'total_features': len(feature_cols),
            'feature_categories': {
                'btc_features': len([c for c in feature_cols if c.startswith('btc_')]),
                'macro_features': len([c for c in feature_cols if 'macro_' in c]),
                'asset_features': len([c for c in feature_cols if any(x in c for x in ['sp500', 'gold', 'treasury', 'vix'])]),
                'technical_features': len([c for c in feature_cols if any(x in c for x in ['lag', 'return', 'volatility', 'sma'])]),
            },
            'target_stats': {
                'mean': float(y.mean()),
                'std': float(y.std()),
                'min': float(y.min()),
                'max': float(y.max())
            }
        }
        
        print(f"   ✅ Dataset ML preparado: {len(X)} amostras, {len(feature_cols)} features")
        
        return {
            'X': X,
            'y': y,
            'feature_names': feature_cols,
            'dataframe': df,
            'stats': ml_stats,
            'forecast_horizon': forecast_horizon
        }

# Instância global
data_processor = DataProcessor()