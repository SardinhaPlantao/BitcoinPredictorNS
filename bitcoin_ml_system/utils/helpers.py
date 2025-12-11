"""
FUNÇÕES AUXILIARES PARA TODO O SISTEMA
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

def calculate_returns(prices: pd.Series, method: str = 'log') -> pd.Series:
    """
    Calcula retornos de uma série de preços
    
    Args:
        prices: Série de preços
        method: 'log' para log-retornos, 'simple' para retornos simples
    
    Returns:
        Série de retornos
    """
    if method == 'log':
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    
    returns.name = f'{prices.name}_returns' if prices.name else 'returns'
    return returns

def calculate_volatility(returns: pd.Series, 
                        window: int = 30, 
                        annualize: bool = True) -> pd.Series:
    """
    Calcula volatilidade rolling
    
    Args:
        returns: Série de retornos
        window: Janela em dias
        annualize: Se True, anualiza a volatilidade
    
    Returns:
        Série de volatilidade
    """
    volatility = returns.rolling(window=window).std()
    
    if annualize:
        volatility = volatility * np.sqrt(365)
    
    volatility.name = f'{returns.name}_volatility' if returns.name else 'volatility'
    return volatility

def align_time_series(*series_list: pd.Series) -> List[pd.Series]:
    """
    Alinha múltiplas séries temporais para o mesmo índice
    
    Args:
        *series_list: Séries temporais para alinhar
    
    Returns:
        Lista de séries alinhadas
    """
    # Encontrar índice comum
    common_index = series_list[0].index
    for series in series_list[1:]:
        common_index = common_index.intersection(series.index)
    
    # Alinhar todas as séries
    aligned_series = []
    for series in series_list:
        aligned = series.reindex(common_index)
        aligned = aligned.ffill().bfill()
        aligned_series.append(aligned)
    
    return aligned_series

def calculate_correlation_matrix(data: pd.DataFrame, 
                               method: str = 'pearson') -> pd.DataFrame:
    """
    Calcula matriz de correlação com tratamento de NaN
    
    Args:
        data: DataFrame com dados
        method: Método de correlação ('pearson', 'spearman', 'kendall')
    
    Returns:
        Matriz de correlação
    """
    # Remover colunas com muitos NaN
    threshold = 0.3  # Máximo 30% de NaN
    valid_cols = data.columns[data.isna().mean() < threshold]
    data_clean = data[valid_cols].copy()
    
    # Preencher NaN restantes
    data_clean = data_clean.ffill().bfill()
    
    # Calcular correlação
    corr_matrix = data_clean.corr(method=method)
    
    return corr_matrix

def create_lagged_features(series: pd.Series, 
                          lags: List[int]) -> pd.DataFrame:
    """
    Cria features com lags para uma série temporal
    
    Args:
        series: Série temporal
        lags: Lista de lags para criar
    
    Returns:
        DataFrame com features com lag
    """
    features = pd.DataFrame(index=series.index)
    
    for lag in lags:
        if lag > 0:
            features[f'lag_{lag}'] = series.shift(lag)
        elif lag < 0:
            features[f'lead_{abs(lag)}'] = series.shift(lag)
        else:
            features['current'] = series
    
    return features

def calculate_technical_indicators(prices: pd.Series, 
                                 volume: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Calcula indicadores técnicos básicos
    
    Args:
        prices: Série de preços
        volume: Série de volume (opcional)
    
    Returns:
        DataFrame com indicadores técnicos
    """
    indicators = pd.DataFrame(index=prices.index)
    
    # Médias móveis
    for window in [7, 20, 50, 200]:
        indicators[f'sma_{window}'] = prices.rolling(window=window).mean()
    
    # Bandas de Bollinger
    sma_20 = prices.rolling(window=20).mean()
    std_20 = prices.rolling(window=20).std()
    indicators['bb_upper'] = sma_20 + 2 * std_20
    indicators['bb_lower'] = sma_20 - 2 * std_20
    indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / sma_20
    
    # RSI
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    indicators['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    indicators['macd'] = exp1 - exp2
    indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
    
    if volume is not None:
        # Volume médio
        indicators['volume_sma_20'] = volume.rolling(window=20).mean()
        indicators['volume_ratio'] = volume / indicators['volume_sma_20']
    
    return indicators

def detect_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """
    Detecta outliers usando o método IQR
    
    Args:
        data: Série de dados
        multiplier: Multiplicador do IQR
    
    Returns:
        Série booleana indicando outliers
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    
    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers

def resample_time_series(data: pd.DataFrame, 
                        freq: str = 'D',
                        method: str = 'ffill') -> pd.DataFrame:
    """
    Reamostra séries temporais para frequência específica
    
    Args:
        data: DataFrame com dados
        freq: Frequência de reamostragem ('D', 'W', 'M', 'Q', 'Y')
        method: Método de preenchimento ('ffill', 'bfill', 'mean', 'last')
    
    Returns:
        DataFrame reamostrado
    """
    if method == 'ffill':
        resampled = data.resample(freq).ffill()
    elif method == 'bfill':
        resampled = data.resample(freq).bfill()
    elif method == 'mean':
        resampled = data.resample(freq).mean()
    elif method == 'last':
        resampled = data.resample(freq).last()
    else:
        raise ValueError(f"Método {method} não suportado")
    
    return resampled

def calculate_drawdown(prices: pd.Series) -> pd.DataFrame:
    """
    Calcula drawdown de uma série de preços
    
    Args:
        prices: Série de preços
    
    Returns:
        DataFrame com drawdown information
    """
    # Calcular máximo acumulado
    cumulative_max = prices.expanding().max()
    
    # Calcular drawdown
    drawdown = (prices - cumulative_max) / cumulative_max
    
    # Calcular drawdown máximo
    max_drawdown = drawdown.expanding().min()
    
    # Calcular duração do drawdown
    drawdown_start = (drawdown == 0).astype(int)
    drawdown_duration = drawdown_start.groupby(
        (drawdown_start != drawdown_start.shift()).cumsum()
    ).cumsum()
    
    result = pd.DataFrame({
        'price': prices,
        'cumulative_max': cumulative_max,
        'drawdown': drawdown,
        'max_drawdown': max_drawdown,
        'drawdown_duration': drawdown_duration
    })
    
    return result

def split_time_series(data: pd.DataFrame, 
                     test_size: float = 0.2,
                     validation_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Divide série temporal em train/validation/test
    
    Args:
        data: DataFrame com dados temporais
        test_size: Proporção de teste
        validation_size: Proporção de validação
    
    Returns:
        Tupla com (train, validation, test)
    """
    n = len(data)
    test_split = int(n * (1 - test_size))
    val_split = int(test_split * (1 - validation_size))
    
    train = data.iloc[:val_split]
    validation = data.iloc[val_split:test_split]
    test = data.iloc[test_split:]
    
    return train, validation, test

def print_data_summary(data: pd.DataFrame, name: str = "Dataset"):
    """
    Imprime resumo dos dados
    
    Args:
        data: DataFrame para resumir
        name: Nome do dataset
    """
    print(f"\n{'='*60}")
    print(f"RESUMO: {name}")
    print(f"{'='*60}")
    print(f"Período: {data.index[0]} a {data.index[-1]}")
    print(f"Dias: {len(data)}")
    print(f"Features: {len(data.columns)}")
    print(f"Memory: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
    print(f"\nTipos de dados:")
    print(data.dtypes.value_counts())
    
    if isinstance(data, pd.DataFrame):
        print(f"\nValores faltantes por coluna:")
        missing = data.isna().sum()
        if missing.sum() > 0:
            for col, count in missing[missing > 0].items():
                print(f"  {col}: {count} ({count/len(data)*100:.1f}%)")
        else:
            print("  Nenhum valor faltante")