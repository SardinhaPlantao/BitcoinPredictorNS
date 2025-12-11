"""
COLETOR DE DADOS YAHOO FINANCE
Coleta dados REAIS de Bitcoin e ativos relacionados
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from core.config import config
from utils.helpers import calculate_returns, calculate_volatility

class YahooDataCollector:
    """
    Coleta dados do Yahoo Finance com tratamento de erros robusto
    """
    
    def __init__(self):
        """Inicializa o coletor do Yahoo Finance"""
        self.session = None
        self._initialize_session()
    
    def _initialize_session(self):
        """Inicializa sessão para múltiplas requisições"""
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
        except:
            self.session = None
    
    def get_historical_data(self, 
                           symbol: str, 
                           period: str = "5y",
                           interval: str = "1d",
                           max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Obtém dados históricos para um símbolo
        
        Args:
            symbol: Símbolo (ex: BTC-USD, ^GSPC)
            period: Período histórico
            interval: Intervalo (1d, 1wk, 1mo)
            max_retries: Número máximo de tentativas
        
        Returns:
            DataFrame com dados históricos ou None
        """
        for attempt in range(max_retries):
            try:
                print(f"📥 Yahoo: {symbol} ({period}, {interval}) - Tentativa {attempt + 1}")
                
                ticker = yf.Ticker(symbol)
                
                # Configurar parâmetros de download
                download_params = {
                    'period': period,
                    'interval': interval,
                    'progress': False,
                    'timeout': 30
                }
                
                # Adicionar sessão se disponível
                if self.session:
                    download_params['session'] = self.session
                
                df = ticker.history(**download_params)
                
                if df.empty:
                    print(f"⚠️ Dados vazios para {symbol}")
                    continue
                
                # Validar dados mínimos
                if len(df) < 10:
                    print(f"⚠️ Poucos dados para {symbol}: {len(df)}")
                    continue
                
                # Limpeza básica
                df = self._clean_dataframe(df, symbol)
                
                # Adicionar colunas calculadas
                df = self._add_calculated_columns(df, symbol)
                
                print(f"✅ {symbol}: {len(df)} períodos, Preço: {df['Close'].iloc[-1]:.2f}")
                return df
                
            except Exception as e:
                print(f"⚠️ Erro na tentativa {attempt + 1} para {symbol}: {e}")
                
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # Backoff exponencial
        
        print(f"❌ Falha após {max_retries} tentativas para {symbol}")
        return None
    
    def _clean_dataframe(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Limpa e padroniza o DataFrame
        
        Args:
            df: DataFrame original
            symbol: Símbolo para identificar tipo de dados
        
        Returns:
            DataFrame limpo
        """
        df = df.copy()
        
        # Remover duplicatas
        df = df[~df.index.duplicated(keep='first')]
        
        # Preencher valores NaN
        df = df.ffill().bfill()
        
        # Remover colunas completamente vazias
        df = df.dropna(axis=1, how='all')
        
        # Garantir que temos as colunas mínimas necessárias
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in required_cols:
            if col not in df.columns:
                print(f"⚠️ Coluna {col} não encontrada em {symbol}")
                # Tentar encontrar coluna alternativa
                if col == 'Close':
                    # Para Bitcoin, algumas fontes usam 'Adj Close'
                    if 'Adj Close' in df.columns:
                        df['Close'] = df['Adj Close']
                    elif 'Price' in df.columns:
                        df['Close'] = df['Price']
                    else:
                        # Usar última coluna numérica
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            df['Close'] = df[numeric_cols[0]]
        
        # Garantir que o índice é datetime
        df.index = pd.to_datetime(df.index)
        
        # Ordenar por data
        df = df.sort_index()
        
        return df
    
    def _add_calculated_columns(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Adiciona colunas calculadas ao DataFrame
        
        Args:
            df: DataFrame com dados básicos
            symbol: Símbolo para identificar cálculos específicos
        
        Returns:
            DataFrame com colunas adicionais
        """
        if 'Close' not in df.columns:
            return df
        
        df = df.copy()
        
        # Retornos
        df['Returns'] = calculate_returns(df['Close'], method='simple')
        df['Log_Returns'] = calculate_returns(df['Close'], method='log')
        
        # Volatilidade (30 dias anualizada)
        df['Volatility_30d'] = calculate_volatility(
            df['Returns'], 
            window=30, 
            annualize=True
        )
        
        # Médias móveis
        for window in [7, 20, 50, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # RSI (simplificado)
        if len(df) >= 15:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Volume médio
        if 'Volume' in df.columns:
            df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # Adicionar metadados
        df['Symbol'] = symbol
        df['Data_Source'] = 'Yahoo Finance'
        
        return df
    
    def get_multiple_symbols(self, 
                            symbols: List[str],
                            period: str = "5y",
                            interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Obtém dados para múltiplos símbolos simultaneamente
        
        Args:
            symbols: Lista de símbolos
            period: Período histórico
            interval: Intervalo
        
        Returns:
            Dicionário com DataFrames por símbolo
        """
        results = {}
        
        for symbol in symbols:
            data = self.get_historical_data(symbol, period, interval)
            if data is not None:
                results[symbol] = data
            else:
                print(f"❌ Não foi possível obter dados para {symbol}")
        
        return results
    
    def get_bitcoin_with_related_assets(self, 
                                       period: str = "5y") -> Dict[str, Any]:
        """
        Obtém dados do Bitcoin e ativos relacionados
        
        Args:
            period: Período histórico
        
        Returns:
            Dicionário com todos os dados organizados
        """
        print(f"\n{'='*60}")
        print("COLETA BITCOIN E ATIVOS RELACIONADOS")
        print(f"{'='*60}")
        
        all_data = {}
        
        # 1. Bitcoin
        print("\n1. Coletando Bitcoin...")
        btc_data = self.get_historical_data(
            symbol=config.BTC_SYMBOL,
            period=period,
            interval="1d"
        )
        
        if btc_data is None:
            raise ValueError("Falha ao coletar dados do Bitcoin")
        
        all_data['bitcoin'] = {
            'data': btc_data,
            'current_price': btc_data['Close'].iloc[-1],
            'stats': self._calculate_asset_statistics(btc_data, 'Bitcoin')
        }
        
        # 2. Ativos relacionados
        print("\n2. Coletando ativos relacionados...")
        related_data = self.get_multiple_symbols(
            symbols=list(config.RELATED_ASSETS.values()),
            period=period,
            interval="1d"
        )
        
        if related_data:
            all_data['related_assets'] = related_data
            
            # Calcular estatísticas para cada ativo
            for symbol, data in related_data.items():
                asset_name = [k for k, v in config.RELATED_ASSETS.items() if v == symbol][0]
                all_data['related_assets_stats'][asset_name] = self._calculate_asset_statistics(data, asset_name)
        
        # 3. Correlações
        print("\n3. Calculando correlações...")
        correlations = self._calculate_correlations(btc_data, related_data)
        all_data['correlations'] = correlations
        
        print(f"\n✅ Coleta completa: {len(btc_data)} dias de dados")
        print(f"   Ativos relacionados: {len(related_data)}")
        
        return all_data
    
    def _calculate_asset_statistics(self, 
                                   data: pd.DataFrame, 
                                   asset_name: str) -> Dict[str, float]:
        """
        Calcula estatísticas para um ativo
        
        Args:
            data: DataFrame com dados do ativo
            asset_name: Nome do ativo
        
        Returns:
            Dicionário com estatísticas
        """
        if 'Close' not in data.columns:
            return {}
        
        stats = {
            'name': asset_name,
            'current_price': float(data['Close'].iloc[-1]),
            'period_start': data.index[0].strftime('%Y-%m-%d'),
            'period_end': data.index[-1].strftime('%Y-%m-%d'),
            'n_periods': len(data)
        }
        
        # Retornos
        if 'Returns' in data.columns:
            stats['mean_daily_return'] = float(data['Returns'].mean() * 100)
            stats['std_daily_return'] = float(data['Returns'].std() * 100)
            stats['total_return'] = float(((data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1) * 100)
            
            # Calcular Sharpe ratio (simplificado)
            risk_free_rate = 0.02 / 365  # 2% anual
            excess_returns = data['Returns'] - risk_free_rate
            if data['Returns'].std() > 0:
                stats['sharpe_ratio'] = float((excess_returns.mean() / data['Returns'].std()) * np.sqrt(365))
        
        # Volatilidade
        if 'Volatility_30d' in data.columns:
            stats['volatility_30d'] = float(data['Volatility_30d'].iloc[-1] * 100)
        
        # RSI
        if 'RSI_14' in data.columns:
            stats['rsi_14'] = float(data['RSI_14'].iloc[-1])
        
        # Drawdown máximo
        cumulative_max = data['Close'].expanding().max()
        drawdown = (data['Close'] - cumulative_max) / cumulative_max
        stats['max_drawdown'] = float(drawdown.min() * 100)
        
        return stats
    
    def _calculate_correlations(self, 
                               btc_data: pd.DataFrame,
                               assets_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """
        Calcula correlações entre Bitcoin e outros ativos
        
        Args:
            btc_data: DataFrame do Bitcoin
            assets_data: Dicionário com DataFrames de outros ativos
        
        Returns:
            Dicionário com correlações
        """
        correlations = {}
        
        if 'Returns' not in btc_data.columns:
            return correlations
        
        btc_returns = btc_data['Returns']
        
        for symbol, asset_data in assets_data.items():
            if 'Returns' not in asset_data.columns:
                continue
            
            # Alinhar séries temporais
            common_index = btc_returns.index.intersection(asset_data.index)
            
            if len(common_index) < 30:
                continue
            
            btc_aligned = btc_returns.reindex(common_index)
            asset_aligned = asset_data['Returns'].reindex(common_index)
            
            # Calcular correlação
            correlation = btc_aligned.corr(asset_aligned)
            
            if not pd.isna(correlation):
                asset_name = [k for k, v in config.RELATED_ASSETS.items() if v == symbol][0]
                correlations[asset_name] = float(correlation)
        
        # Ordenar por correlação absoluta
        correlations = dict(sorted(
            correlations.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        ))
        
        return correlations
    
    def validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Valida a qualidade dos dados coletados
        
        Args:
            data: DataFrame para validar
        
        Returns:
            Dicionário com métricas de qualidade
        """
        quality_report = {
            'valid': True,
            'issues': [],
            'metrics': {}
        }
        
        # Verificar se DataFrame não está vazio
        if data.empty:
            quality_report['valid'] = False
            quality_report['issues'].append('DataFrame vazio')
            return quality_report
        
        # Verificar número de linhas
        quality_report['metrics']['n_rows'] = len(data)
        
        if len(data) < 100:
            quality_report['issues'].append(f'Poucas linhas: {len(data)}')
        
        # Verificar valores NaN
        nan_counts = data.isna().sum()
        total_nan = nan_counts.sum()
        
        quality_report['metrics']['total_nan'] = int(total_nan)
        quality_report['metrics']['nan_percentage'] = float((total_nan / (len(data) * len(data.columns))) * 100)
        
        if total_nan > 0:
            quality_report['issues'].append(f'{total_nan} valores NaN encontrados')
        
        # Verificar colunas essenciais
        essential_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in essential_cols if col not in data.columns]
        
        if missing_cols:
            quality_report['valid'] = False
            quality_report['issues'].append(f'Colunas faltando: {missing_cols}')
        
        # Verificar preços positivos
        if 'Close' in data.columns:
            negative_prices = (data['Close'] <= 0).sum()
            if negative_prices > 0:
                quality_report['valid'] = False
                quality_report['issues'].append(f'{negative_prices} preços não-positivos')
        
        # Verificar datas
        date_diff = data.index.to_series().diff().dt.days
        irregular_gaps = (date_diff > 7).sum()
        
        if irregular_gaps > 0:
            quality_report['issues'].append(f'{irregular_gaps} gaps temporais irregulares')
        
        return quality_report

# Instância global do coletor Yahoo
yahoo_collector = YahooDataCollector()