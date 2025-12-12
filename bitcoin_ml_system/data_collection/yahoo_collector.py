"""
COLETOR DE DADOS YAHOO FINANCE - VERSÃO CORRIGIDA
Coleta dados REAIS de Bitcoin e ativos relacionados
Versão corrigida para yfinance 0.2.33
"""

import sys
import os
from pathlib import Path

# Adiciona o diretório pai ao PYTHONPATH para resolver imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent  # bitcoin_ml_system
sys.path.insert(0, str(parent_dir))

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

print(f"📦 yfinance version: {yf.__version__}")

# Configuração padrão (fallback se config não estiver disponível)
DEFAULT_CONFIG = {
    'BTC_SYMBOL': 'BTC-USD',
    'RELATED_ASSETS': {
        'Ethereum': 'ETH-USD',
        'Gold': 'GC=F',
        'S&P500': '^GSPC',
        'US_Dollar': 'DX-Y.NYB',
        'NASDAQ': '^IXIC',
        'Tesla': 'TSLA',
        'Microsoft': 'MSFT'
    }
}

# Tenta importar config, senão usa DEFAULT_CONFIG
try:
    from core.config import config as app_config
    print("✅ Configuração importada do core.config")
    
    # Garantir que config tenha os atributos necessários
    if not hasattr(app_config, 'BTC_SYMBOL'):
        app_config.BTC_SYMBOL = DEFAULT_CONFIG['BTC_SYMBOL']
    if not hasattr(app_config, 'RELATED_ASSETS'):
        app_config.RELATED_ASSETS = DEFAULT_CONFIG['RELATED_ASSETS']
        
except ImportError as e:
    print(f"⚠️ Não foi possível importar core.config: {e}")
    print("⚠️ Usando configuração padrão")
    
    # Cria um objeto config com atributos
    class DefaultConfig:
        BTC_SYMBOL = DEFAULT_CONFIG['BTC_SYMBOL']
        RELATED_ASSETS = DEFAULT_CONFIG['RELATED_ASSETS']
    
    app_config = DefaultConfig()


class YahooDataCollector:
    """
    Coleta dados do Yahoo Finance com tratamento de erros robusto
    Versão corrigida para yfinance 0.2.33
    """
    
    def __init__(self, use_session: bool = True):
        """Inicializa o coletor do Yahoo Finance"""
        self.use_session = use_session
        self.session = None
        self._user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        
        if use_session:
            self._initialize_session()
    
    def _initialize_session(self):
        """Inicializa sessão para múltiplas requisições"""
        try:
            import requests
            self.session = requests.Session()
            self.session.headers.update({'User-Agent': self._user_agent})
            print("✅ Sessão HTTP inicializada")
        except Exception as e:
            print(f"⚠️ Não foi possível inicializar sessão: {e}")
            self.session = None
    
    def get_historical_data(self, 
                           symbol: str, 
                           period: str = "1mo",
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
                print(f"📥 Yahoo: {symbol} ({period}, {interval}) - Tentativa {attempt + 1}/{max_retries}")
                
                # Configurar parâmetros - REMOVIDO 'progress' que causa erro
                ticker = yf.Ticker(symbol)
                
                # Ajustar intervalo se necessário
                if period in ["5y", "10y", "max"] and interval in ["1m", "2m", "5m", "15m", "30m", "1h"]:
                    print(f"⚠️ Intervalo {interval} não suportado para {period}. Usando '1d'")
                    interval = "1d"
                
                # CORREÇÃO: Removido o parâmetro 'progress' que causa erro
                # Também removido 'session' que pode causar problemas
                download_params = {
                    'period': period,
                    'interval': interval,
                    'auto_adjust': True,
                    'actions': True  # Inclui dividends e stock splits
                }
                
                # Baixar dados
                df = ticker.history(**download_params)
                
                if df.empty:
                    print(f"⚠️ Dados vazios para {symbol}")
                    
                    # Tentar método alternativo
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)  # Backoff exponencial
                        continue
                    
                    # Tentar com yf.download como fallback
                    if attempt == max_retries - 1:
                        print("🔄 Tentando método alternativo de download...")
                        try:
                            # Para yf.download, podemos usar progress
                            df = yf.download(
                                symbol, 
                                period=period, 
                                interval=interval,
                                progress=False,
                                auto_adjust=True
                            )
                            if df.empty:
                                return None
                        except Exception as e2:
                            print(f"⚠️ Fallback também falhou: {e2}")
                            return None
                
                # Validar dados mínimos
                if len(df) < 5:
                    print(f"⚠️ Poucos dados para {symbol}: {len(df)} linhas")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                        continue
                
                # Limpeza básica
                df = self._clean_dataframe(df, symbol)
                
                # Adicionar colunas calculadas
                df = self._add_calculated_columns(df, symbol)
                
                print(f"✅ {symbol}: {len(df)} períodos, Preço: ${df['Close'].iloc[-1]:,.2f}")
                return df
                
            except Exception as e:
                print(f"⚠️ Erro na tentativa {attempt + 1} para {symbol}: {str(e)[:100]}")
                
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"⏳ Aguardando {wait_time} segundos...")
                    time.sleep(wait_time)
                else:
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
        if df.empty:
            return df
        
        df = df.copy()
        
        # Remover duplicatas
        df = df[~df.index.duplicated(keep='first')]
        
        # Garantir que o índice é datetime
        df.index = pd.to_datetime(df.index)
        
        # Remover timezone se presente
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        
        # Ordenar por data
        df = df.sort_index()
        
        # Renomear colunas para padrão se necessário
        column_mapping = {
            'Adj Close': 'Close',
            'adjclose': 'Close',
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
        
        # Garantir colunas essenciais
        essential_cols = ['Open', 'High', 'Low', 'Close']
        available_cols = [col for col in essential_cols if col in df.columns]
        
        if len(available_cols) < 2:
            print(f"⚠️ Colunas insuficientes para {symbol}: {df.columns.tolist()}")
            # Tentar usar o que temos
            if 'Close' not in df.columns:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df['Close'] = df[numeric_cols[0]]
                    print(f"⚠️ {symbol}: Usando '{numeric_cols[0]}' como 'Close'")
        
        # Preencher valores NaN
        df = df.ffill().bfill()
        
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
        if df.empty or 'Close' not in df.columns:
            return df
        
        df = df.copy()
        
        # Retornos simples
        df['Returns'] = df['Close'].pct_change()
        
        # Retornos logarítmicos
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatilidade (30 dias anualizada)
        if len(df) >= 30:
            df['Volatility_30d'] = df['Returns'].rolling(30).std() * np.sqrt(365)
        
        # Médias móveis
        for window in [7, 20]:
            if len(df) >= window:
                df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Volume em USD (se Volume estiver disponível)
        if 'Volume' in df.columns and 'Close' in df.columns:
            df['Volume_USD'] = df['Volume'] * df['Close']
        
        # Adicionar metadados como atributos
        df.attrs['symbol'] = symbol
        df.attrs['last_updated'] = datetime.now()
        df.attrs['data_source'] = 'Yahoo Finance'
        
        return df
    
    def get_multiple_symbols(self, 
                            symbols: Dict[str, str],
                            period: str = "1mo",
                            interval: str = "1d") -> Dict[str, pd.DataFrame]:
        """
        Obtém dados para múltiplos símbolos
        
        Args:
            symbols: Dicionário {nome: símbolo}
            period: Período histórico
            interval: Intervalo
        
        Returns:
            Dicionário com DataFrames por nome
        """
        results = {}
        
        print(f"📥 Baixando {len(symbols)} símbolos...")
        
        # Primeiro tentar baixar tudo de uma vez
        try:
            # Tentar baixar tudo de uma vez
            symbols_list = list(symbols.values())
            symbols_str = ' '.join(symbols_list)
            
            print(f"🔧 Download conjunto de: {symbols_str}")
            
            # Para yf.download podemos usar progress
            data = yf.download(
                tickers=symbols_str,
                period=period,
                interval=interval,
                group_by='ticker',
                progress=True,
                auto_adjust=True,
                threads=True
            )
            
            # Processar cada símbolo
            for name, symbol in symbols.items():
                if isinstance(data.columns, pd.MultiIndex) and symbol in data.columns.levels[0]:
                    df = data[symbol].copy()
                    if not df.empty:
                        df = self._clean_dataframe(df, symbol)
                        df = self._add_calculated_columns(df, symbol)
                        results[name] = df
                        print(f"   ✅ {name}: {len(df)} períodos")
                    else:
                        print(f"   ⚠️ {name}: dados vazios no download conjunto")
                        # Tentar individualmente
                        df_single = self.get_historical_data(symbol, period, interval)
                        if df_single is not None:
                            results[name] = df_single
                else:
                    print(f"   ⚠️ {name}: símbolo não encontrado, tentando individualmente...")
                    df_single = self.get_historical_data(symbol, period, interval)
                    if df_single is not None:
                        results[name] = df_single
                        
        except Exception as e:
            print(f"⚠️ Erro no download conjunto: {e}")
            print("🔄 Revertendo para downloads individuais...")
            
            # Fallback: baixar um por um
            for name, symbol in symbols.items():
                print(f"   📥 Baixando {name} ({symbol})...")
                data = self.get_historical_data(symbol, period, interval)
                if data is not None:
                    results[name] = data
                    print(f"     ✅ {name}: {len(data)} períodos")
                else:
                    print(f"     ❌ Falha ao baixar {name}")
        
        return results
    
    def get_bitcoin_data_direct(self, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Tenta obter dados do Bitcoin usando método direto
        """
        print(f"\n🔍 Tentando método direto para Bitcoin...")
        
        # Tenta diferentes métodos
        methods = [
            self._try_direct_download,
            self._try_with_session,
            self._try_different_periods
        ]
        
        for method in methods:
            try:
                df = method(period, interval)
                if df is not None and not df.empty:
                    return df
            except Exception as e:
                print(f"⚠️ Método falhou: {e}")
                continue
        
        return None
    
    def _try_direct_download(self, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Tenta download direto"""
        print("   Método 1: Download direto")
        try:
            df = yf.download(
                'BTC-USD',
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            if not df.empty:
                df = self._clean_dataframe(df, 'BTC-USD')
                df = self._add_calculated_columns(df, 'BTC-USD')
            return df
        except Exception as e:
            print(f"     ❌ Falha: {e}")
            return None
    
    def _try_with_session(self, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Tenta com sessão personalizada"""
        print("   Método 2: Com sessão HTTP")
        try:
            import requests
            session = requests.Session()
            session.headers.update({'User-Agent': self._user_agent})
            
            ticker = yf.Ticker('BTC-USD', session=session)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            
            if not df.empty:
                df = self._clean_dataframe(df, 'BTC-USD')
                df = self._add_calculated_columns(df, 'BTC-USD')
            return df
        except Exception as e:
            print(f"     ❌ Falha: {e}")
            return None
    
    def _try_different_periods(self, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Tenta diferentes períodos"""
        print("   Método 3: Testando diferentes períodos")
        
        test_periods = ["1wk", "5d", "1d"]
        
        for test_period in test_periods:
            try:
                print(f"     Testando período: {test_period}")
                ticker = yf.Ticker('BTC-USD')
                df = ticker.history(period=test_period, interval='1d', auto_adjust=True)
                
                if not df.empty:
                    df = self._clean_dataframe(df, 'BTC-USD')
                    df = self._add_calculated_columns(df, 'BTC-USD')
                    print(f"     ✅ Sucesso com período {test_period}")
                    return df
            except Exception as e:
                print(f"     ❌ Falha com {test_period}: {e}")
                continue
        
        return None
    
    def get_bitcoin_data(self, period: str = "1mo", interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Obtém dados do Bitcoin
        
        Args:
            period: Período histórico
            interval: Intervalo
        
        Returns:
            DataFrame com dados do Bitcoin
        """
        print(f"\n{'='*50}")
        print(f"COLETA DE DADOS DO BITCOIN")
        print(f"{'='*50}")
        
        btc_symbol = getattr(app_config, 'BTC_SYMBOL', 'BTC-USD')
        
        # Primeiro tentar o método normal
        print("📥 Tentando método padrão...")
        df = self.get_historical_data(
            symbol=btc_symbol,
            period=period,
            interval=interval,
            max_retries=2  # Menos tentativas para ser mais rápido
        )
        
        # Se falhar, tentar métodos alternativos
        if df is None or df.empty:
            print("🔄 Método padrão falhou, tentando alternativas...")
            df = self.get_bitcoin_data_direct(period, interval)
        
        if df is not None and not df.empty:
            print(f"\n📊 RESUMO BITCOIN:")
            print(f"   Período: {df.index[0].date()} até {df.index[-1].date()}")
            print(f"   Dias: {len(df)}")
            print(f"   Preço atual: ${df['Close'].iloc[-1]:,.2f}")
            
            if 'Returns' in df.columns and len(df) > 1:
                print(f"   Variação 24h: {df['Returns'].iloc[-1]*100:.2f}%")
            
            if 'Volatility_30d' in df.columns:
                print(f"   Volatilidade (30d): {df['Volatility_30d'].iloc[-1]*100:.2f}%")
            
            # Calcular retorno total no período
            if len(df) > 1:
                total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
                print(f"   Retorno total: {total_return:.2f}%")
        else:
            print("❌ Não foi possível obter dados do Bitcoin")
        
        return df
    
    def get_related_assets(self, period: str = "1mo") -> Dict[str, pd.DataFrame]:
        """
        Obtém dados de ativos relacionados
        
        Args:
            period: Período histórico
        
        Returns:
            Dicionário com DataFrames por nome do ativo
        """
        print(f"\n{'='*50}")
        print(f"COLETA DE ATIVOS RELACIONADOS")
        print(f"{'='*50}")
        
        related_assets = getattr(app_config, 'RELATED_ASSETS', DEFAULT_CONFIG['RELATED_ASSETS'])
        
        assets_data = self.get_multiple_symbols(
            symbols=related_assets,
            period=period,
            interval="1d"
        )
        
        if assets_data:
            print(f"\n📊 RESUMO ATIVOS:")
            for name, data in assets_data.items():
                if not data.empty and 'Close' in data.columns:
                    current_price = data['Close'].iloc[-1]
                    symbol = related_assets.get(name, name)
                    print(f"   {name:15} ({symbol:10}): ${current_price:,.2f}")
        
        return assets_data
    
    def calculate_correlations(self, 
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
        
        if btc_data is None or btc_data.empty or 'Returns' not in btc_data.columns:
            print("⚠️ Dados do Bitcoin insuficientes para calcular correlações")
            return correlations
        
        btc_returns = btc_data['Returns']
        
        for name, asset_data in assets_data.items():
            if asset_data is None or asset_data.empty or 'Returns' not in asset_data.columns:
                continue
            
            # Alinhar séries temporais
            common_index = btc_returns.index.intersection(asset_data.index)
            
            if len(common_index) < 5:  # Reduzido para aceitar menos dados
                continue
            
            btc_aligned = btc_returns.loc[common_index]
            asset_aligned = asset_data['Returns'].loc[common_index]
            
            # Calcular correlação
            correlation = btc_aligned.corr(asset_aligned)
            
            if not pd.isna(correlation):
                correlations[name] = float(correlation)
        
        # Ordenar por correlação absoluta
        correlations = dict(sorted(
            correlations.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        ))
        
        return correlations
    
    def save_data(self, 
                 data: pd.DataFrame, 
                 filename: str,
                 directory: str = "data_cache") -> bool:
        """
        Salva dados em arquivo CSV
        
        Args:
            data: DataFrame para salvar
            filename: Nome do arquivo
            directory: Diretório para salvar
        
        Returns:
            True se salvo com sucesso
        """
        try:
            # Criar diretório se não existir
            os.makedirs(directory, exist_ok=True)
            
            # Definir caminho completo
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            filepath = os.path.join(directory, filename)
            
            # Salvar como CSV
            data.to_csv(filepath)
            
            print(f"💾 Dados salvos em: {filepath}")
            print(f"   Linhas: {len(data)}, Colunas: {len(data.columns)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar dados: {e}")
            return False
    
    def load_data(self, filename: str, directory: str = "data_cache") -> Optional[pd.DataFrame]:
        """
        Carrega dados de arquivo CSV
        
        Args:
            filename: Nome do arquivo
            directory: Diretório onde está o arquivo
        
        Returns:
            DataFrame carregado ou None
        """
        try:
            # Definir caminho completo
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            filepath = os.path.join(directory, filename)
            
            if not os.path.exists(filepath):
                print(f"⚠️ Arquivo não encontrado: {filepath}")
                return None
            
            # Carregar CSV
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
            
            print(f"📂 Dados carregados de: {filepath}")
            print(f"   Linhas: {len(df)}, Período: {df.index[0].date()} até {df.index[-1].date()}")
            
            return df
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {e}")
            return None

# Instância global do coletor Yahoo
yahoo_collector = YahooDataCollector()


# ========== FUNÇÃO PARA TESTE/RUN DIRETO ==========
def main():
    """
    Função principal para executar o coletor diretamente
    """
    print("🚀 INICIANDO COLETOR YAHOO FINANCE")
    print("=" * 50)
    
    # Criar instância do coletor
    collector = YahooDataCollector()
    
    # 1. Coletar dados do Bitcoin
    print("\n1. COLETANDO DADOS DO BITCOIN...")
    btc_data = collector.get_bitcoin_data(period="1mo", interval="1d")
    
    if btc_data is not None and not btc_data.empty:
        # Salvar dados
        collector.save_data(btc_data, "bitcoin_data")
        
        # Mostrar estatísticas
        print(f"\n📈 ESTATÍSTICAS BITCOIN:")
        if 'High' in btc_data.columns:
            print(f"   Preço mais alto: ${btc_data['High'].max():,.2f}")
        if 'Low' in btc_data.columns:
            print(f"   Preço mais baixo: ${btc_data['Low'].min():,.2f}")
        if 'Volume_USD' in btc_data.columns:
            print(f"   Volume médio diário: ${btc_data['Volume_USD'].mean():,.0f}")
    else:
        print("❌ Não foi possível coletar dados do Bitcoin")
    
    # 2. Coletar ativos relacionados
    print("\n2. COLETANDO ATIVOS RELACIONADOS...")
    assets_data = collector.get_related_assets(period="1mo")
    
    if assets_data and btc_data is not None and not btc_data.empty:
        # 3. Calcular correlações
        print("\n3. CALCULANDO CORRELAÇÕES...")
        correlations = collector.calculate_correlations(btc_data, assets_data)
        
        if correlations:
            print("\n🔗 CORRELAÇÕES COM BITCOIN:")
            for asset, corr in correlations.items():
                correlation_type = "Positiva" if corr > 0 else "Negativa"
                strength = "Forte" if abs(corr) > 0.5 else "Moderada" if abs(corr) > 0.3 else "Fraca"
                print(f"   {asset:15}: {corr:+.3f} ({strength} {correlation_type})")
    
    print("\n" + "=" * 50)
    print("✅ COLETA CONCLUÍDA!")
    
    return btc_data, assets_data


# Executar apenas se chamado diretamente
if __name__ == "__main__":
    try:
        btc_data, assets_data = main()
        
        # Verificar se temos dados
        if btc_data is not None and not btc_data.empty:
            print(f"\n🎯 DADOS COLETADOS COM SUCESSO!")
            print(f"   Bitcoin: {len(btc_data)} registros")
            if assets_data:
                print(f"   Ativos relacionados: {len(assets_data)}")
            
            # Mostrar últimos preços
            print(f"\n💰 ÚLTIMOS PREÇOS:")
            print(f"   Bitcoin: ${btc_data['Close'].iloc[-1]:,.2f}")
            
            for name, data in assets_data.items():
                if data is not None and not data.empty and 'Close' in data.columns:
                    print(f"   {name:15}: ${data['Close'].iloc[-1]:,.2f}")
        else:
            print("\n⚠️ Atenção: Não foi possível coletar dados do Bitcoin")
            print("   Mas coletamos dados de outros ativos para análise")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Coleta interrompida pelo usuário")
    except Exception as e:
        print(f"\n❌ ERRO NA EXECUÇÃO: {e}")
        import traceback
        traceback.print_exc()