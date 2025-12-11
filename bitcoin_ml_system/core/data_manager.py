"""
GERENCIADOR CENTRAL DE DADOS - SINGLETON
Garante que todos os módulos usem a mesma base de dados
Evita múltiplas requisições às APIs
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import pickle
import os
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from core.config import config

@dataclass
class DataCache:
    """Estrutura para armazenamento em cache"""
    data: Any
    timestamp: datetime
    hash: str

class DataManager:
    """
    Singleton que gerencia todos os dados do sistema
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DataManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Inicializa o gerenciador de dados"""
        if not self._initialized:
            self._initialized = True
            self._cache: Dict[str, DataCache] = {}
            self._data_registry: Dict[str, Any] = {}
            self._last_update: Dict[str, datetime] = {}
            
            # Criar diretórios necessários
            self._create_directories()
            
            print("✅ DataManager inicializado (Singleton)")
    
    def _create_directories(self):
        """Cria diretórios necessários para o sistema"""
        paths = config.get_paths()
        for path_name, path in paths.items():
            if not os.path.exists(path):
                os.makedirs(path, exist_ok=True)
    
    def _get_cache_key(self, data_type: str, params: Dict) -> str:
        """Gera uma chave única para cache baseada nos parâmetros"""
        param_str = str(sorted(params.items()))
        key_string = f"{data_type}_{param_str}"
        return hashlib.md5(key_string.encode()).hexdigest()[:16]
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Verifica se o cache ainda é válido"""
        if cache_key not in self._cache:
            return False
        
        cache_duration = timedelta(hours=config.CACHE_DURATION_HOURS)
        cache_age = datetime.now() - self._cache[cache_key].timestamp
        
        return cache_age < cache_duration
    
    def get_data(self, 
                 data_type: str, 
                 params: Dict,
                 force_update: bool = False) -> Optional[Any]:
        """
        Método principal para obter dados
        
        Args:
            data_type: Tipo de dados ('btc_historical', 'macro', 'related_assets', etc.)
            params: Parâmetros para coleta dos dados
            force_update: Ignora cache e busca dados novos
        
        Returns:
            Dados solicitados ou None se erro
        """
        cache_key = self._get_cache_key(data_type, params)
        
        # Verificar cache
        if not force_update and self._is_cache_valid(cache_key):
            print(f"📦 Retornando {data_type} do cache")
            return self._cache[cache_key].data
        
        # Buscar dados
        data = self._fetch_data(data_type, params)
        
        if data is not None:
            # Salvar no cache
            self._cache[cache_key] = DataCache(
                data=data,
                timestamp=datetime.now(),
                hash=cache_key
            )
            
            # Registrar nos dados globais
            self._data_registry[data_type] = data
            self._last_update[data_type] = datetime.now()
            
            print(f"✅ {data_type} carregado e cacheados")
        
        return data
    
    def _fetch_data(self, data_type: str, params: Dict) -> Optional[Any]:
        """Busca dados da fonte apropriada"""
        try:
            if data_type == 'btc_historical':
                return self._fetch_btc_data(**params)
            elif data_type == 'macro':
                return self._fetch_macro_data(**params)
            elif data_type == 'related_assets':
                return self._fetch_related_assets(**params)
            elif data_type == 'fear_greed':
                return self._fetch_fear_greed(**params)
            else:
                raise ValueError(f"Tipo de dados desconhecido: {data_type}")
                
        except Exception as e:
            print(f"❌ Erro ao buscar {data_type}: {e}")
            return None
    
    def _fetch_btc_data(self, 
                       period: str = "5y",
                       interval: str = "1d") -> Optional[pd.DataFrame]:
        """Busca dados históricos do Bitcoin"""
        try:
            import yfinance as yf
            
            print(f"📥 Buscando Bitcoin: {period}, {interval}")
            
            btc = yf.Ticker(config.BTC_SYMBOL)
            df = btc.history(period=period, interval=interval)
            
            if df.empty:
                raise ValueError("Dados Bitcoin vazios")
            
            # Limpeza básica
            df = df.dropna()
            
            # Adicionar colunas calculadas
            df['Returns'] = df['Close'].pct_change()
            df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
            df['Volatility'] = df['Returns'].rolling(30).std() * np.sqrt(365)
            
            print(f"✅ Bitcoin: {len(df)} períodos, Preço: ${df['Close'].iloc[-1]:,.2f}")
            return df
            
        except Exception as e:
            print(f"❌ Erro Bitcoin: {e}")
            return None
    
    def _fetch_macro_data(self, 
                         years_back: int = 5,
                         indicators: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """Busca dados macroeconômicos do FRED"""
        try:
            from fredapi import Fred
            
            if indicators is None:
                indicators = list(config.MACRO_INDICATORS.values())
            
            print(f"📥 Buscando {len(indicators)} indicadores macro ({years_back} anos)")
            
            fred = Fred(api_key=config.FRED_API_KEY)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=years_back * 365)
            
            all_series = {}
            
            for series_id in indicators:
                try:
                    series = fred.get_series(
                        series_id,
                        observation_start=start_date.strftime('%Y-%m-%d'),
                        observation_end=end_date.strftime('%Y-%m-%d')
                    )
                    
                    if not series.empty:
                        # Converter para frequência diária (preencher valores)
                        series_daily = series.resample('D').ffill()
                        all_series[series_id] = series_daily
                        
                except Exception as e:
                    print(f"⚠️ Indicador {series_id} não disponível: {e}")
                    continue
            
            if not all_series:
                raise ValueError("Nenhum indicador macro coletado")
            
            # Criar DataFrame combinado
            df = pd.DataFrame(all_series)
            df.index = pd.to_datetime(df.index)
            
            # Preencher valores NaN
            df = df.ffill().bfill()
            
            print(f"✅ Macro: {df.shape[0]} períodos, {df.shape[1]} indicadores")
            return df
            
        except Exception as e:
            print(f"❌ Erro Macro: {e}")
            return None
    
    def _fetch_related_assets(self,
                             period: str = "5y",
                             interval: str = "1d") -> Optional[Dict[str, pd.DataFrame]]:
        """Busca dados de ativos relacionados"""
        try:
            import yfinance as yf
            
            assets_data = {}
            
            for asset_name, symbol in config.RELATED_ASSETS.items():
                try:
                    print(f"📥 Buscando {asset_name} ({symbol})")
                    
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period=period, interval=interval)
                    
                    if not df.empty:
                        assets_data[asset_name] = df
                        print(f"   ✅ {asset_name}: {len(df)} períodos")
                    else:
                        print(f"   ⚠️ {asset_name}: dados vazios")
                        
                except Exception as e:
                    print(f"   ⚠️ Erro em {asset_name}: {e}")
                    continue
            
            return assets_data
            
        except Exception as e:
            print(f"❌ Erro ativos relacionados: {e}")
            return None
    
    def _fetch_fear_greed(self, days: int = 365) -> Optional[pd.DataFrame]:
        """Busca Fear & Greed Index"""
        try:
            import requests
            
            url = f"https://api.alternative.me/fng/?limit={days}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    records = []
                    for item in data['data']:
                        records.append({
                            'date': pd.to_datetime(item['timestamp'], unit='s'),
                            'value': int(item['value']),
                            'classification': item.get('value_classification', 'Neutral')
                        })
                    
                    if records:
                        df = pd.DataFrame(records)
                        df.set_index('date', inplace=True)
                        df = df.sort_index()
                        
                        print(f"✅ Fear & Greed: {len(df)} dias")
                        return df
            
            print("⚠️ Fear & Greed não disponível")
            return None
            
        except Exception as e:
            print(f"⚠️ Erro Fear & Greed: {e}")
            return None
    
    # ========== MÉTODOS PÚBLICOS ==========
    
    def get_all_data(self, 
                    btc_period: str = "5y",
                    macro_years: int = 5,
                    assets_period: str = "5y") -> Dict[str, Any]:
        """
        Coleta todos os dados necessários para análise
        
        Returns:
            Dicionário com todos os dados organizados
        """
        print("\n" + "="*60)
        print("COLETA COMPLETA DE DADOS")
        print("="*60)
        
        data_package = {}
        
        # 1. Dados Bitcoin
        btc_data = self.get_data(
            data_type='btc_historical',
            params={'period': btc_period, 'interval': '1d'}
        )
        
        if btc_data is None:
            raise ValueError("Falha ao coletar dados Bitcoin")
        
        data_package['bitcoin'] = {
            'data': btc_data,
            'prices': btc_data['Close'].tolist(),
            'dates': btc_data.index.tolist(),
            'current_price': btc_data['Close'].iloc[-1]
        }
        
        # 2. Dados macro
        macro_data = self.get_data(
            data_type='macro',
            params={'years_back': macro_years}
        )
        
        if macro_data is not None:
            data_package['macro'] = macro_data
        
        # 3. Ativos relacionados
        assets_data = self.get_data(
            data_type='related_assets',
            params={'period': assets_period, 'interval': '1d'}
        )
        
        if assets_data is not None:
            data_package['related_assets'] = assets_data
        
        # 4. Fear & Greed
        fear_greed_data = self.get_data(
            data_type='fear_greed',
            params={'days': 365}
        )
        
        if fear_greed_data is not None:
            data_package['fear_greed'] = fear_greed_data
        
        # 5. Estatísticas
        data_package['metadata'] = {
            'collection_time': datetime.now(),
            'btc_periods': len(btc_data),
            'macro_indicators': len(macro_data.columns) if macro_data is not None else 0,
            'assets_count': len(assets_data) if assets_data is not None else 0
        }
        
        print(f"\n✅ Pacote de dados completo:")
        print(f"   - Bitcoin: {len(btc_data)} períodos")
        print(f"   - Macro: {len(macro_data.columns) if macro_data is not None else 0} indicadores")
        print(f"   - Ativos: {len(assets_data) if assets_data is not None else 0} ativos relacionados")
        
        return data_package
    
    def clear_cache(self, data_type: Optional[str] = None):
        """Limpa o cache"""
        if data_type:
            keys_to_remove = [k for k in self._cache.keys() if k.startswith(data_type)]
            for key in keys_to_remove:
                del self._cache[key]
            print(f"🗑️ Cache de {data_type} limpo")
        else:
            self._cache.clear()
            print("🗑️ Cache completo limpo")
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status atual do DataManager"""
        status = {
            'cache_size': len(self._cache),
            'data_types_cached': list(self._data_registry.keys()),
            'last_updates': {k: v.strftime('%Y-%m-%d %H:%M') 
                           for k, v in self._last_update.items()},
            'cache_stats': {
                'total_items': len(self._cache),
                'oldest_item': min([c.timestamp for c in self._cache.values()]) 
                               if self._cache else None
            }
        }
        return status
    
    def save_state(self, filename: str = "data_manager_state.pkl"):
        """Salva estado atual do DataManager"""
        try:
            state = {
                'cache': self._cache,
                'data_registry': self._data_registry,
                'last_update': self._last_update
            }
            
            path = os.path.join(config.get_paths()['data_cache'], filename)
            with open(path, 'wb') as f:
                pickle.dump(state, f)
            
            print(f"💾 Estado salvo em {path}")
            return True
            
        except Exception as e:
            print(f"❌ Erro ao salvar estado: {e}")
            return False
    
    def load_state(self, filename: str = "data_manager_state.pkl"):
        """Carrega estado anterior do DataManager"""
        try:
            path = os.path.join(config.get_paths()['data_cache'], filename)
            
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    state = pickle.load(f)
                
                self._cache = state.get('cache', {})
                self._data_registry = state.get('data_registry', {})
                self._last_update = state.get('last_update', {})
                
                print(f"📂 Estado carregado de {path}")
                return True
            else:
                print(f"⚠️ Arquivo de estado não encontrado: {path}")
                return False
                
        except Exception as e:
            print(f"❌ Erro ao carregar estado: {e}")
            return False

# Instância global do DataManager
data_manager = DataManager()