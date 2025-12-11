"""
ANÁLISE DE VOLUME E MOMENTUM
Analisa volume, liquidez e momentum do mercado
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from scipy import stats

from core.config import config
from utils.helpers import calculate_returns

class VolumeAnalyzer:
    """
    Analisador de volume e momentum
    """
    
    def __init__(self):
        """Inicializa o analisador de volume"""
        self.volume_indicators = {}
        
    def analyze_volume(self, 
                      price_data: pd.DataFrame,
                      include_advanced: bool = True) -> pd.DataFrame:
        """
        Analisa volume e relacionamento preço-volume
        
        Args:
            price_data: DataFrame com preços e volume
            include_advanced: Incluir métricas avançadas
        
        Returns:
            DataFrame com análise de volume
        """
        print(f"\n📈 Analisando volume (advanced={include_advanced})")
        
        if 'Volume' not in price_data.columns or 'Close' not in price_data.columns:
            raise ValueError("DataFrame deve conter colunas 'Close' e 'Volume'")
        
        analysis = pd.DataFrame(index=price_data.index)
        
        # 1. VOLUME BÁSICO
        analysis['volume'] = price_data['Volume']
        analysis['volume_sma_20'] = price_data['Volume'].rolling(20).mean()
        analysis['volume_ratio'] = price_data['Volume'] / analysis['volume_sma_20']
        
        # 2. VOLUME RELATIVO
        analysis['volume_rank'] = price_data['Volume'].rolling(30).rank(pct=True)
        analysis['volume_zscore'] = (price_data['Volume'] - price_data['Volume'].rolling(30).mean()) / \
                                   price_data['Volume'].rolling(30).std()
        
        # 3. PREÇO-VOLUME RELATIONSHIP
        returns = calculate_returns(price_data['Close'], 'simple')
        analysis['price_volume_corr'] = returns.rolling(20).corr(price_data['Volume'])
        
        # Volume em dias de alta vs baixa
        analysis['up_volume'] = np.where(returns > 0, price_data['Volume'], 0)
        analysis['down_volume'] = np.where(returns < 0, price_data['Volume'], 0)
        
        analysis['up_down_volume_ratio'] = analysis['up_volume'].rolling(20).sum() / \
                                          analysis['down_volume'].rolling(20).sum().replace(0, np.nan)
        
        # 4. VOLUME PROFILE
        analysis = self._calculate_volume_profile(analysis, price_data)
        
        if include_advanced:
            # 5. ANÁLISE AVANÇADA DE VOLUME
            analysis = self._calculate_advanced_volume_metrics(analysis, price_data)
        
        # 6. SINAIS DE VOLUME
        analysis = self._generate_volume_signals(analysis)
        
        # Preencher NaN
        analysis = analysis.ffill().bfill()
        
        # Salvar no cache
        self.volume_indicators = analysis
        
        print(f"✅ {len(analysis.columns)} métricas de volume calculadas")
        return analysis
    
    def _calculate_volume_profile(self, 
                                analysis: pd.DataFrame,
                                price_data: pd.DataFrame) -> pd.DataFrame:
        """Calcula perfil de volume"""
        # Volume por faixa de preço (simplificado)
        if 'High' in price_data.columns and 'Low' in price_data.columns:
            price_range = price_data['High'] - price_data['Low']
            analysis['volume_per_price_unit'] = price_data['Volume'] / price_range.replace(0, np.nan)
        
        # Volume acumulado
        analysis['cumulative_volume'] = price_data['Volume'].cumsum()
        
        # Taxa de variação do volume
        analysis['volume_roc'] = price_data['Volume'].pct_change(periods=5)
        
        return analysis
    
    def _calculate_advanced_volume_metrics(self,
                                         analysis: pd.DataFrame,
                                         price_data: pd.DataFrame) -> pd.DataFrame:
        """Calcula métricas avançadas de volume"""
        # Volume-Weighted Average Price (VWAP)
        if all(col in price_data.columns for col in ['High', 'Low', 'Close']):
            typical_price = (price_data['High'] + price_data['Low'] + price_data['Close']) / 3
            analysis['vwap'] = (typical_price * price_data['Volume']).cumsum() / \
                              price_data['Volume'].cumsum()
            
            # Desvio do VWAP
            analysis['vwap_deviation'] = (price_data['Close'] - analysis['vwap']) / analysis['vwap'] * 100
        
        # Money Flow Index (MFI)
        if all(col in price_data.columns for col in ['High', 'Low', 'Close', 'Volume']):
            analysis['mfi'] = self._calculate_mfi(price_data)
        
        # Ease of Movement (EMV)
        if all(col in price_data.columns for col in ['High', 'Low', 'Volume']):
            analysis['emv'] = self._calculate_emv(price_data)
        
        # Volume Price Trend (VPT)
        if 'Volume' in price_data.columns:
            returns = calculate_returns(price_data['Close'], 'simple')
            analysis['vpt'] = (returns * price_data['Volume']).cumsum()
        
        # Negative Volume Index (NVI) e Positive Volume Index (PVI)
        analysis['nvi'], analysis['pvi'] = self._calculate_nvi_pvi(price_data)
        
        return analysis
    
    def _calculate_mfi(self, price_data: pd.DataFrame) -> pd.Series:
        """Calcula Money Flow Index"""
        typical_price = (price_data['High'] + price_data['Low'] + price_data['Close']) / 3
        money_flow = typical_price * price_data['Volume']
        
        # Positive and negative money flow
        typical_price_diff = typical_price.diff()
        positive_mf = np.where(typical_price_diff > 0, money_flow, 0)
        negative_mf = np.where(typical_price_diff < 0, money_flow, 0)
        
        # 14-period sums
        positive_mf_sum = pd.Series(positive_mf, index=price_data.index).rolling(14).sum()
        negative_mf_sum = pd.Series(negative_mf, index=price_data.index).rolling(14).sum()
        
        # Money Flow Ratio e MFI
        mfr = positive_mf_sum / negative_mf_sum.replace(0, np.nan)
        mfi = 100 - (100 / (1 + mfr))
        
        return mfi.fillna(50)
    
    def _calculate_emv(self, price_data: pd.DataFrame) -> pd.Series:
        """Calcula Ease of Movement"""
        high_low_avg = (price_data['High'] + price_data['Low']) / 2
        high_low_diff = price_data['High'] - price_data['Low']
        
        box_ratio = price_data['Volume'] / 1000000 / high_low_diff.replace(0, np.nan)
        distance_moved = high_low_avg.diff()
        
        emv = distance_moved / box_ratio
        emv_smoothed = emv.rolling(14).mean()
        
        return emv_smoothed
    
    def _calculate_nvi_pvi(self, price_data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """Calcula Negative Volume Index e Positive Volume Index"""
        returns = calculate_returns(price_data['Close'], 'simple')
        volume_change = price_data['Volume'].pct_change()
        
        # NVI: Só acumula quando volume diminui
        nvi = pd.Series(1000, index=price_data.index)  # Valor inicial
        for i in range(1, len(price_data)):
            if volume_change.iloc[i] < 0:
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + returns.iloc[i])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        # PVI: Só acumula quando volume aumenta
        pvi = pd.Series(1000, index=price_data.index)  # Valor inicial
        for i in range(1, len(price_data)):
            if volume_change.iloc[i] > 0:
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + returns.iloc[i])
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        
        return nvi, pvi
    
    def _generate_volume_signals(self, analysis: pd.DataFrame) -> pd.DataFrame:
        """Gera sinais baseados em volume"""
        signals = pd.DataFrame(index=analysis.index)
        
        # 1. Sinal de Volume Alto
        signals['high_volume_signal'] = 0
        signals.loc[analysis['volume_ratio'] > 2, 'high_volume_signal'] = 1
        
        # 2. Sinal de Acumulação/Distribuição
        signals['accumulation_signal'] = 0
        # Volume alto em dias de alta = acumulação
        high_volume_up = (analysis['volume_ratio'] > 1.5) & (analysis.index.isin(
            analysis[analysis['up_volume'] > analysis['down_volume']].index
        ))
        signals.loc[high_volume_up, 'accumulation_signal'] = 1
        
        # Volume alto em dias de baixa = distribuição
        high_volume_down = (analysis['volume_ratio'] > 1.5) & (analysis.index.isin(
            analysis[analysis['down_volume'] > analysis['up_volume']].index
        ))
        signals.loc[high_volume_down, 'accumulation_signal'] = -1
        
        # 3. Sinal MFI
        signals['mfi_signal'] = 0
        signals.loc[analysis['mfi'] < 20, 'mfi_signal'] = 1  # Oversold
        signals.loc[analysis['mfi'] > 80, 'mfi_signal'] = -1  # Overbought
        
        # 4. Sinal VWAP
        if 'vwap_deviation' in analysis.columns:
            signals['vwap_signal'] = 0
            signals.loc[analysis['vwap_deviation'] < -2, 'vwap_signal'] = 1  # Abaixo do VWAP
            signals.loc[analysis['vwap_deviation'] > 2, 'vwap_signal'] = -1  # Acima do VWAP
        
        # 5. Sinal composto de volume
        volume_signal_cols = [col for col in signals.columns if col.endswith('_signal')]
        if volume_signal_cols:
            signals['volume_composite_signal'] = signals[volume_signal_cols].mean(axis=1)
        
        return pd.concat([analysis, signals], axis=1)
    
    def analyze_market_liquidity(self,
                                price_data: pd.DataFrame,
                                window: int = 30) -> Dict[str, Any]:
        """
        Analisa liquidez do mercado
        
        Args:
            price_data: DataFrame com preços e volume
            window: Janela para análise
        
        Returns:
            Análise de liquidez
        """
        if 'Volume' not in price_data.columns or 'Close' not in price_data.columns:
            raise ValueError("DataFrame deve conter colunas 'Close' e 'Volume'")
        
        recent_data = price_data.iloc[-window:]
        
        # Métricas de liquidez
        avg_daily_volume = recent_data['Volume'].mean()
        volume_volatility = recent_data['Volume'].std() / avg_daily_volume
        
        # Volume por faixa de preço
        price_ranges = pd.cut(recent_data['Close'], bins=10)
        volume_by_price = recent_data.groupby(price_ranges)['Volume'].sum()
        
        # Impacto de volume no preço
        returns = calculate_returns(recent_data['Close'], 'simple')
        volume_impact_corr = returns.corr(recent_data['Volume'])
        
        # Depth ratio (simplificado)
        high_volume_days = (recent_data['Volume'] > recent_data['Volume'].rolling(5).mean() * 1.5).sum()
        low_volume_days = (recent_data['Volume'] < recent_data['Volume'].rolling(5).mean() * 0.5).sum()
        
        analysis = {
            'avg_daily_volume': float(avg_daily_volume),
            'volume_volatility': float(volume_volatility),
            'volume_impact_correlation': float(volume_impact_corr),
            'high_volume_days_ratio': float(high_volume_days / window),
            'low_volume_days_ratio': float(low_volume_days / window),
            'current_volume_ratio': float(recent_data['Volume'].iloc[-1] / avg_daily_volume),
            'volume_concentration': float(volume_by_price.max() / volume_by_price.sum() if volume_by_price.sum() > 0 else 0),
            'liquidity_score': self._calculate_liquidity_score(
                avg_daily_volume, 
                volume_volatility, 
                volume_impact_corr
            )
        }
        
        return analysis
    
    def _calculate_liquidity_score(self,
                                  avg_volume: float,
                                  volume_volatility: float,
                                  volume_impact: float) -> float:
        """
        Calcula score de liquidez (0-100)
        """
        score = 0
        
        # Fator 1: Volume absoluto
        # Bitcoin: > $10B volume diário = boa liquidez
        volume_score = min(100, avg_volume / 1000000000 * 10)  # $1B = 10 pontos
        
        # Fator 2: Estabilidade do volume
        volatility_score = max(0, 100 - (volume_volatility * 100))
        
        # Fator 3: Impacto do volume no preço
        # Baixa correlação = melhor liquidez (preço não é movido por volume)
        impact_score = max(0, 100 - (abs(volume_impact) * 100))
        
        # Média ponderada
        score = (volume_score * 0.4 + volatility_score * 0.3 + impact_score * 0.3)
        
        return min(100, max(0, score))
    
    def calculate_momentum_indicators(self,
                                     price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula indicadores de momentum
        
        Args:
            price_data: DataFrame com preços
        
        Returns:
            DataFrame com indicadores de momentum
        """
        if 'Close' not in price_data.columns:
            raise ValueError("DataFrame deve conter coluna 'Close'")
        
        momentum = pd.DataFrame(index=price_data.index)
        
        # 1. Momentum simples
        for period in [5, 10, 20, 30, 60]:
            momentum[f'momentum_{period}'] = (price_data['Close'] / price_data['Close'].shift(period) - 1) * 100
        
        # 2. Rate of Change (ROC)
        for period in [5, 10, 20]:
            momentum[f'roc_{period}'] = price_data['Close'].pct_change(period) * 100
        
        # 3. Momentum Oscillator
        momentum['momentum_oscillator'] = price_data['Close'] - price_data['Close'].shift(10)
        
        # 4. Williams %R (já calculado no price_analyzer, mas recalculamos se necessário)
        if all(col in price_data.columns for col in ['High', 'Low']):
            period = 14
            highest_high = price_data['High'].rolling(period).max()
            lowest_low = price_data['Low'].rolling(period).min()
            momentum['williams_r'] = -100 * (highest_high - price_data['Close']) / (highest_high - lowest_low)
        
        # 5. Stochastic Momentum Index (simplificado)
        if all(col in price_data.columns for col in ['High', 'Low']):
            period = 14
            high_low_range = price_data['High'].rolling(period).max() - price_data['Low'].rolling(period).min()
            close_low = price_data['Close'] - price_data['Low'].rolling(period).min()
            momentum['stochastic_momentum'] = (close_low / high_low_range.replace(0, np.nan)) * 100
        
        # 6. Acceleration
        momentum['acceleration'] = momentum['momentum_10'].diff()
        
        # 7. Momentum qualidade (R² da tendência)
        for period in [20, 50]:
            momentum[f'momentum_quality_{period}'] = self._calculate_momentum_quality(
                price_data['Close'], period
            )
        
        return momentum
    
    def _calculate_momentum_quality(self, prices: pd.Series, period: int) -> pd.Series:
        """
        Calcula qualidade do momentum (R² da regressão linear)
        """
        quality = pd.Series(index=prices.index, dtype=float)
        
        for i in range(period, len(prices)):
            window = prices.iloc[i-period:i]
            x = np.arange(len(window))
            y = window.values
            
            if len(y) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                quality.iloc[i] = r_value ** 2
            else:
                quality.iloc[i] = 0
        
        return quality
    
    def generate_volume_report(self,
                              volume_analysis: pd.DataFrame,
                              price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera relatório de volume e momentum
        
        Args:
            volume_analysis: DataFrame com análise de volume
            price_data: DataFrame com preços
        
        Returns:
            Relatório de volume
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'volume_metrics': {},
            'momentum_metrics': {},
            'liquidity_analysis': {},
            'signals': {}
        }
        
        if not volume_analysis.empty:
            last_row = volume_analysis.iloc[-1]
            
            # Métricas de volume atuais
            report['volume_metrics'] = {
                'current_volume': float(last_row.get('volume', 0)),
                'volume_ratio': float(last_row.get('volume_ratio', 1)),
                'volume_zscore': float(last_row.get('volume_zscore', 0)),
                'up_down_volume_ratio': float(last_row.get('up_down_volume_ratio', 1)),
                'mfi': float(last_row.get('mfi', 50)) if 'mfi' in last_row else 50,
                'vwap_deviation': float(last_row.get('vwap_deviation', 0)) if 'vwap_deviation' in last_row else 0
            }
        
        # Métricas de momentum
        momentum_indicators = self.calculate_momentum_indicators(price_data)
        if not momentum_indicators.empty:
            last_momentum = momentum_indicators.iloc[-1]
            
            report['momentum_metrics'] = {
                'momentum_10d': float(last_momentum.get('momentum_10', 0)),
                'momentum_30d': float(last_momentum.get('momentum_30', 0)),
                'roc_10d': float(last_momentum.get('roc_10', 0)),
                'williams_r': float(last_momentum.get('williams_r', -50)) if 'williams_r' in last_momentum else -50,
                'acceleration': float(last_momentum.get('acceleration', 0)) if 'acceleration' in last_momentum else 0,
                'momentum_quality_20d': float(last_momentum.get('momentum_quality_20', 0)) if 'momentum_quality_20' in last_momentum else 0
            }
        
        # Análise de liquidez
        report['liquidity_analysis'] = self.analyze_market_liquidity(price_data)
        
        # Sinais atuais
        if 'volume_composite_signal' in volume_analysis.columns:
            report['signals'] = {
                'volume_composite': float(volume_analysis['volume_composite_signal'].iloc[-1]),
                'high_volume': int(volume_analysis['high_volume_signal'].iloc[-1]) if 'high_volume_signal' in volume_analysis.columns else 0,
                'accumulation': int(volume_analysis['accumulation_signal'].iloc[-1]) if 'accumulation_signal' in volume_analysis.columns else 0,
                'mfi_signal': int(volume_analysis['mfi_signal'].iloc[-1]) if 'mfi_signal' in volume_analysis.columns else 0
            }
        
        return report

# Instância global do analisador de volume
volume_analyzer = VolumeAnalyzer()