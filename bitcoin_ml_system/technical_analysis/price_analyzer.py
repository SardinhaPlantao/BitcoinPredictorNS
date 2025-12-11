"""
ANALISADOR TÉCNICO DE PREÇOS
Calcula indicadores técnicos para Bitcoin
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.signal import argrelextrema

from core.config import config
from utils.helpers import calculate_returns, calculate_volatility

class TechnicalAnalyzer:
    """
    Analisador técnico completo para Bitcoin
    """
    
    def __init__(self):
        """Inicializa o analisador técnico"""
        self.indicators_cache = {}
        self.signals_cache = {}
        
    def calculate_all_indicators(self, 
                                price_data: pd.DataFrame,
                                include_advanced: bool = True) -> pd.DataFrame:
        """
        Calcula todos os indicadores técnicos
        
        Args:
            price_data: DataFrame com preços (Open, High, Low, Close, Volume)
            include_advanced: Incluir indicadores avançados
        
        Returns:
            DataFrame com todos os indicadores
        """
        print(f"\n📊 Calculando indicadores técnicos (advanced={include_advanced})")
        
        if 'Close' not in price_data.columns:
            raise ValueError("DataFrame deve conter coluna 'Close'")
        
        indicators = pd.DataFrame(index=price_data.index)
        
        # 1. PREÇOS E RETORNOS
        indicators['close'] = price_data['Close']
        
        if 'Open' in price_data.columns:
            indicators['open'] = price_data['Open']
            indicators['high'] = price_data['High']
            indicators['low'] = price_data['Low']
            indicators['range'] = (price_data['High'] - price_data['Low']) / price_data['Close']
        
        # Retornos
        indicators['returns'] = calculate_returns(price_data['Close'], 'simple')
        indicators['log_returns'] = calculate_returns(price_data['Close'], 'log')
        
        # 2. MÉDIAS MÓVEIS
        for period in [5, 10, 20, 50, 100, 200]:
            indicators[f'sma_{period}'] = price_data['Close'].rolling(period).mean()
            indicators[f'ema_{period}'] = price_data['Close'].ewm(span=period, adjust=False).mean()
        
        # Distância das médias
        indicators['dist_sma_20'] = (price_data['Close'] / indicators['sma_20'] - 1) * 100
        indicators['dist_sma_50'] = (price_data['Close'] / indicators['sma_50'] - 1) * 100
        indicators['dist_sma_200'] = (price_data['Close'] / indicators['sma_200'] - 1) * 100
        
        # 3. BANDAS DE BOLLINGER
        bb_period = 20
        sma_20 = indicators['sma_20']
        std_20 = price_data['Close'].rolling(bb_period).std()
        
        indicators['bb_upper'] = sma_20 + 2 * std_20
        indicators['bb_lower'] = sma_20 - 2 * std_20
        indicators['bb_width'] = (indicators['bb_upper'] - indicators['bb_lower']) / sma_20
        indicators['bb_position'] = (price_data['Close'] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'])
        
        # 4. RSI (Relative Strength Index)
        indicators['rsi_14'] = self._calculate_rsi(price_data['Close'], period=14)
        indicators['rsi_7'] = self._calculate_rsi(price_data['Close'], period=7)
        indicators['rsi_28'] = self._calculate_rsi(price_data['Close'], period=28)
        
        # 5. MACD (Moving Average Convergence Divergence)
        macd_line, signal_line, histogram = self._calculate_macd(price_data['Close'])
        indicators['macd_line'] = macd_line
        indicators['macd_signal'] = signal_line
        indicators['macd_histogram'] = histogram
        
        # 6. VOLATILIDADE
        for period in [7, 14, 30, 60]:
            indicators[f'volatility_{period}'] = calculate_volatility(
                indicators['returns'], 
                window=period, 
                annualize=True
            )
        
        # 7. MOMENTUM
        for period in [7, 14, 30, 60]:
            indicators[f'momentum_{period}'] = (price_data['Close'] / price_data['Close'].shift(period) - 1) * 100
        
        # 8. VOLUME (se disponível)
        if 'Volume' in price_data.columns:
            indicators['volume'] = price_data['Volume']
            indicators['volume_sma_20'] = price_data['Volume'].rolling(20).mean()
            indicators['volume_ratio'] = price_data['Volume'] / indicators['volume_sma_20']
            
            # OBV (On-Balance Volume)
            indicators['obv'] = self._calculate_obv(
                price_data['Close'], 
                price_data['Volume']
            )
        
        if include_advanced:
            # 9. INDICADORES AVANÇADADOS
            indicators = self._calculate_advanced_indicators(indicators, price_data)
        
        # 10. SINAIS DE COMPRA/VENDA
        indicators = self._generate_trading_signals(indicators)
        
        # Preencher NaN
        indicators = indicators.ffill().bfill()
        
        # Salvar no cache
        self.indicators_cache = indicators
        
        print(f"✅ {len(indicators.columns)} indicadores calculados")
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        
        # Separar ganhos e perdas
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Médias móveis
        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Calcular RS e RSI
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calcula MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_obv(self, prices: pd.Series, volume: pd.Series) -> pd.Series:
        """Calcula On-Balance Volume"""
        obv = pd.Series(index=prices.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(prices)):
            if prices.iloc[i] > prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif prices.iloc[i] < prices.iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    def _calculate_advanced_indicators(self, 
                                     indicators: pd.DataFrame,
                                     price_data: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos avançados"""
        # ATR (Average True Range)
        if all(col in price_data.columns for col in ['High', 'Low', 'Close']):
            high_low = price_data['High'] - price_data['Low']
            high_close = abs(price_data['High'] - price_data['Close'].shift())
            low_close = abs(price_data['Low'] - price_data['Close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            indicators['atr_14'] = true_range.rolling(14).mean()
            indicators['atr_percent'] = indicators['atr_14'] / price_data['Close'] * 100
        
        # Stochastic Oscillator
        if all(col in price_data.columns for col in ['High', 'Low', 'Close']):
            low_14 = price_data['Low'].rolling(14).min()
            high_14 = price_data['High'].rolling(14).max()
            
            indicators['stoch_k'] = 100 * (price_data['Close'] - low_14) / (high_14 - low_14)
            indicators['stoch_d'] = indicators['stoch_k'].rolling(3).mean()
        
        # Williams %R
        if all(col in price_data.columns for col in ['High', 'Low']):
            highest_high = price_data['High'].rolling(14).max()
            lowest_low = price_data['Low'].rolling(14).min()
            
            indicators['williams_r'] = -100 * (highest_high - price_data['Close']) / (highest_high - lowest_low)
        
        # CCI (Commodity Channel Index)
        if all(col in price_data.columns for col in ['High', 'Low', 'Close']):
            typical_price = (price_data['High'] + price_data['Low'] + price_data['Close']) / 3
            sma_typical = typical_price.rolling(20).mean()
            mean_deviation = abs(typical_price - sma_typical).rolling(20).mean()
            
            indicators['cci'] = (typical_price - sma_typical) / (0.015 * mean_deviation)
        
        # ADX (Average Directional Index)
        if all(col in price_data.columns for col in ['High', 'Low', 'Close']):
            indicators['adx'] = self._calculate_adx(price_data)
        
        # Fibonacci Retracement Levels
        if len(price_data) > 100:
            recent_high = price_data['High'].rolling(100).max()
            recent_low = price_data['Low'].rolling(100).min()
            
            price_range = recent_high - recent_low
            
            indicators['fib_236'] = recent_high - 0.236 * price_range
            indicators['fib_382'] = recent_high - 0.382 * price_range
            indicators['fib_500'] = recent_high - 0.5 * price_range
            indicators['fib_618'] = recent_high - 0.618 * price_range
            indicators['fib_786'] = recent_high - 0.786 * price_range
        
        return indicators
    
    def _calculate_adx(self, price_data: pd.DataFrame) -> pd.Series:
        """Calcula ADX (Average Directional Index)"""
        high = price_data['High']
        low = price_data['Low']
        close = price_data['Close']
        
        # +DM e -DM
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Smoothing
        period = 14
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * pd.Series(plus_dm, index=price_data.index).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=price_data.index).rolling(period).mean() / atr
        
        # DX e ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def _generate_trading_signals(self, indicators: pd.DataFrame) -> pd.DataFrame:
        """Gera sinais de trading baseados em indicadores"""
        signals = pd.DataFrame(index=indicators.index)
        
        # 1. Sinal RSI
        signals['rsi_signal'] = 0
        signals.loc[indicators['rsi_14'] < 30, 'rsi_signal'] = 1  # Oversold -> Buy
        signals.loc[indicators['rsi_14'] > 70, 'rsi_signal'] = -1  # Overbought -> Sell
        
        # 2. Sinal MACD
        signals['macd_signal'] = 0
        macd_cross_up = (indicators['macd_line'] > indicators['macd_signal']) & \
                       (indicators['macd_line'].shift(1) <= indicators['macd_signal'].shift(1))
        macd_cross_down = (indicators['macd_line'] < indicators['macd_signal']) & \
                         (indicators['macd_line'].shift(1) >= indicators['macd_signal'].shift(1))
        
        signals.loc[macd_cross_up, 'macd_signal'] = 1
        signals.loc[macd_cross_down, 'macd_signal'] = -1
        
        # 3. Sinal Bollinger Bands
        signals['bb_signal'] = 0
        signals.loc[indicators['close'] < indicators['bb_lower'], 'bb_signal'] = 1
        signals.loc[indicators['close'] > indicators['bb_upper'], 'bb_signal'] = -1
        
        # 4. Sinal Média Móvel
        signals['ma_signal'] = 0
        golden_cross = (indicators['sma_50'] > indicators['sma_200']) & \
                      (indicators['sma_50'].shift(1) <= indicators['sma_200'].shift(1))
        death_cross = (indicators['sma_50'] < indicators['sma_200']) & \
                     (indicators['sma_50'].shift(1) >= indicators['sma_200'].shift(1))
        
        signals.loc[golden_cross, 'ma_signal'] = 1
        signals.loc[death_cross, 'ma_signal'] = -1
        
        # 5. Sinal de Momentum
        signals['momentum_signal'] = 0
        signals.loc[indicators['momentum_30'] > 10, 'momentum_signal'] = 1
        signals.loc[indicators['momentum_30'] < -10, 'momentum_signal'] = -1
        
        # 6. Sinal composto (média dos sinais)
        signal_columns = [col for col in signals.columns if col.endswith('_signal')]
        signals['composite_signal'] = signals[signal_columns].mean(axis=1)
        
        # Classificar sinal composto
        signals['signal_strength'] = pd.cut(
            signals['composite_signal'],
            bins=[-np.inf, -0.5, -0.2, 0.2, 0.5, np.inf],
            labels=['Strong Sell', 'Sell', 'Neutral', 'Buy', 'Strong Buy']
        )
        
        # Salvar sinais
        self.signals_cache = signals
        
        return pd.concat([indicators, signals], axis=1)
    
    def analyze_support_resistance(self, 
                                  price_data: pd.DataFrame,
                                  window: int = 20) -> Dict[str, Any]:
        """
        Identifica níveis de suporte e resistência
        
        Args:
            price_data: DataFrame com preços
            window: Janela para identificar extremos locais
        
        Returns:
            Dicionário com níveis identificados
        """
        if 'Close' not in price_data.columns:
            raise ValueError("DataFrame deve conter coluna 'Close'")
        
        prices = price_data['Close'].values
        
        # Encontrar mínimos locais (suportes)
        local_minima = argrelextrema(prices, np.less, order=window)[0]
        support_levels = prices[local_minima]
        
        # Encontrar máximos locais (resistências)
        local_maxima = argrelextrema(prices, np.greater, order=window)[0]
        resistance_levels = prices[local_maxima]
        
        # Filtrar níveis significativos
        current_price = prices[-1]
        price_range = current_price * 0.3  # 30% do preço atual
        
        significant_supports = [
            float(level) for level in support_levels 
            if current_price - price_range < level < current_price + price_range
        ]
        
        significant_resistances = [
            float(level) for level in resistance_levels 
            if current_price - price_range < level < current_price + price_range
        ]
        
        # Remover duplicatas próximas
        significant_supports = self._cluster_levels(significant_supports, tolerance=0.02)
        significant_resistances = self._cluster_levels(significant_resistances, tolerance=0.02)
        
        # Ordenar por proximidade ao preço atual
        significant_supports.sort(key=lambda x: abs(x - current_price))
        significant_resistances.sort(key=lambda x: abs(x - current_price))
        
        analysis = {
            'current_price': float(current_price),
            'supports': significant_supports[:5],  # Top 5 suportes mais próximos
            'resistances': significant_resistances[:5],  # Top 5 resistências mais próximas
            'nearest_support': float(significant_supports[0]) if significant_supports else None,
            'nearest_resistance': float(significant_resistances[0]) if significant_resistances else None,
            'support_distance': float((current_price - significant_supports[0]) / current_price * 100) if significant_supports else None,
            'resistance_distance': float((significant_resistances[0] - current_price) / current_price * 100) if significant_resistances else None
        }
        
        return analysis
    
    def _cluster_levels(self, levels: List[float], tolerance: float = 0.02) -> List[float]:
        """
        Agrupa níveis próximos para evitar duplicatas
        """
        if not levels:
            return []
        
        levels_sorted = sorted(levels)
        clusters = []
        current_cluster = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                current_cluster.append(level)
            else:
                clusters.append(np.mean(current_cluster))
                current_cluster = [level]
        
        clusters.append(np.mean(current_cluster))
        return clusters
    
    def calculate_trend_strength(self, 
                                price_data: pd.DataFrame,
                                period: int = 30) -> Dict[str, Any]:
        """
        Calcula força e direção da tendência
        
        Args:
            price_data: DataFrame com preços
            period: Período para análise de tendência
        
        Returns:
            Análise da tendência
        """
        if 'Close' not in price_data.columns:
            raise ValueError("DataFrame deve conter coluna 'Close'")
        
        prices = price_data['Close']
        
        # Regressão linear para determinar tendência
        x = np.arange(len(prices[-period:]))
        y = prices[-period:].values
        
        if len(y) < 10:
            return {
                'trend': 'indeterminate',
                'strength': 0,
                'slope': 0,
                'r_squared': 0
            }
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Determinar direção da tendência
        if slope > 0 and p_value < 0.05:
            trend = 'uptrend'
        elif slope < 0 and p_value < 0.05:
            trend = 'downtrend'
        else:
            trend = 'sideways'
        
        # Força da tendência (R²)
        strength = r_value ** 2
        
        analysis = {
            'trend': trend,
            'strength': float(strength),
            'slope': float(slope),
            'r_squared': float(r_value ** 2),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'daily_change_percent': float(slope / prices.iloc[-period] * 100),
            'period_days': period
        }
        
        return analysis
    
    def generate_technical_report(self, 
                                indicators: pd.DataFrame,
                                price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Gera relatório técnico completo
        
        Args:
            indicators: DataFrame com indicadores
            price_data: DataFrame com preços
        
        Returns:
            Relatório técnico
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {},
            'signals': {},
            'levels': {},
            'trend': {}
        }
        
        # Sumário de indicadores atuais
        if not indicators.empty:
            last_row = indicators.iloc[-1]
            
            report['summary'] = {
                'price': float(last_row.get('close', 0)),
                'rsi_14': float(last_row.get('rsi_14', 50)),
                'rsi_status': 'oversold' if last_row.get('rsi_14', 50) < 30 else 
                             'overbought' if last_row.get('rsi_14', 50) > 70 else 'neutral',
                'macd': float(last_row.get('macd_histogram', 0)),
                'macd_signal': 'bullish' if last_row.get('macd_histogram', 0) > 0 else 'bearish',
                'bb_position': float(last_row.get('bb_position', 0.5)),
                'bb_status': 'oversold' if last_row.get('bb_position', 0.5) < 0.2 else 
                            'overbought' if last_row.get('bb_position', 0.5) > 0.8 else 'neutral',
                'volatility_30d': float(last_row.get('volatility_30', 0)),
                'momentum_30d': float(last_row.get('momentum_30', 0))
            }
        
        # Sinais atuais
        if 'composite_signal' in indicators.columns:
            last_signal = indicators['composite_signal'].iloc[-1]
            signal_strength = indicators['signal_strength'].iloc[-1] if 'signal_strength' in indicators.columns else 'Neutral'
            
            report['signals'] = {
                'composite_signal': float(last_signal),
                'signal_strength': str(signal_strength),
                'rsi_signal': int(indicators['rsi_signal'].iloc[-1]) if 'rsi_signal' in indicators.columns else 0,
                'macd_signal': int(indicators['macd_signal'].iloc[-1]) if 'macd_signal' in indicators.columns else 0,
                'ma_signal': int(indicators['ma_signal'].iloc[-1]) if 'ma_signal' in indicators.columns else 0
            }
        
        # Suportes e resistências
        report['levels'] = self.analyze_support_resistance(price_data)
        
        # Análise de tendência
        report['trend'] = self.calculate_trend_strength(price_data)
        
        # Recomendação baseada em múltiplos fatores
        recommendation = self._generate_recommendation(report)
        report['recommendation'] = recommendation
        
        return report
    
    def _generate_recommendation(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gera recomendação baseada em análise técnica
        """
        score = 0
        reasons = []
        
        # Fator 1: RSI
        rsi = report['summary'].get('rsi_14', 50)
        if rsi < 30:
            score += 2  # Oversold -> positivo para compra
            reasons.append('RSI indica oversold')
        elif rsi > 70:
            score -= 2  # Overbought -> negativo
            reasons.append('RSI indica overbought')
        
        # Fator 2: MACD
        macd_signal = report['summary'].get('macd_signal', '')
        if macd_signal == 'bullish':
            score += 1
            reasons.append('MACD bullish')
        elif macd_signal == 'bearish':
            score -= 1
            reasons.append('MACD bearish')
        
        # Fator 3: Bollinger Bands
        bb_status = report['summary'].get('bb_status', 'neutral')
        if bb_status == 'oversold':
            score += 1
            reasons.append('Nas bandas inferiores de Bollinger')
        elif bb_status == 'overbought':
            score -= 1
            reasons.append('Nas bandas superiores de Bollinger')
        
        # Fator 4: Tendência
        trend = report['trend'].get('trend', 'sideways')
        if trend == 'uptrend':
            score += 1
            reasons.append('Em tendência de alta')
        elif trend == 'downtrend':
            score -= 1
            reasons.append('Em tendência de baixa')
        
        # Fator 5: Distância do suporte/resistência
        support_dist = report['levels'].get('support_distance')
        resistance_dist = report['levels'].get('resistance_distance')
        
        if support_dist is not None and resistance_dist is not None:
            if support_dist < 5:  # Próximo do suporte
                score += 1
                reasons.append('Próximo do suporte')
            elif resistance_dist < 5:  # Próximo da resistência
                score -= 1
                reasons.append('Próximo da resistência')
        
        # Determinar recomendação
        if score >= 3:
            recommendation = 'STRONG_BUY'
            color = 'green'
        elif score >= 1:
            recommendation = 'BUY'
            color = 'lightgreen'
        elif score >= -1:
            recommendation = 'NEUTRAL'
            color = 'yellow'
        elif score >= -3:
            recommendation = 'SELL'
            color = 'orange'
        else:
            recommendation = 'STRONG_SELL'
            color = 'red'
        
        return {
            'action': recommendation,
            'score': score,
            'reasons': reasons,
            'color': color,
            'confidence': min(0.95, max(0.3, 0.5 + abs(score) * 0.1))
        }

# Instância global do analisador técnico
technical_analyzer = TechnicalAnalyzer()