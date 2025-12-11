"""
ANALISADOR DE CICLOS ECONÔMICOS
Identifica fases do ciclo econômico usando dados do FRED
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from core.config import config
from utils.helpers import align_time_series, calculate_returns

class EconomicCycleAnalyzer:
    """
    Analisa ciclos econômicos usando múltiplos indicadores
    """
    
    def __init__(self):
        """Inicializa o analisador de ciclos"""
        self.cycle_history = {}
        self.current_phase = None
        self.phase_model = None
        self.scaler = StandardScaler()
        
    def identify_cycle_phases(self, 
                             macro_data: pd.DataFrame,
                             method: str = 'composite') -> pd.DataFrame:
        """
        Identifica fases do ciclo econômico
        
        Args:
            macro_data: DataFrame com dados macro
            method: Método de identificação ('composite', 'nber', 'yield_curve')
        
        Returns:
            DataFrame com fase do ciclo identificada
        """
        print(f"\n🔄 Identificando fases do ciclo econômico ({method})")
        
        if macro_data.empty:
            raise ValueError("Dados macro vazios")
        
        result = macro_data.copy()
        
        if method == 'composite':
            result['cycle_phase'] = self._identify_by_composite_index(macro_data)
        elif method == 'nber':
            result['cycle_phase'] = self._identify_by_nber_method(macro_data)
        elif method == 'yield_curve':
            result['cycle_phase'] = self._identify_by_yield_curve(macro_data)
        else:
            raise ValueError(f"Método {method} não suportado")
        
        # Adicionar labels descritivos
        result['cycle_phase_label'] = result['cycle_phase'].apply(
            lambda x: self._get_phase_label(x)
        )
        
        # Calcular duração da fase atual
        result = self._calculate_phase_duration(result)
        
        # Salvar histórico
        self.cycle_history['phases'] = result[['cycle_phase', 'cycle_phase_label']]
        
        print(f"✅ Fases identificadas: {result['cycle_phase_label'].iloc[-1]}")
        return result
    
    def _identify_by_composite_index(self, macro_data: pd.DataFrame) -> pd.Series:
        """
        Identifica fases usando índice composto
        """
        # Criar índice composto simplificado
        composite_components = []
        
        # 1. Produção industrial
        if 'INDPRO' in macro_data.columns:
            production_trend = macro_data['INDPRO'].rolling(12).mean()
            production_gap = (macro_data['INDPRO'] - production_trend) / production_trend * 100
            composite_components.append(production_gap)
        
        # 2. Desemprego (invertido)
        if 'UNRATE' in macro_data.columns:
            unemployment_trend = macro_data['UNRATE'].rolling(12).mean()
            unemployment_gap = (unemployment_trend - macro_data['UNRATE']) / unemployment_trend * 100
            composite_components.append(unemployment_gap)
        
        # 3. Sentimento do consumidor
        if 'UMCSENT' in macro_data.columns:
            sentiment_norm = (macro_data['UMCSENT'] - macro_data['UMCSENT'].mean()) / macro_data['UMCSENT'].std()
            composite_components.append(sentiment_norm)
        
        if not composite_components:
            # Fallback: usar yield curve
            return self._identify_by_yield_curve(macro_data)
        
        # Combinar componentes
        composite = pd.concat(composite_components, axis=1).mean(axis=1)
        
        # Normalizar
        composite_norm = (composite - composite.rolling(60).mean()) / composite.rolling(60).std()
        
        # Identificar fases baseadas no índice
        phases = pd.Series(2, index=composite_norm.index)  # Default: Expansão
        
        # Recessão: índice abaixo de -0.5
        phases[composite_norm < -0.5] = 0
        
        # Recuperação: índice entre -0.5 e 0
        phases[(composite_norm >= -0.5) & (composite_norm < 0)] = 1
        
        # Desaceleração: índice acima de 0.5
        phases[composite_norm > 0.5] = 3
        
        return phases
    
    def _identify_by_nber_method(self, macro_data: pd.DataFrame) -> pd.Series:
        """
        Identifica fases usando método similar ao NBER
        (National Bureau of Economic Research)
        """
        phases = pd.Series(2, index=macro_data.index)  # Default: Expansão
        
        # Regras baseadas em múltiplos indicadores
        recession_signals = 0
        
        # 1. Yield curve invertida
        if 'T10Y2Y' in macro_data.columns:
            yield_inverted = macro_data['T10Y2Y'] < 0
            recession_signals += yield_inverted.astype(int)
        
        # 2. Desemprego em alta
        if 'UNRATE' in macro_data.columns:
            unemployment_rising = macro_data['UNRATE'].diff(90) > 0.3
            recession_signals += unemployment_rising.astype(int)
        
        # 3. Produção industrial em queda
        if 'INDPRO' in macro_data.columns:
            production_falling = macro_data['INDPRO'].pct_change(90) < -0.01
            recession_signals += production_falling.astype(int)
        
        # 4. Sentimento em queda
        if 'UMCSENT' in macro_data.columns:
            sentiment_falling = macro_data['UMCSENT'].diff(90) < -5
            recession_signals += sentiment_falling.astype(int)
        
        # Determinar fases
        # Recessão: pelo menos 3 sinais
        phases[recession_signals >= 3] = 0
        
        # Recuperação: 1-2 sinais após recessão
        # (implementação simplificada)
        
        return phases
    
    def _identify_by_yield_curve(self, macro_data: pd.DataFrame) -> pd.Series:
        """
        Identifica fases baseado na yield curve
        """
        phases = pd.Series(2, index=macro_data.index)  # Default: Expansão
        
        if 'T10Y2Y' not in macro_data.columns:
            return phases
        
        yield_curve = macro_data['T10Y2Y']
        
        # Recessão: yield curve invertida por 3 meses
        yield_ma = yield_curve.rolling(90).mean()
        inverted = yield_ma < 0
        
        # Expansão: yield curve positiva e subindo
        rising = (yield_curve.diff(30) > 0) & (yield_curve > 0)
        
        # Desaceleração: yield curve positiva mas caindo
        falling = (yield_curve.diff(30) < 0) & (yield_curve > 0)
        
        phases[inverted] = 0  # Recessão
        phases[rising] = 2    # Expansão
        phases[falling] = 3   # Desaceleração
        
        return phases
    
    def _get_phase_label(self, phase_code: int) -> str:
        """Retorna label descritivo para a fase"""
        labels = {
            0: 'Recessão',
            1: 'Recuperação',
            2: 'Expansão',
            3: 'Desaceleração'
        }
        return labels.get(phase_code, 'Desconhecido')
    
    def _calculate_phase_duration(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula duração da fase atual
        """
        if 'cycle_phase' not in data.columns:
            return data
        
        data = data.copy()
        data['phase_duration'] = 0
        
        current_phase = None
        phase_start = None
        
        for i, (date, row) in enumerate(data.iterrows()):
            phase = row['cycle_phase']
            
            if phase != current_phase:
                current_phase = phase
                phase_start = date
                data.loc[date, 'phase_duration'] = 1
            else:
                if phase_start:
                    duration = (date - phase_start).days
                    data.loc[date, 'phase_duration'] = duration
        
        return data
    
    def analyze_current_cycle(self, 
                            macro_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analisa o ciclo econômico atual
        
        Args:
            macro_data: Dados macro atuais
        
        Returns:
            Análise detalhada do ciclo atual
        """
        # Identificar fases
        data_with_phases = self.identify_cycle_phases(macro_data)
        
        # Última fase
        current_phase = data_with_phases['cycle_phase'].iloc[-1]
        current_label = data_with_phases['cycle_phase_label'].iloc[-1]
        phase_duration = data_with_phases['phase_duration'].iloc[-1]
        
        # Calcular momentum do ciclo
        cycle_momentum = self._calculate_cycle_momentum(data_with_phases)
        
        # Identificar pontos de virada
        turning_points = self._identify_turning_points(data_with_phases)
        
        # Prever próxima fase
        next_phase_prediction = self._predict_next_phase(data_with_phases)
        
        analysis = {
            'current_phase': {
                'code': int(current_phase),
                'label': current_label,
                'duration_days': int(phase_duration)
            },
            'cycle_momentum': cycle_momentum,
            'turning_points': turning_points,
            'next_phase_prediction': next_phase_prediction,
            'confidence_score': self._calculate_confidence_score(data_with_phases),
            'key_drivers': self._identify_key_drivers(macro_data, current_phase),
            'analysis_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        self.current_phase = analysis
        return analysis
    
    def _calculate_cycle_momentum(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula momentum do ciclo econômico
        """
        momentum = {}
        
        # Usar índice composto se disponível
        if 'cycle_phase' in data.columns:
            # Calcular mudança na fase
            phase_changes = data['cycle_phase'].diff()
            
            momentum['phase_change_rate'] = float(phase_changes.iloc[-30:].mean())
            
            # Calcular aceleração
            if len(phase_changes) >= 60:
                momentum['phase_acceleration'] = float(
                    phase_changes.iloc[-30:].mean() - phase_changes.iloc[-60:-30].mean()
                )
        
        return momentum
    
    def _identify_turning_points(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identifica pontos de virada no ciclo
        """
        turning_points = []
        
        if 'cycle_phase' not in data.columns:
            return turning_points
        
        phases = data['cycle_phase'].values
        
        # Encontrar mudanças de fase
        changes = np.diff(phases)
        change_indices = np.where(changes != 0)[0]
        
        for idx in change_indices:
            if idx < len(data) - 1:
                date = data.index[idx + 1]
                from_phase = self._get_phase_label(phases[idx])
                to_phase = self._get_phase_label(phases[idx + 1])
                
                turning_points.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'from_phase': from_phase,
                    'to_phase': to_phase,
                    'direction': 'positive' if phases[idx + 1] > phases[idx] else 'negative'
                })
        
        return turning_points
    
    def _predict_next_phase(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Prediz a próxima fase do ciclo
        """
        prediction = {
            'next_phase': 'Desconhecido',
            'probability': 0.5,
            'expected_in_days': 90,
            'confidence': 'Baixa'
        }
        
        if 'cycle_phase' not in data.columns:
            return prediction
        
        current_phase = data['cycle_phase'].iloc[-1]
        phase_duration = data['phase_duration'].iloc[-1]
        
        # Lógica simplificada baseada na duração da fase
        avg_phase_duration = {
            0: 180,  # Recessão: ~6 meses
            1: 270,  # Recuperação: ~9 meses
            2: 540,  # Expansão: ~18 meses
            3: 180   # Desaceleração: ~6 meses
        }
        
        expected_duration = avg_phase_duration.get(current_phase, 180)
        
        if phase_duration > expected_duration * 0.8:
            # Fase madura, possibilidade de transição
            if current_phase in [0, 3]:  # Recessão ou Desaceleração
                next_phase = 1 if current_phase == 0 else 0  # Para Recuperação ou Recessão
            else:
                next_phase = current_phase + 1 if current_phase < 3 else 0
            
            prediction['next_phase'] = self._get_phase_label(next_phase)
            prediction['probability'] = min(0.8, phase_duration / expected_duration)
            prediction['expected_in_days'] = max(30, expected_duration - phase_duration)
            prediction['confidence'] = 'Alta' if prediction['probability'] > 0.7 else 'Média'
        
        return prediction
    
    def _calculate_confidence_score(self, data: pd.DataFrame) -> float:
        """
        Calcula score de confiança da análise
        """
        score = 0.7  # Base
        
        # Fator 1: Consistência dos indicadores
        if 'cycle_phase' in data.columns:
            recent_phases = data['cycle_phase'].iloc[-10:]
            consistency = 1 - (recent_phases.std() / 2)  # 0-1 score
            score *= 0.7 + 0.3 * consistency
        
        # Fator 2: Duração da fase atual
        if 'phase_duration' in data.columns:
            duration = data['phase_duration'].iloc[-1]
            # Fases muito curtas ou muito longas reduzem confiança
            if 30 < duration < 365:
                duration_factor = 1.0
            else:
                duration_factor = 0.7
            score *= duration_factor
        
        return min(0.95, max(0.3, score))
    
    def _identify_key_drivers(self, 
                             macro_data: pd.DataFrame, 
                             current_phase: int) -> List[str]:
        """
        Identifica os principais drivers da fase atual
        """
        drivers = []
        
        # Análise baseada na fase
        if current_phase == 0:  # Recessão
            if 'UNRATE' in macro_data.columns and macro_data['UNRATE'].iloc[-1] > 6:
                drivers.append('Alto Desemprego')
            
            if 'T10Y2Y' in macro_data.columns and macro_data['T10Y2Y'].iloc[-1] < 0:
                drivers.append('Yield Curve Invertida')
        
        elif current_phase == 2:  # Expansão
            if 'INDPRO' in macro_data.columns and macro_data['INDPRO'].pct_change(365).iloc[-1] > 0.02:
                drivers.append('Crescimento Industrial')
            
            if 'UMCSENT' in macro_data.columns and macro_data['UMCSENT'].iloc[-1] > 80:
                drivers.append('Alto Sentimento')
        
        return drivers if drivers else ['Fatores Múltiplos']
    
    def create_cycle_visualization(self, 
                                 data_with_phases: pd.DataFrame) -> Dict[str, Any]:
        """
        Cria visualizações do ciclo econômico
        
        Args:
            data_with_phases: DataFrame com fases identificadas
        
        Returns:
            Dicionário com dados para visualização
        """
        viz_data = {
            'timeline': [],
            'phases_distribution': {},
            'phase_durations': []
        }
        
        if 'cycle_phase_label' not in data_with_phases.columns:
            return viz_data
        
        # Preparar dados da timeline
        current_phase = None
        phase_start = None
        
        for date, row in data_with_phases.iterrows():
            phase = row['cycle_phase_label']
            
            if phase != current_phase:
                if current_phase is not None and phase_start is not None:
                    # Finalizar fase anterior
                    viz_data['timeline'].append({
                        'phase': current_phase,
                        'start': phase_start.strftime('%Y-%m-%d'),
                        'end': date.strftime('%Y-%m-%d'),
                        'duration': (date - phase_start).days
                    })
                
                current_phase = phase
                phase_start = date
        
        # Adicionar fase atual
        if current_phase is not None and phase_start is not None:
            viz_data['timeline'].append({
                'phase': current_phase,
                'start': phase_start.strftime('%Y-%m-%d'),
                'end': data_with_phases.index[-1].strftime('%Y-%m-%d'),
                'duration': (data_with_phases.index[-1] - phase_start).days
            })
        
        # Distribuição das fases
        phase_counts = data_with_phases['cycle_phase_label'].value_counts()
        for phase, count in phase_counts.items():
            viz_data['phases_distribution'][phase] = {
                'count': int(count),
                'percentage': float(count / len(data_with_phases) * 100)
            }
        
        # Durações das fases
        for phase_event in viz_data['timeline']:
            viz_data['phase_durations'].append({
                'phase': phase_event['phase'],
                'duration': phase_event['duration']
            })
        
        return viz_data

# Instância global do analisador de ciclos
cycle_analyzer = EconomicCycleAnalyzer()