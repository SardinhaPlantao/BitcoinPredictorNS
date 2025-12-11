"""
ANALISADOR DE CORRELAÇÕES MACRO-BITCOIN
Analisa relações entre indicadores macro e preço do Bitcoin
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

from core.config import config
from utils.helpers import calculate_returns, align_time_series

class BitcoinMacroCorrelator:
    """
    Analisa correlações entre Bitcoin e indicadores macro
    """
    
    def __init__(self):
        """Inicializa o correlator"""
        self.correlation_history = {}
        self.causal_relationships = {}
        
    def calculate_correlations(self,
                             btc_data: pd.DataFrame,
                             macro_data: pd.DataFrame,
                             method: str = 'pearson',
                             rolling_window: int = 90) -> pd.DataFrame:
        """
        Calcula correlações entre Bitcoin e indicadores macro
        
        Args:
            btc_data: DataFrame do Bitcoin
            macro_data: DataFrame com indicadores macro
            method: Método de correlação ('pearson', 'spearman', 'kendall')
            rolling_window: Janela para correlações rolling
        
        Returns:
            DataFrame com correlações
        """
        print(f"\n🔗 Calculando correlações Bitcoin-Macro ({method}, window={rolling_window})")
        
        if btc_data.empty or macro_data.empty:
            raise ValueError("Dados Bitcoin ou macro vazios")
        
        if 'Returns' not in btc_data.columns:
            btc_data = btc_data.copy()
            btc_data['Returns'] = calculate_returns(btc_data['Close'])
        
        # Alinhar séries temporais
        btc_returns = btc_data['Returns']
        aligned_data = self._align_time_series(btc_returns, macro_data)
        
        if aligned_data is None:
            raise ValueError("Não foi possível alinhar séries temporais")
        
        btc_aligned, macro_aligned = aligned_data
        
        # Calcular correlações
        correlations = {}
        p_values = {}
        
        for macro_col in macro_aligned.columns:
            # Remover NaN
            valid_mask = btc_aligned.notna() & macro_aligned[macro_col].notna()
            
            if valid_mask.sum() < 30:  # Mínimo 30 pontos
                continue
            
            btc_valid = btc_aligned[valid_mask]
            macro_valid = macro_aligned[macro_col][valid_mask]
            
            # Calcular correlação
            if method == 'pearson':
                corr, pval = stats.pearsonr(btc_valid, macro_valid)
            elif method == 'spearman':
                corr, pval = stats.spearmanr(btc_valid, macro_valid)
            elif method == 'kendall':
                corr, pval = stats.kendalltau(btc_valid, macro_valid)
            else:
                raise ValueError(f"Método {method} não suportado")
            
            correlations[macro_col] = corr
            p_values[macro_col] = pval
        
        # Criar DataFrame de resultados
        results = pd.DataFrame({
            'correlation': correlations,
            'p_value': p_values,
            'significant': pd.Series(p_values) < 0.05
        })
        
        results = results.sort_values('correlation', key=abs, ascending=False)
        
        # Calcular correlações rolling
        rolling_corrs = self._calculate_rolling_correlations(
            btc_aligned, macro_aligned, rolling_window
        )
        
        # Salvar resultados
        self.correlation_history = {
            'static': results,
            'rolling': rolling_corrs,
            'calculation_date': datetime.now().strftime('%Y-%m-%d')
        }
        
        print(f"✅ {len(results)} correlações calculadas")
        print(f"   Top 3 positivas: {results.head(3).index.tolist()}")
        print(f"   Top 3 negativas: {results.tail(3).index.tolist()}")
        
        return results
    
    def _align_time_series(self, 
                          btc_series: pd.Series,
                          macro_data: pd.DataFrame) -> Optional[Tuple[pd.Series, pd.DataFrame]]:
        """
        Alinha séries temporais do Bitcoin e macro
        """
        # Encontrar índice comum
        common_index = btc_series.index.intersection(macro_data.index)
        
        if len(common_index) < 30:
            print(f"⚠️ Poucos dados comuns: {len(common_index)}")
            return None
        
        # Alinhar séries
        btc_aligned = btc_series.reindex(common_index).ffill().bfill()
        macro_aligned = macro_data.reindex(common_index).ffill().bfill()
        
        return btc_aligned, macro_aligned
    
    def _calculate_rolling_correlations(self,
                                       btc_series: pd.Series,
                                       macro_data: pd.DataFrame,
                                       window: int = 90) -> pd.DataFrame:
        """
        Calcula correlações rolling
        """
        rolling_corrs = pd.DataFrame(index=btc_series.index)
        
        for col in macro_data.columns:
            # Calcular correlação rolling
            rolling_corr = btc_series.rolling(window).corr(macro_data[col])
            rolling_corrs[col] = rolling_corr
        
        return rolling_corrs
    
    def analyze_causal_relationships(self,
                                    btc_data: pd.DataFrame,
                                    macro_data: pd.DataFrame,
                                    max_lag: int = 5) -> Dict[str, Any]:
        """
        Analisa relações causais usando teste de Granger
        
        Args:
            btc_data: DataFrame do Bitcoin
            macro_data: DataFrame macro
            max_lag: Número máximo de lags para testar
        
        Returns:
            Dicionário com resultados de causalidade
        """
        print(f"\n🔍 Analisando causalidade de Granger (max_lag={max_lag})")
        
        if 'Returns' not in btc_data.columns:
            btc_data = btc_data.copy()
            btc_data['Returns'] = calculate_returns(btc_data['Close'])
        
        btc_returns = btc_data['Returns']
        
        # Alinhar séries
        aligned_data = self._align_time_series(btc_returns, macro_data)
        
        if aligned_data is None:
            print("⚠️ Não foi possível alinhar séries para análise de causalidade")
            return {}
        
        btc_aligned, macro_aligned = aligned_data
        
        causal_results = {}
        
        for macro_col in macro_aligned.columns:
            # Criar DataFrame para teste de Granger
            test_data = pd.DataFrame({
                'btc': btc_aligned,
                'macro': macro_aligned[macro_col]
            }).dropna()
            
            if len(test_data) < max_lag * 2:
                continue
            
            try:
                # Teste de causalidade de Granger
                granger_test = grangercausalitytests(
                    test_data[['btc', 'macro']].values,
                    maxlag=max_lag,
                    verbose=False
                )
                
                # Extrair resultados
                best_lag = None
                best_p_value = 1.0
                
                for lag, results in granger_test.items():
                    p_value = results[0]['ssr_ftest'][1]
                    
                    if p_value < best_p_value:
                        best_p_value = p_value
                        best_lag = lag
                
                # Determinar direção da causalidade
                if best_p_value < 0.05:
                    # Testar direção oposta
                    test_data_rev = pd.DataFrame({
                        'macro': macro_aligned[macro_col],
                        'btc': btc_aligned
                    }).dropna()
                    
                    granger_test_rev = grangercausalitytests(
                        test_data_rev[['macro', 'btc']].values,
                        maxlag=max_lag,
                        verbose=False
                    )
                    
                    best_p_value_rev = 1.0
                    for lag, results in granger_test_rev.items():
                        p_value_rev = results[0]['ssr_ftest'][1]
                        if p_value_rev < best_p_value_rev:
                            best_p_value_rev = p_value_rev
                    
                    if best_p_value_rev < 0.05:
                        direction = 'bidirectional'
                    else:
                        direction = f'{macro_col} → Bitcoin'
                else:
                    direction = 'no causality'
                
                causal_results[macro_col] = {
                    'p_value': float(best_p_value),
                    'best_lag': int(best_lag) if best_lag else None,
                    'direction': direction,
                    'significant': best_p_value < 0.05
                }
                
            except Exception as e:
                print(f"⚠️ Erro no teste de Granger para {macro_col}: {e}")
                continue
        
        # Salvar resultados
        self.causal_relationships = causal_results
        
        # Contar resultados significativos
        significant = sum(1 for r in causal_results.values() if r['significant'])
        
        print(f"✅ {len(causal_results)} testes realizados")
        print(f"   {significant} relações causais significativas")
        
        return causal_results
    
    def identify_leading_indicators(self,
                                   btc_data: pd.DataFrame,
                                   macro_data: pd.DataFrame,
                                   max_lag: int = 30) -> Dict[str, Any]:
        """
        Identifica indicadores que antecedem o Bitcoin
        
        Args:
            btc_data: DataFrame do Bitcoin
            macro_data: DataFrame macro
            max_lag: Número máximo de lags para testar
        
        Returns:
            Dicionário com indicadores líderes
        """
        print(f"\n🎯 Identificando indicadores líderes (max_lag={max_lag})")
        
        if 'Returns' not in btc_data.columns:
            btc_data = btc_data.copy()
            btc_data['Returns'] = calculate_returns(btc_data['Close'])
        
        btc_returns = btc_data['Returns']
        
        # Alinhar séries
        aligned_data = self._align_time_series(btc_returns, macro_data)
        
        if aligned_data is None:
            return {}
        
        btc_aligned, macro_aligned = aligned_data
        
        leading_indicators = {}
        
        for macro_col in macro_aligned.columns:
            best_correlation = 0
            best_lag = 0
            
            # Testar diferentes lags
            for lag in range(1, max_lag + 1):
                # Shift macro series
                macro_shifted = macro_aligned[macro_col].shift(lag)
                
                # Calcular correlação
                valid_mask = btc_aligned.notna() & macro_shifted.notna()
                
                if valid_mask.sum() < 30:
                    continue
                
                correlation = btc_aligned[valid_mask].corr(macro_shifted[valid_mask])
                
                if abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    best_lag = lag
            
            if abs(best_correlation) > 0.2:  # Correlação significativa
                leading_indicators[macro_col] = {
                    'correlation': float(best_correlation),
                    'optimal_lag': int(best_lag),
                    'direction': 'leading' if best_lag > 0 else 'lagging',
                    'strength': 'strong' if abs(best_correlation) > 0.4 else 'moderate'
                }
        
        # Ordenar por força da correlação
        leading_indicators = dict(sorted(
            leading_indicators.items(),
            key=lambda x: abs(x[1]['correlation']),
            reverse=True
        ))
        
        print(f"✅ {len(leading_indicators)} indicadores líderes identificados")
        
        return leading_indicators
    
    def create_correlation_report(self,
                                 correlation_results: pd.DataFrame,
                                 causal_results: Dict[str, Any],
                                 leading_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Cria relatório consolidado de correlações
        
        Args:
            correlation_results: DataFrame com correlações
            causal_results: Resultados de causalidade
            leading_indicators: Indicadores líderes
        
        Returns:
            Relatório consolidado
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_indicators_analyzed': len(correlation_results),
                'significant_correlations': correlation_results['significant'].sum(),
                'causal_relationships': sum(1 for r in causal_results.values() if r['significant']),
                'leading_indicators': len(leading_indicators)
            },
            'top_correlations': {
                'positive': [],
                'negative': []
            },
            'key_findings': []
        }
        
        # Top correlações positivas
        positive_corrs = correlation_results[correlation_results['correlation'] > 0]
        if not positive_corrs.empty:
            report['top_correlations']['positive'] = positive_corrs.head(5).to_dict('records')
        
        # Top correlações negativas
        negative_corrs = correlation_results[correlation_results['correlation'] < 0]
        if not negative_corrs.empty:
            report['top_correlations']['negative'] = negative_corrs.head(5).to_dict('records')
        
        # Principais achados
        findings = []
        
        # Achado 1: Correlação mais forte
        if not correlation_results.empty:
            strongest = correlation_results.iloc[0]
            findings.append({
                'type': 'strongest_correlation',
                'indicator': correlation_results.index[0],
                'correlation': float(strongest['correlation']),
                'interpretation': self._interpret_correlation(float(strongest['correlation']))
            })
        
        # Achado 2: Indicador líder mais forte
        if leading_indicators:
            strongest_leader = list(leading_indicators.items())[0]
            findings.append({
                'type': 'strongest_leading_indicator',
                'indicator': strongest_leader[0],
                'correlation': strongest_leader[1]['correlation'],
                'lag': strongest_leader[1]['optimal_lag'],
                'interpretation': f"Antecipa Bitcoin em {strongest_leader[1]['optimal_lag']} dias"
            })
        
        # Achado 3: Relação causal mais forte
        significant_causal = {k: v for k, v in causal_results.items() if v['significant']}
        if significant_causal:
            strongest_causal = list(significant_causal.items())[0]
            findings.append({
                'type': 'strongest_causal_relationship',
                'indicator': strongest_causal[0],
                'direction': strongest_causal[1]['direction'],
                'p_value': strongest_causal[1]['p_value']
            })
        
        report['key_findings'] = findings
        
        return report
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpreta o valor da correlação"""
        abs_corr = abs(correlation)
        
        if abs_corr >= 0.7:
            return "Correlação muito forte"
        elif abs_corr >= 0.5:
            return "Correlação forte"
        elif abs_corr >= 0.3:
            return "Correlação moderada"
        elif abs_corr >= 0.1:
            return "Correlação fraca"
        else:
            return "Correlação insignificante"
    
    def visualize_correlations(self,
                              btc_data: pd.DataFrame,
                              macro_data: pd.DataFrame,
                              top_n: int = 10) -> Dict[str, Any]:
        """
        Prepara dados para visualização de correlações
        
        Args:
            btc_data: DataFrame do Bitcoin
            macro_data: DataFrame macro
            top_n: Número de top indicadores para visualizar
        
        Returns:
            Dicionário com dados para visualização
        """
        viz_data = {
            'correlation_matrix': None,
            'top_correlations': [],
            'rolling_correlations': {}
        }
        
        # Calcular correlações
        correlation_results = self.calculate_correlations(btc_data, macro_data)
        
        if correlation_results.empty:
            return viz_data
        
        # Top correlações
        top_positive = correlation_results.nlargest(top_n // 2, 'correlation')
        top_negative = correlation_results.nsmallest(top_n // 2, 'correlation')
        
        for idx, (indicator, row) in enumerate(top_positive.iterrows()):
            viz_data['top_correlations'].append({
                'rank': idx + 1,
                'indicator': indicator,
                'correlation': float(row['correlation']),
                'p_value': float(row['p_value']),
                'direction': 'positive'
            })
        
        for idx, (indicator, row) in enumerate(top_negative.iterrows()):
            viz_data['top_correlations'].append({
                'rank': idx + 1,
                'indicator': indicator,
                'correlation': float(row['correlation']),
                'p_value': float(row['p_value']),
                'direction': 'negative'
            })
        
        # Correlações rolling
        if 'rolling' in self.correlation_history:
            rolling_data = self.correlation_history['rolling']
            
            # Selecionar alguns indicadores para visualização rolling
            sample_indicators = list(correlation_results.index[:5])
            
            for indicator in sample_indicators:
                if indicator in rolling_data.columns:
                    viz_data['rolling_correlations'][indicator] = {
                        'dates': rolling_data.index.strftime('%Y-%m-%d').tolist(),
                        'correlations': rolling_data[indicator].fillna(0).tolist()
                    }
        
        return viz_data

# Instância global do correlator
macro_correlator = BitcoinMacroCorrelator()