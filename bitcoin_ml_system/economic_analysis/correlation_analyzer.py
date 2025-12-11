"""
ANALISADOR DE CORRELAÇÕES - Analisa relações entre Bitcoin e variáveis macro
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

class CorrelationAnalyzer:
    """
    Analisa correlações e relações causais entre Bitcoin e indicadores macro
    """
    
    def __init__(self, min_correlation_samples: int = 30):
        self.min_samples = min_correlation_samples
    
    def analyze_correlations(self, 
                            btc_series: pd.Series,
                            macro_data: pd.DataFrame,
                            max_lag: int = 30) -> Dict[str, Any]:
        """
        Analisa correlações entre Bitcoin e múltiplos indicadores macro
        
        Args:
            btc_series: Série temporal do preço Bitcoin
            macro_data: DataFrame com dados macro
            max_lag: Número máximo de lags para análise
            
        Returns:
            Análise completa de correlações
        """
        print(f"\n📈 Analisando correlações (max lag: {max_lag} dias)...")
        
        if btc_series is None or btc_series.empty:
            return {'error': 'Dados Bitcoin não disponíveis'}
        
        if macro_data is None or macro_data.empty:
            return {'error': 'Dados macro não disponíveis'}
        
        # 1. Alinhar séries temporais
        aligned_data = self._align_time_series(btc_series, macro_data)
        
        if aligned_data is None:
            return {'error': 'Falha ao alinhar séries temporais'}
        
        btc_aligned, macro_aligned = aligned_data
        
        # 2. Calcular retornos do Bitcoin
        btc_returns = btc_aligned.pct_change().dropna()
        
        # 3. Analisar cada indicador macro
        correlation_results = {}
        
        for indicator in macro_aligned.columns:
            try:
                macro_series = macro_aligned[indicator].dropna()
                
                if len(macro_series) < self.min_samples:
                    continue
                
                # Analisar correlação com diferentes lags
                indicator_analysis = self._analyze_indicator_correlation(
                    btc_returns, macro_series, indicator, max_lag
                )
                
                if indicator_analysis:
                    correlation_results[indicator] = indicator_analysis
                    
            except Exception as e:
                print(f"⚠️  Erro ao analisar {indicator}: {e}")
                continue
        
        # 4. Ordenar por correlação absoluta
        sorted_results = self._sort_correlation_results(correlation_results)
        
        # 5. Análise agregada
        aggregate_analysis = self._analyze_aggregate_correlations(correlation_results)
        
        print(f"✅ {len(correlation_results)} indicadores analisados")
        
        return {
            'correlations': sorted_results,
            'aggregate_analysis': aggregate_analysis,
            'btc_returns_stats': {
                'mean': float(btc_returns.mean()),
                'std': float(btc_returns.std()),
                'skew': float(btc_returns.skew()),
                'kurtosis': float(btc_returns.kurtosis())
            },
            'analysis_parameters': {
                'max_lag': max_lag,
                'min_samples': self.min_samples,
                'total_indicators': len(macro_aligned.columns),
                'indicators_analyzed': len(correlation_results)
            }
        }
    
    def _align_time_series(self, 
                          btc_series: pd.Series, 
                          macro_data: pd.DataFrame) -> Optional[Tuple]:
        """
        Alinha séries temporais para mesma frequência e período
        """
        try:
            # Garantir que são séries temporais
            btc_series = btc_series.copy()
            macro_data = macro_data.copy()
            
            # Converter para frequência diária se necessário
            if btc_series.index.freq is None:
                btc_series = btc_series.asfreq('D')
            
            # Para macro, usar frequência diária (preencher valores)
            macro_daily = macro_data.resample('D').ffill()
            
            # Encontrar período comum
            common_index = btc_series.index.intersection(macro_daily.index)
            
            if len(common_index) < self.min_samples:
                return None
            
            # Alinhar ambas as séries
            btc_aligned = btc_series.reindex(common_index).ffill().bfill()
            macro_aligned = macro_daily.reindex(common_index).ffill().bfill()
            
            return btc_aligned, macro_aligned
            
        except Exception as e:
            print(f"Erro no alinhamento: {e}")
            return None
    
    def _analyze_indicator_correlation(self,
                                      btc_returns: pd.Series,
                                      macro_series: pd.Series,
                                      indicator_name: str,
                                      max_lag: int) -> Optional[Dict]:
        """
        Analisa correlação entre Bitcoin e um indicador macro específico
        """
        # Garantir alinhamento
        common_index = btc_returns.index.intersection(macro_series.index)
        
        if len(common_index) < self.min_samples:
            return None
        
        btc_aligned = btc_returns.reindex(common_index).dropna()
        macro_aligned = macro_series.reindex(common_index).dropna()
        
        # Calcular mudanças no indicador macro
        macro_changes = macro_aligned.pct_change().dropna()
        
        # Encontrar índice comum após remover NaN
        final_index = btc_aligned.index.intersection(macro_changes.index)
        
        if len(final_index) < self.min_samples:
            return None
        
        btc_final = btc_aligned.reindex(final_index)
        macro_final = macro_changes.reindex(final_index)
        
        # Correlação contemporânea (lag 0)
        corr_contemp, pval_contemp = stats.pearsonr(btc_final, macro_final)
        
        # Analisar diferentes lags
        lag_analysis = []
        best_lag = 0
        best_corr = corr_contemp
        best_pval = pval_contemp
        
        for lag in range(-max_lag, max_lag + 1):
            if lag == 0:
                continue
            
            try:
                if lag > 0:
                    # Macro lidera Bitcoin
                    btc_shifted = btc_final.shift(-lag)
                    macro_original = macro_final
                    relationship = f'macro_leads_btc_by_{lag}_days'
                else:
                    # Bitcoin lidera macro
                    btc_original = btc_final
                    macro_shifted = macro_final.shift(abs(lag))
                    relationship = f'btc_leads_macro_by_{abs(lag)}_days'
                    btc_shifted = btc_original
                    macro_original = macro_shifted
                
                # Remover NaN resultante do shift
                valid_mask = btc_shifted.notna() & macro_original.notna()
                if valid_mask.sum() < self.min_samples:
                    continue
                
                corr, pval = stats.pearsonr(
                    btc_shifted[valid_mask], 
                    macro_original[valid_mask]
                )
                
                lag_analysis.append({
                    'lag': lag,
                    'correlation': float(corr),
                    'p_value': float(pval),
                    'relationship': relationship,
                    'significant': pval < 0.05
                })
                
                # Atualizar melhor lag se melhor correlação
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_pval = pval
                    best_lag = lag
                    
            except Exception as e:
                continue
        
        # Determinar direção da relação
        if best_lag > 0:
            direction = f'{indicator_name} → Bitcoin (lead: {best_lag} dias)'
            causal_inference = 'macro_leads_btc'
        elif best_lag < 0:
            direction = f'Bitcoin → {indicator_name} (lead: {abs(best_lag)} dias)'
            causal_inference = 'btc_leads_macro'
        else:
            direction = 'contemporânea'
            causal_inference = 'contemporaneous'
        
        # Calcular rolling correlation para ver estabilidade
        rolling_corr = btc_final.rolling(window=90).corr(macro_final)
        rolling_corr_mean = rolling_corr.mean() if len(rolling_corr) > 0 else 0
        rolling_corr_std = rolling_corr.std() if len(rolling_corr) > 0 else 0
        
        return {
            'indicator': indicator_name,
            'pearson_corr': float(corr_contemp),
            'pearson_pval': float(pval_contemp),
            'best_lag': best_lag,
            'best_corr': float(best_corr),
            'best_pval': float(best_pval),
            'direction': direction,
            'causal_inference': causal_inference,
            'significant': best_pval < 0.05,
            'correlation_strength': self._classify_correlation_strength(abs(best_corr)),
            'samples': len(final_index),
            'rolling_corr_mean': float(rolling_corr_mean),
            'rolling_corr_std': float(rolling_corr_std),
            'rolling_corr_stable': rolling_corr_std < 0.2 if rolling_corr_std else False,
            'lag_analysis': lag_analysis[:10]  # Top 10 lags
        }
    
    def _classify_correlation_strength(self, corr_abs: float) -> str:
        """Classifica força da correlação"""
        if corr_abs >= 0.7:
            return 'very_strong'
        elif corr_abs >= 0.5:
            return 'strong'
        elif corr_abs >= 0.3:
            return 'moderate'
        elif corr_abs >= 0.1:
            return 'weak'
        else:
            return 'very_weak_or_none'
    
    def _sort_correlation_results(self, results: Dict) -> List[Dict]:
        """Ordena resultados por correlação absoluta"""
        sorted_items = sorted(
            results.items(),
            key=lambda x: abs(x[1]['best_corr']),
            reverse=True
        )
        
        return [item[1] for item in sorted_items]
    
    def _analyze_aggregate_correlations(self, results: Dict) -> Dict[str, Any]:
        """Analisa padrões agregados nas correlações"""
        if not results:
            return {'error': 'Nenhum resultado para análise agregada'}
        
        # Estatísticas gerais
        all_corrs = [r['best_corr'] for r in results.values()]
        all_pvals = [r['best_pval'] for r in results.values()]
        
        # Contar direções
        macro_leads = sum(1 for r in results.values() if r['best_lag'] > 0)
        btc_leads = sum(1 for r in results.values() if r['best_lag'] < 0)
        contemporaneous = sum(1 for r in results.values() if r['best_lag'] == 0)
        
        # Contar significativos
        significant = sum(1 for r in results.values() if r['significant'])
        
        # Agrupar por categoria de indicador
        indicator_categories = {}
        for name, analysis in results.items():
            # Determinar categoria baseada no nome
            category = self._categorize_indicator(name)
            if category not in indicator_categories:
                indicator_categories[category] = []
            indicator_categories[category].append(analysis)
        
        # Analisar cada categoria
        category_analysis = {}
        for category, analyses in indicator_categories.items():
            cat_corrs = [a['best_corr'] for a in analyses]
            cat_significant = sum(1 for a in analyses if a['significant'])
            
            category_analysis[category] = {
                'count': len(analyses),
                'mean_correlation': float(np.mean(cat_corrs)) if cat_corrs else 0,
                'median_correlation': float(np.median(cat_corrs)) if cat_corrs else 0,
                'significant_count': cat_significant,
                'significant_percentage': cat_significant / len(analyses) * 100 if analyses else 0,
                'strong_correlations': sum(1 for c in cat_corrs if abs(c) >= 0.5)
            }
        
        return {
            'overall_stats': {
                'total_indicators': len(results),
                'mean_correlation': float(np.mean(all_corrs)) if all_corrs else 0,
                'median_correlation': float(np.median(all_corrs)) if all_corrs else 0,
                'std_correlation': float(np.std(all_corrs)) if all_corrs else 0,
                'significant_count': significant,
                'significant_percentage': significant / len(results) * 100 if results else 0,
                'macro_leads_count': macro_leads,
                'btc_leads_count': btc_leads,
                'contemporaneous_count': contemporaneous
            },
            'category_analysis': category_analysis,
            'top_correlations': self._get_top_correlations(results, n=5),
            'interpretation': self._generate_aggregate_interpretation(results)
        }
    
    def _categorize_indicator(self, indicator_name: str) -> str:
        """Categoriza indicador macro baseado no nome"""
        name_lower = indicator_name.lower()
        
        if any(x in name_lower for x in ['rate', 'funds', 'yield', 'spread']):
            return 'interest_rates'
        elif any(x in name_lower for x in ['cpi', 'infla', 'price']):
            return 'inflation'
        elif any(x in name_lower for x in ['unemp', 'employ', 'payroll']):
            return 'labor_market'
        elif any(x in name_lower for x in ['prod', 'gdp', 'sales', 'retail']):
            return 'economic_activity'
        elif any(x in name_lower for x in ['m2', 'money', 'supply']):
            return 'money_supply'
        elif any(x in name_lower for x in ['sentiment', 'confidence', 'vix']):
            return 'sentiment'
        elif any(x in name_lower for x in ['housing', 'start', 'houst']):
            return 'housing'
        else:
            return 'other'
    
    def _get_top_correlations(self, results: Dict, n: int = 5) -> List[Dict]:
        """Retorna as n maiores correlações"""
        sorted_results = self._sort_correlation_results(results)
        return sorted_results[:min(n, len(sorted_results))]
    
    def _generate_aggregate_interpretation(self, results: Dict) -> str:
        """Gera interpretação agregada das correlações"""
        if not results:
            return "Nenhuma correlação significativa encontrada"
        
        aggregate = self._analyze_aggregate_correlations(results)
        stats = aggregate['overall_stats']
        
        interpretation = "📊 Análise Agregada de Correlações:\n\n"
        
        # Correlação média
        mean_corr = stats['mean_correlation']
        if abs(mean_corr) > 0.3:
            interpretation += f"• Correlação média: {mean_corr:.3f} ({'positiva' if mean_corr > 0 else 'negativa'})\n"
        else:
            interpretation += f"• Correlação média fraca: {mean_corr:.3f}\n"
        
        # Significância
        sig_pct = stats['significant_percentage']
        if sig_pct > 50:
            interpretation += f"• {sig_pct:.1f}% dos indicadores com correlação significativa (alta relação)\n"
        elif sig_pct > 25:
            interpretation += f"• {sig_pct:.1f}% dos indicadores com correlação significativa (relação moderada)\n"
        else:
            interpretation += f"• Apenas {sig_pct:.1f}% dos indicadores com correlação significativa\n"
        
        # Direção das relações
        if stats['macro_leads_count'] > stats['btc_leads_count']:
            interpretation += "• Padrão: indicadores macro geralmente antecedem movimentos do Bitcoin\n"
        elif stats['btc_leads_count'] > stats['macro_leads_count']:
            interpretation += "• Padrão: Bitcoin geralmente antecede mudanças nos indicadores macro\n"
        else:
            interpretation += "• Sem padrão claro de liderança entre Bitcoin e macro\n"
        
        # Recomendação
        if sig_pct > 40 and abs(mean_corr) > 0.2:
            interpretation += "\n✅ RECOMENDAÇÃO: Incluir indicadores macro no modelo ML"
        else:
            interpretation += "\n⚠️  RECOMENDAÇÃO: Focar mais em análise técnica e dados próprios do Bitcoin"
        
        return interpretation
    
    def generate_correlation_report(self, correlation_analysis: Dict) -> Dict[str, Any]:
        """
        Gera relatório completo de correlações
        
        Args:
            correlation_analysis: Resultados de analyze_correlations
            
        Returns:
            Relatório formatado para dashboard
        """
        if 'error' in correlation_analysis:
            return {'error': correlation_analysis['error']}
        
        # Preparar dados para visualização
        top_correlations = correlation_analysis.get('correlations', [])[:10]
        aggregate = correlation_analysis.get('aggregate_analysis', {})
        
        # Formatar top correlações
        formatted_top = []
        for corr in top_correlations:
            formatted_top.append({
                'indicator': corr['indicator'],
                'correlation': f"{corr['best_corr']:.3f}",
                'lag': f"{corr['best_lag']} dias",
                'direction': corr['direction'],
                'significance': '✅' if corr['significant'] else '❌',
                'strength': corr['correlation_strength']
            })
        
        # Sumário executivo
        executive_summary = {
            'total_indicators_analyzed': len(correlation_analysis.get('correlations', [])),
            'significant_correlations': aggregate.get('overall_stats', {}).get('significant_count', 0),
            'average_correlation': aggregate.get('overall_stats', {}).get('mean_correlation', 0),
            'dominant_pattern': self._identify_dominant_pattern(aggregate),
            'ml_recommendation': 'INCLUIR' if aggregate.get('overall_stats', {}).get('significant_percentage', 0) > 40 else 'LIMITAR'
        }
        
        return {
            'executive_summary': executive_summary,
            'top_correlations': formatted_top,
            'category_analysis': aggregate.get('category_analysis', {}),
            'detailed_analysis': correlation_analysis,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _identify_dominant_pattern(self, aggregate: Dict) -> str:
        """Identifica padrão dominante nas correlações"""
        stats = aggregate.get('overall_stats', {})
        
        macro_leads = stats.get('macro_leads_count', 0)
        btc_leads = stats.get('btc_leads_count', 0)
        
        if macro_leads > btc_leads * 1.5:
            return 'MACRO_LEADS_BTC'
        elif btc_leads > macro_leads * 1.5:
            return 'BTC_LEADS_MACRO'
        else:
            return 'MIXED_OR_CONTEMPORANEOUS'