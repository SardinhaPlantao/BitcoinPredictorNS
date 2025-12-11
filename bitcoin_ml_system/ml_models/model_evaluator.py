"""
AVALIADOR DE MODELOS
Avalia performance de modelos e compara resultados
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    r2_score, explained_variance_score,
    mean_absolute_percentage_error
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """
    Sistema completo de avaliação de modelos
    """
    
    def __init__(self):
        """Inicializa o avaliador"""
        self.evaluation_results = {}
        self.benchmark_models = {}
        
    def evaluate_model(self,
                      y_true: np.ndarray,
                      y_pred: np.ndarray,
                      model_name: str = "Model",
                      dataset_type: str = "test") -> Dict[str, Any]:
        """
        Avalia um modelo individual
        
        Args:
            y_true: Valores reais
            y_pred: Valores previstos
            model_name: Nome do modelo
            dataset_type: Tipo de dataset ('train', 'test', 'val')
        
        Returns:
            Métricas de avaliação
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true e y_pred devem ter o mesmo tamanho")
        
        # Métricas básicas
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        evs = explained_variance_score(y_true, y_pred)
        
        # MAPE (com tratamento para zeros)
        try:
            mape = mean_absolute_percentage_error(y_true, y_pred)
        except:
            # Calcular MAPE manualmente
            mask = y_true != 0
            if np.sum(mask) > 0:
                mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
            else:
                mape = np.nan
        
        # Métricas de direção
        direction_accuracy = self._calculate_direction_accuracy(y_true, y_pred)
        
        # Métricas estatísticas
        bias = np.mean(y_pred - y_true)
        std_error = np.std(y_pred - y_true)
        
        # Teste de normalidade dos resíduos
        residuals = y_true - y_pred
        if len(residuals) > 3:
            _, normality_p = stats.shapiro(residuals)
            is_normal = normality_p > 0.05
        else:
            normality_p = np.nan
            is_normal = False
        
        # Auto-correlação dos resíduos
        if len(residuals) > 1:
            autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        else:
            autocorr = np.nan
        
        # Criar dict de resultados
        results = {
            'model_name': model_name,
            'dataset_type': dataset_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_samples': len(y_true),
            'metrics': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'explained_variance': float(evs),
                'mape': float(mape),
                'direction_accuracy': float(direction_accuracy),
                'bias': float(bias),
                'std_error': float(std_error)
            },
            'statistical_tests': {
                'normality_p': float(normality_p) if not np.isnan(normality_p) else None,
                'is_normal': is_normal,
                'residual_autocorrelation': float(autocorr) if not np.isnan(autocorr) else None
            },
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'residuals': residuals.tolist()
            }
        }
        
        # Salvar resultados
        key = f"{model_name}_{dataset_type}"
        self.evaluation_results[key] = results
        
        print(f"✅ {model_name} avaliado ({dataset_type})")
        print(f"   RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        print(f"   Direção: {direction_accuracy:.1%}")
        
        return results
    
    def _calculate_direction_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula acurácia na previsão da direção"""
        if len(y_true) < 2:
            return 0.0
        
        # Calcular direções
        true_directions = np.diff(y_true) > 0
        pred_directions = np.diff(y_pred) > 0
        
        # Calcular acurácia
        accuracy = np.mean(true_directions == pred_directions)
        
        return float(accuracy)
    
    def compare_models(self,
                      results_list: List[Dict[str, Any]],
                      metric: str = 'rmse') -> pd.DataFrame:
        """
        Compara múltiplos modelos
        
        Args:
            results_list: Lista de resultados de avaliação
            metric: Métrica para comparação
        
        Returns:
            DataFrame com comparação
        """
        comparison_data = []
        
        for result in results_list:
            model_name = result['model_name']
            dataset_type = result['dataset_type']
            metrics = result['metrics']
            
            comparison_data.append({
                'model': model_name,
                'dataset': dataset_type,
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'direction_accuracy': metrics['direction_accuracy'],
                'mape': metrics['mape'],
                'bias': metrics['bias']
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Ordenar pela métrica especificada
        if metric in df.columns:
            df = df.sort_values(metric, ascending=(metric != 'r2'))
        
        return df
    
    def perform_statistical_tests(self,
                                 results_dict: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Realiza testes estatísticos para comparar modelos
        
        Args:
            results_dict: Dicionário com resultados de modelos
        
        Returns:
            Resultados dos testes estatísticos
        """
        tests = {}
        
        # Coletar resíduos de cada modelo
        residuals_dict = {}
        for model_name, result in results_dict.items():
            if 'predictions' in result and 'residuals' in result['predictions']:
                residuals_dict[model_name] = np.array(result['predictions']['residuals'])
        
        if len(residuals_dict) < 2:
            return {"error": "Pelo menos 2 modelos necessários para testes"}
        
        # Teste t pareado para diferença de médias dos resíduos
        model_names = list(residuals_dict.keys())
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                res1 = residuals_dict[model1]
                res2 = residuals_dict[model2]
                
                # Garantir mesmo tamanho
                min_len = min(len(res1), len(res2))
                res1 = res1[:min_len]
                res2 = res2[:min_len]
                
                # Teste t pareado
                t_stat, p_value = stats.ttest_rel(res1, res2)
                
                # Teste Wilcoxon (não paramétrico)
                if len(res1) > 10:
                    w_stat, w_p_value = stats.wilcoxon(res1, res2)
                else:
                    w_stat, w_p_value = np.nan, np.nan
                
                test_key = f"{model1}_vs_{model2}"
                tests[test_key] = {
                    't_test': {
                        'statistic': float(t_stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    },
                    'wilcoxon_test': {
                        'statistic': float(w_stat) if not np.isnan(w_stat) else None,
                        'p_value': float(w_p_value) if not np.isnan(w_p_value) else None
                    },
                    'mean_diff': float(np.mean(res1 - res2)),
                    'std_diff': float(np.std(res1 - res2))
                }
        
        # Teste de Friedman para múltiplos modelos
        if len(residuals_dict) >= 3:
            residuals_matrix = []
            for model_name in model_names:
                residuals_matrix.append(residuals_dict[model_name][:100])  # Limitando a 100
            
            # Completar com NaN se necessário
            max_len = max(len(r) for r in residuals_matrix)
            for i in range(len(residuals_matrix)):
                if len(residuals_matrix[i]) < max_len:
                    residuals_matrix[i] = np.pad(
                        residuals_matrix[i],
                        (0, max_len - len(residuals_matrix[i])),
                        constant_values=np.nan
                    )
            
            residuals_matrix = np.column_stack(residuals_matrix)
            
            # Teste de Friedman
            friedman_stat, friedman_p = stats.friedmanchisquare(*residuals_matrix.T)
            
            tests['friedman_test'] = {
                'statistic': float(friedman_stat),
                'p_value': float(friedman_p),
                'significant': friedman_p < 0.05
            }
        
        return tests
    
    def calculate_confidence_intervals(self,
                                     y_true: np.ndarray,
                                     y_pred: np.ndarray,
                                     residuals: np.ndarray = None,
                                     confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Calcula intervalos de confiança para previsões
        
        Args:
            y_true: Valores reais
            y_pred: Valores previstos
            residuals: Resíduos (se None, calcula de y_true e y_pred)
            confidence_level: Nível de confiança
        
        Returns:
            Intervalos de confiança
        """
        if residuals is None:
            residuals = y_true - y_pred
        
        # Estatísticas dos resíduos
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Testar normalidade
        if len(residuals) > 3:
            _, normality_p = stats.shapiro(residuals)
            is_normal = normality_p > 0.05
        else:
            is_normal = False
        
        # Calcular intervalos
        if is_normal and len(residuals) > 1:
            # Intervalo normal
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            ci_lower = y_pred - z_score * std_residual
            ci_upper = y_pred + z_score * std_residual
            method = 'normal'
        else:
            # Intervalo por bootstrap
            ci_lower, ci_upper = self._bootstrap_confidence_interval(
                residuals, y_pred, confidence_level
            )
            method = 'bootstrap'
        
        # Calcular cobertura (quantas observações estão dentro do intervalo)
        in_interval = (y_true >= ci_lower) & (y_true <= ci_upper)
        coverage = np.mean(in_interval)
        
        # Largura média do intervalo
        mean_width = np.mean(ci_upper - ci_lower)
        
        return {
            'confidence_level': confidence_level,
            'method': method,
            'ci_lower': ci_lower.tolist(),
            'ci_upper': ci_upper.tolist(),
            'coverage': float(coverage),
            'mean_width': float(mean_width),
            'is_normal': is_normal,
            'residual_stats': {
                'mean': float(mean_residual),
                'std': float(std_residual),
                'skew': float(stats.skew(residuals)) if len(residuals) > 2 else np.nan,
                'kurtosis': float(stats.kurtosis(residuals)) if len(residuals) > 3 else np.nan
            }
        }
    
    def _bootstrap_confidence_interval(self,
                                      residuals: np.ndarray,
                                      predictions: np.ndarray,
                                      confidence_level: float = 0.95,
                                      n_bootstrap: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Calcula intervalo de confiança por bootstrap"""
        n_samples = len(predictions)
        bootstrap_predictions = []
        
        for _ in range(n_bootstrap):
            # Amostrar resíduos com reposição
            bootstrap_residuals = np.random.choice(
                residuals, 
                size=n_samples, 
                replace=True
            )
            
            # Gerar previsões bootstrap
            bootstrap_pred = predictions + bootstrap_residuals
            bootstrap_predictions.append(bootstrap_pred)
        
        bootstrap_predictions = np.array(bootstrap_predictions)
        
        # Calcular percentis
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
        ci_upper = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
        
        return ci_lower, ci_upper
    
    def create_evaluation_report(self,
                                evaluation_results: Dict[str, Any],
                                include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Cria relatório completo de avaliação
        
        Args:
            evaluation_results: Resultados da avaliação
            include_visualizations: Incluir dados para visualizações
        
        Returns:
            Relatório completo
        """
        report = {
            'summary': {},
            'model_comparison': {},
            'statistical_tests': {},
            'confidence_intervals': {},
            'recommendations': [],
            'visualization_data': {}
        }
        
        # Extrair métricas principais
        metrics_summary = []
        for key, result in evaluation_results.items():
            metrics = result.get('metrics', {})
            metrics_summary.append({
                'model': key,
                'rmse': metrics.get('rmse', np.nan),
                'mae': metrics.get('mae', np.nan),
                'r2': metrics.get('r2', np.nan),
                'direction_accuracy': metrics.get('direction_accuracy', np.nan)
            })
        
        report['summary'] = pd.DataFrame(metrics_summary).to_dict('records')
        
        # Comparação de modelos
        if len(evaluation_results) > 1:
            comparison_df = self.compare_models(list(evaluation_results.values()))
            report['model_comparison'] = comparison_df.to_dict('records')
            
            # Testes estatísticos
            tests = self.perform_statistical_tests(evaluation_results)
            report['statistical_tests'] = tests
        
        # Intervalos de confiança
        ci_data = {}
        for key, result in evaluation_results.items():
            if 'predictions' in result:
                y_true = np.array(result['predictions']['y_true'])
                y_pred = np.array(result['predictions']['y_pred'])
                
                ci = self.calculate_confidence_intervals(y_true, y_pred)
                ci_data[key] = ci
        
        report['confidence_intervals'] = ci_data
        
        # Recomendações
        recommendations = self._generate_recommendations(evaluation_results)
        report['recommendations'] = recommendations
        
        # Dados para visualização
        if include_visualizations:
            report['visualization_data'] = self._prepare_visualization_data(evaluation_results)
        
        return report
    
    def _generate_recommendations(self, evaluation_results: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas na avaliação"""
        recommendations = []
        
        if not evaluation_results:
            return ["Nenhum modelo avaliado"]
        
        # Encontrar melhor modelo por RMSE
        best_rmse = float('inf')
        best_model_rmse = None
        
        for key, result in evaluation_results.items():
            rmse = result['metrics']['rmse']
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_rmse = key
        
        recommendations.append(f"Melhor modelo por RMSE: {best_model_rmse} (RMSE={best_rmse:.4f})")
        
        # Verificar overfitting
        for key, result in evaluation_results.items():
            # Se tivermos dados de treino e teste, verificar diferença
            if 'train' in key and 'test' in key:
                train_result = evaluation_results.get(key.replace('test', 'train'))
                if train_result:
                    train_rmse = train_result['metrics']['rmse']
                    test_rmse = result['metrics']['rmse']
                    
                    if test_rmse > train_rmse * 1.5:
                        recommendations.append(f"Possível overfitting em {key}: RMSE treino={train_rmse:.4f}, teste={test_rmse:.4f}")
        
        # Verificar viés (bias)
        for key, result in evaluation_results.items():
            bias = result['metrics']['bias']
            if abs(bias) > 0.1:  # Threshold arbitrário
                recommendations.append(f"Modelo {key} tem viés significativo: {bias:.4f}")
        
        # Verificar acurácia de direção
        for key, result in evaluation_results.items():
            direction_acc = result['metrics']['direction_accuracy']
            if direction_acc < 0.5:
                recommendations.append(f"Modelo {key} tem baixa acurácia de direção: {direction_acc:.1%}")
            elif direction_acc > 0.7:
                recommendations.append(f"Modelo {key} tem excelente acurácia de direção: {direction_acc:.1%}")
        
        return recommendations
    
    def _prepare_visualization_data(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara dados para visualização"""
        viz_data = {
            'predictions': {},
            'residuals': {},
            'metrics_comparison': {}
        }
        
        # Dados de previsões e resíduos
        for key, result in evaluation_results.items():
            if 'predictions' in result:
                viz_data['predictions'][key] = {
                    'y_true': result['predictions']['y_true'],
                    'y_pred': result['predictions']['y_pred']
                }
                viz_data['residuals'][key] = result['predictions']['residuals']
        
        # Comparação de métricas
        metrics_df = self.compare_models(list(evaluation_results.values()))
        viz_data['metrics_comparison'] = metrics_df.to_dict('records')
        
        return viz_data
    
    def visualize_results(self, evaluation_results: Dict[str, Any], save_path: str = None):
        """
        Cria visualizações dos resultados
        
        Args:
            evaluation_results: Resultados da avaliação
            save_path: Path para salvar figuras
        """
        import matplotlib.pyplot as plt
        
        if not evaluation_results:
            print("⚠️ Nenhum resultado para visualizar")
            return
        
        # Criar figura
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Gráfico de previsões vs reais
        ax1 = plt.subplot(2, 3, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, len(evaluation_results)))
        
        for (key, result), color in zip(evaluation_results.items(), colors):
            if 'predictions' in result:
                y_true = result['predictions']['y_true']
                y_pred = result['predictions']['y_pred']
                
                # Plotar apenas 100 pontos para clareza
                n_points = min(100, len(y_true))
                indices = np.linspace(0, len(y_true)-1, n_points, dtype=int)
                
                ax1.scatter(
                    np.array(y_true)[indices],
                    np.array(y_pred)[indices],
                    alpha=0.6,
                    label=key,
                    color=color,
                    s=20
                )
        
        # Linha de referência (y=x)
        min_val = min([min(result['predictions']['y_true']) for result in evaluation_results.values()])
        max_val = max([max(result['predictions']['y_true']) for result in evaluation_results.values()])
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Ideal')
        
        ax1.set_xlabel('Valores Reais')
        ax1.set_ylabel('Valores Previstos')
        ax1.set_title('Previsões vs Reais')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Gráfico de resíduos
        ax2 = plt.subplot(2, 3, 2)
        
        for (key, result), color in zip(evaluation_results.items(), colors):
            if 'predictions' in result:
                residuals = result['predictions']['residuals']
                
                ax2.scatter(
                    range(len(residuals))[:100],  # Primeiros 100 pontos
                    residuals[:100],
                    alpha=0.6,
                    label=key,
                    color=color,
                    s=20
                )
        
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Amostra')
        ax2.set_ylabel('Resíduos')
        ax2.set_title('Distribuição dos Resíduos')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Histograma dos resíduos
        ax3 = plt.subplot(2, 3, 3)
        
        for (key, result), color in zip(evaluation_results.items(), colors):
            if 'predictions' in result:
                residuals = result['predictions']['residuals']
                
                ax3.hist(
                    residuals,
                    bins=30,
                    alpha=0.5,
                    label=key,
                    color=color,
                    density=True
                )
        
        ax3.set_xlabel('Resíduos')
        ax3.set_ylabel('Densidade')
        ax3.set_title('Distribuição dos Resíduos')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Comparação de métricas (RMSE)
        ax4 = plt.subplot(2, 3, 4)
        
        model_names = []
        rmse_values = []
        
        for key, result in evaluation_results.items():
            model_names.append(key)
            rmse_values.append(result['metrics']['rmse'])
        
        bars = ax4.bar(range(len(model_names)), rmse_values, color=colors)
        ax4.set_xlabel('Modelos')
        ax4.set_ylabel('RMSE')
        ax4.set_title('Comparação de RMSE')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, rmse_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 5. Comparação de R²
        ax5 = plt.subplot(2, 3, 5)
        
        r2_values = []
        for key, result in evaluation_results.items():
            r2_values.append(result['metrics']['r2'])
        
        bars = ax5.bar(range(len(model_names)), r2_values, color=colors)
        ax5.set_xlabel('Modelos')
        ax5.set_ylabel('R²')
        ax5.set_title('Comparação de R²')
        ax5.set_xticks(range(len(model_names)))
        ax5.set_xticklabels(model_names, rotation=45, ha='right')
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, r2_values):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.4f}', ha='center', va='bottom')
        
        # 6. Acurácia de direção
        ax6 = plt.subplot(2, 3, 6)
        
        direction_acc = []
        for key, result in evaluation_results.items():
            direction_acc.append(result['metrics']['direction_accuracy'])
        
        bars = ax6.bar(range(len(model_names)), direction_acc, color=colors)
        ax6.set_xlabel('Modelos')
        ax6.set_ylabel('Acurácia de Direção')
        ax6.set_title('Acurácia na Previsão da Direção')
        ax6.set_xticks(range(len(model_names)))
        ax6.set_xticklabels(model_names, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Linha de referência (50%)
        ax6.axhline(y=0.5, color='k', linestyle='--', alpha=0.5)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, direction_acc):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{value:.1%}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Salvar figura se especificado
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Figuras salvas em {save_path}")
        
        plt.show()

# Instância global do avaliador
model_evaluator = ModelEvaluator()