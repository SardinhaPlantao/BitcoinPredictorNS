"""
MOTOR DE PREVISÃO
Sistema principal para fazer previsões com modelos treinados
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from sklearn.preprocessing import StandardScaler

class PredictionEngine:
    """
    Sistema completo de previsão do Bitcoin
    """
    
    def __init__(self):
        """Inicializa o motor de previsão"""
        self.trained_models = {}
        self.prediction_history = {}
        self.confidence_intervals = {}
        
    def predict_with_model(self,
                          model: Any,
                          X: pd.DataFrame,
                          model_type: str = 'sklearn',
                          return_confidence: bool = True,
                          confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Faz previsões com um modelo específico
        
        Args:
            model: Modelo treinado
            X: Features para previsão
            model_type: Tipo de modelo ('sklearn', 'keras', 'ensemble')
            return_confidence: Retornar intervalos de confiança
            confidence_level: Nível de confiança
        
        Returns:
            Previsões e metadados
        """
        print(f"\n🎯 Fazendo previsões (model_type={model_type})...")
        
        predictions = {}
        
        try:
            # Fazer previsões baseadas no tipo de modelo
            if model_type == 'sklearn':
                y_pred = model.predict(X)
            elif model_type == 'keras':
                y_pred = model.predict(X, verbose=0).flatten()
            elif model_type == 'ensemble':
                if hasattr(model, 'predict_with_uncertainty'):
                    result = model.predict_with_uncertainty(X)
                    y_pred = result['mean']
                    predictions['uncertainty'] = result
                else:
                    y_pred = model.predict(X)
            else:
                raise ValueError(f"Tipo de modelo não suportado: {model_type}")
            
            predictions['point_predictions'] = y_pred
            
            # Calcular intervalos de confiança se solicitado
            if return_confidence:
                ci = self._calculate_prediction_intervals(
                    model, X, y_pred, model_type, confidence_level
                )
                predictions['confidence_intervals'] = ci
            
            # Estatísticas das previsões
            predictions['statistics'] = {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'min': float(np.min(y_pred)),
                'max': float(np.max(y_pred)),
                'median': float(np.median(y_pred))
            }
            
            print(f"✅ {len(y_pred)} previsões realizadas")
            print(f"   Média: {predictions['statistics']['mean']:.4f}, "
                  f"Std: {predictions['statistics']['std']:.4f}")
            
        except Exception as e:
            print(f"❌ Erro ao fazer previsões: {str(e)}")
            predictions['error'] = str(e)
        
        return predictions
    
    def _calculate_prediction_intervals(self,
                                      model: Any,
                                      X: pd.DataFrame,
                                      y_pred: np.ndarray,
                                      model_type: str,
                                      confidence_level: float) -> Dict[str, Any]:
        """
        Calcula intervalos de predição
        """
        ci = {
            'confidence_level': confidence_level,
            'method': 'parametric'
        }
        
        try:
            if model_type == 'sklearn' and hasattr(model, 'predict'):
                # Tentar métodos específicos para diferentes modelos
                if hasattr(model, 'estimators_'):
                    # Ensemble model - usar desvio padrão das previsões individuais
                    individual_preds = []
                    for estimator in model.estimators_:
                        individual_preds.append(estimator.predict(X))
                    
                    std_pred = np.std(individual_preds, axis=0)
                    ci['method'] = 'ensemble_std'
                    
                elif hasattr(model, 'sigma_'):
                    # Gaussian Process
                    std_pred = model.sigma_
                    ci['method'] = 'gaussian_process'
                    
                else:
                    # Método paramétrico geral
                    # Estimativa conservadora baseada no erro médio
                    if hasattr(model, 'training_history_'):
                        # Usar RMSE do treinamento
                        train_rmse = np.sqrt(model.training_history_['final_loss'])
                        std_pred = np.full_like(y_pred, train_rmse)
                    else:
                        # Usar desvio padrão dos dados de treino
                        std_pred = np.std(y_pred) * np.ones_like(y_pred)
                    
                    ci['method'] = 'parametric_approximation'
                
                # Calcular intervalos
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
                ci['lower_bound'] = y_pred - z_score * std_pred
                ci['upper_bound'] = y_pred + z_score * std_pred
                ci['std'] = std_pred.tolist()
                
            elif model_type == 'ensemble' and 'uncertainty' in model.__dict__:
                # Usar incerteza do ensemble
                ci.update(model.uncertainty)
                ci['method'] = 'ensemble_uncertainty'
                
            else:
                # Método bootstrap simples
                ci = self._bootstrap_prediction_intervals(model, X, y_pred, confidence_level)
                ci['method'] = 'bootstrap'
                
        except Exception as e:
            print(f"⚠️ Erro ao calcular intervalos de confiança: {e}")
            # Intervalos padrão baseados no desvio das previsões
            std_pred = np.std(y_pred) * np.ones_like(y_pred)
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            
            ci['lower_bound'] = y_pred - z_score * std_pred
            ci['upper_bound'] = y_pred + z_score * std_pred
            ci['std'] = std_pred.tolist()
            ci['method'] = 'fallback_parametric'
        
        return ci
    
    def _bootstrap_prediction_intervals(self,
                                       model: Any,
                                       X: pd.DataFrame,
                                       y_pred: np.ndarray,
                                       confidence_level: float,
                                       n_bootstrap: int = 100) -> Dict[str, Any]:
        """Calcula intervalos de predição por bootstrap"""
        bootstrap_preds = []
        
        for i in range(n_bootstrap):
            try:
                # Bootstrap dos dados
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_bootstrap = X.iloc[indices] if hasattr(X, 'iloc') else X[indices]
                
                # Fazer previsão
                if hasattr(model, 'predict'):
                    pred = model.predict(X_bootstrap)
                    bootstrap_preds.append(pred)
            except:
                continue
        
        if not bootstrap_preds:
            return {}
        
        bootstrap_preds = np.array(bootstrap_preds)
        
        # Calcular percentis
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_preds, lower_percentile, axis=0)
        ci_upper = np.percentile(bootstrap_preds, upper_percentile, axis=0)
        
        return {
            'lower_bound': ci_lower.tolist(),
            'upper_bound': ci_upper.tolist(),
            'bootstrap_samples': n_bootstrap,
            'bootstrap_std': np.std(bootstrap_preds, axis=0).tolist()
        }
    
    def predict_future(self,
                      model: Any,
                      last_data: pd.DataFrame,
                      steps: int = 30,
                      feature_engineer: Any = None,
                      model_type: str = 'sklearn') -> Dict[str, Any]:
        """
        Previsão multi-step no futuro
        
        Args:
            model: Modelo treinado
            last_data: Últimos dados conhecidos
            steps: Número de passos à frente
            feature_engineer: Engine para criar features futuras
            model_type: Tipo de modelo
        
        Returns:
            Previsões futuras
        """
        print(f"\n🔮 Prevendo {steps} passos à frente...")
        
        future_predictions = {
            'dates': [],
            'predictions': [],
            'lower_bounds': [],
            'upper_bounds': []
        }
        
        current_data = last_data.copy()
        
        for step in range(steps):
            # Criar data futura
            future_date = pd.Timestamp.now() + timedelta(days=step)
            future_predictions['dates'].append(future_date)
            
            # Criar features para o passo atual
            if feature_engineer:
                # Se tivermos um feature engineer, usá-lo
                current_features = feature_engineer.create_features_for_date(
                    current_data, future_date
                )
            else:
                # Método simples: usar últimos dados conhecidos
                current_features = current_data.iloc[-1:].copy()
                
                # Adicionar features temporais
                current_features['day_of_week'] = future_date.dayofweek
                current_features['month'] = future_date.month
                current_features['day_of_year'] = future_date.dayofyear
            
            # Fazer previsão
            try:
                if model_type == 'sklearn':
                    pred = model.predict(current_features)[0]
                elif model_type == 'keras':
                    pred = model.predict(current_features, verbose=0)[0, 0]
                else:
                    pred = model.predict(current_features)[0]
                
                future_predictions['predictions'].append(float(pred))
                
                # Atualizar dados atuais com a previsão
                # (para modelos auto-regressivos)
                if step < steps - 1:
                    new_row = current_data.iloc[-1:].copy()
                    # Aqui precisaríamos atualizar as features apropriadas
                    # Por simplicidade, apenas adicionamos a previsão
                    current_data = pd.concat([current_data, new_row])
                
            except Exception as e:
                print(f"⚠️ Erro no passo {step}: {e}")
                future_predictions['predictions'].append(np.nan)
        
        # Calcular estatísticas
        valid_preds = [p for p in future_predictions['predictions'] if not np.isnan(p)]
        
        if valid_preds:
            future_predictions['statistics'] = {
                'mean': float(np.mean(valid_preds)),
                'std': float(np.std(valid_preds)),
                'trend': self._calculate_trend(valid_preds),
                'percent_change': ((valid_preds[-1] / valid_preds[0] - 1) * 100) if len(valid_preds) > 1 else 0
            }
        
        print(f"✅ {len(valid_preds)}/{steps} previsões futuras realizadas")
        
        return future_predictions
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula tendência da série"""
        if len(values) < 2:
            return "stable"
        
        # Regressão linear simples
        x = np.arange(len(values))
        slope, _, r_value, _, _ = stats.linregress(x, values)
        
        if slope > 0.01 and r_value > 0.5:
            return "upward"
        elif slope < -0.01 and r_value > 0.5:
            return "downward"
        else:
            return "stable"
    
    def ensemble_predict(self,
                        models: Dict[str, Any],
                        X: pd.DataFrame,
                        method: str = 'weighted_average',
                        weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Previsão com ensemble de múltiplos modelos
        
        Args:
            models: Dicionário de modelos treinados
            X: Features para previsão
            method: Método de combinação
            weights: Pesos para cada modelo
        
        Returns:
            Previsões do ensemble
        """
        print(f"\n🎭 Fazendo previsões com ensemble ({method})...")
        
        if not models:
            raise ValueError("Nenhum modelo fornecido para ensemble")
        
        # Coletar previsões individuais
        individual_predictions = {}
        
        for name, model_info in models.items():
            model = model_info.get('model', model_info)  # Suporta dict ou modelo direto
            model_type = model_info.get('type', 'sklearn') if isinstance(model_info, dict) else 'sklearn'
            
            print(f"  Modelo {name}...")
            
            try:
                pred_result = self.predict_with_model(
                    model, X, model_type, return_confidence=False
                )
                
                individual_predictions[name] = {
                    'predictions': pred_result['point_predictions'],
                    'statistics': pred_result['statistics']
                }
                
            except Exception as e:
                print(f"    ❌ Erro no modelo {name}: {e}")
                continue
        
        if not individual_predictions:
            raise ValueError("Nenhum modelo produziu previsões válidas")
        
        # Combinar previsões
        if method == 'weighted_average':
            ensemble_pred = self._weighted_average_ensemble(individual_predictions, weights)
        elif method == 'median':
            ensemble_pred = self._median_ensemble(individual_predictions)
        elif method == 'voting':
            ensemble_pred = self._voting_ensemble(individual_predictions)
        else:
            raise ValueError(f"Método {method} não suportado")
        
        # Calcular incerteza do ensemble
        ensemble_uncertainty = self._calculate_ensemble_uncertainty(individual_predictions)
        
        results = {
            'ensemble_predictions': ensemble_pred['predictions'],
            'individual_predictions': individual_predictions,
            'ensemble_method': method,
            'ensemble_weights': ensemble_pred.get('weights', {}),
            'ensemble_statistics': ensemble_pred.get('statistics', {}),
            'uncertainty': ensemble_uncertainty,
            'model_agreement': self._calculate_model_agreement(individual_predictions)
        }
        
        print(f"✅ Ensemble criado com {len(individual_predictions)} modelos")
        print(f"   RMSE estimado: {ensemble_uncertainty.get('rmse_estimate', 0):.4f}")
        
        return results
    
    def _weighted_average_ensemble(self,
                                  individual_predictions: Dict[str, Any],
                                  weights: Dict[str, float] = None) -> Dict[str, Any]:
        """Combina previsões por média ponderada"""
        if weights is None:
            # Pesos iguais
            weights = {name: 1.0 for name in individual_predictions.keys()}
        
        # Normalizar pesos
        total_weight = sum(weights.values())
        normalized_weights = {name: w/total_weight for name, w in weights.items()}
        
        # Calcular média ponderada
        weighted_sum = None
        for name, pred_info in individual_predictions.items():
            weight = normalized_weights[name]
            predictions = pred_info['predictions']
            
            if weighted_sum is None:
                weighted_sum = predictions * weight
            else:
                weighted_sum += predictions * weight
        
        ensemble_pred = weighted_sum
        
        # Estatísticas
        stats = {
            'mean': float(np.mean(ensemble_pred)),
            'std': float(np.std(ensemble_pred)),
            'min': float(np.min(ensemble_pred)),
            'max': float(np.max(ensemble_pred))
        }
        
        return {
            'predictions': ensemble_pred,
            'weights': normalized_weights,
            'statistics': stats
        }
    
    def _median_ensemble(self, individual_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combina previsões por mediana"""
        all_predictions = []
        for pred_info in individual_predictions.values():
            all_predictions.append(pred_info['predictions'])
        
        all_predictions = np.column_stack(all_predictions)
        ensemble_pred = np.median(all_predictions, axis=1)
        
        stats = {
            'mean': float(np.mean(ensemble_pred)),
            'std': float(np.std(ensemble_pred))
        }
        
        return {
            'predictions': ensemble_pred,
            'statistics': stats
        }
    
    def _voting_ensemble(self, individual_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combina previsões por votação (para classificação)"""
        # Para regressão, usar média ponderada como fallback
        return self._weighted_average_ensemble(individual_predictions)
    
    def _calculate_ensemble_uncertainty(self, individual_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula incerteza do ensemble"""
        all_predictions = []
        for pred_info in individual_predictions.values():
            all_predictions.append(pred_info['predictions'])
        
        all_predictions = np.array(all_predictions)
        
        # Variância entre modelos
        between_model_var = np.var(all_predictions, axis=0)
        
        # Estimativa de erro (simplificada)
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0)
        
        return {
            'between_model_variance': between_model_var.tolist(),
            'mean_std': std_pred.tolist(),
            'coefficient_of_variation': (std_pred / np.abs(mean_pred)).tolist(),
            'rmse_estimate': np.sqrt(np.mean(between_model_var))
        }
    
    def _calculate_model_agreement(self, individual_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula concordância entre modelos"""
        all_predictions = []
        model_names = []
        
        for name, pred_info in individual_predictions.items():
            all_predictions.append(pred_info['predictions'])
            model_names.append(name)
        
        all_predictions = np.array(all_predictions)
        
        # Correlação entre previsões dos modelos
        n_models = len(model_names)
        correlation_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                corr = np.corrcoef(all_predictions[i], all_predictions[j])[0, 1]
                correlation_matrix[i, j] = corr if not np.isnan(corr) else 0
        
        # Concordância média
        mean_correlation = np.mean(correlation_matrix[np.triu_indices(n_models, k=1)])
        
        return {
            'correlation_matrix': correlation_matrix.tolist(),
            'model_names': model_names,
            'mean_correlation': float(mean_correlation),
            'agreement_level': 'high' if mean_correlation > 0.7 else 
                              'medium' if mean_correlation > 0.4 else 'low'
        }
    
    def create_prediction_report(self,
                                predictions: Dict[str, Any],
                                actual_values: np.ndarray = None,
                                include_evaluation: bool = True) -> Dict[str, Any]:
        """
        Cria relatório completo de previsões
        
        Args:
            predictions: Resultados das previsões
            actual_values: Valores reais (para avaliação)
            include_evaluation: Incluir avaliação se valores reais disponíveis
        
        Returns:
            Relatório de previsões
        """
        report = {
            'summary': {},
            'predictions': {},
            'uncertainty': {},
            'evaluation': {},
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Sumário
        if 'statistics' in predictions:
            report['summary'] = predictions['statistics']
        
        # Previsões
        if 'point_predictions' in predictions:
            report['predictions']['point'] = predictions['point_predictions'].tolist()
        
        if 'confidence_intervals' in predictions:
            report['predictions']['intervals'] = predictions['confidence_intervals']
        
        # Incerteza
        if 'uncertainty' in predictions:
            report['uncertainty'] = predictions['uncertainty']
        
        # Avaliação se valores reais disponíveis
        if include_evaluation and actual_values is not None and 'point_predictions' in predictions:
            from ml_models.model_evaluator import model_evaluator
            
            eval_result = model_evaluator.evaluate_model(
                actual_values,
                predictions['point_predictions'],
                model_name='prediction_engine',
                dataset_type='prediction'
            )
            
            report['evaluation'] = eval_result
        
        # Recomendações
        report['recommendations'] = self._generate_prediction_recommendations(predictions)
        
        return report
    
    def _generate_prediction_recommendations(self, predictions: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas nas previsões"""
        recommendations = []
        
        if 'statistics' in predictions:
            stats = predictions['statistics']
            
            # Recomendação baseada na tendência
            if 'trend' in stats:
                if stats['trend'] == 'upward':
                    recommendations.append("Tendência de alta prevista - considere posições longas")
                elif stats['trend'] == 'downward':
                    recommendations.append("Tendência de baixa prevista - considere posições curtas ou hedge")
            
            # Recomendação baseada na volatilidade
            if 'std' in stats and stats['std'] > stats.get('mean', 1) * 0.1:
                recommendations.append("Alta volatilidade prevista - ajuste tamanho da posição")
        
        # Recomendação baseada na incerteza
        if 'confidence_intervals' in predictions:
            ci = predictions['confidence_intervals']
            if 'std' in ci:
                mean_std = np.mean(ci['std'])
                if mean_std > 0.1:
                    recommendations.append("Alta incerteza nas previsões - espere confirmação do mercado")
        
        if not recommendations:
            recommendations.append("Previsões dentro de parâmetros normais - mantenha estratégia atual")
        
        return recommendations
    
    def save_predictions(self, predictions: Dict[str, Any], path: str = "predictions"):
        """Salva previsões para análise futura"""
        import os
        import json
        import pickle
        
        os.makedirs(path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(path, f"predictions_{timestamp}.pkl")
        
        # Adicionar metadados
        predictions['metadata'] = {
            'saved_at': timestamp,
            'engine_version': '1.0',
            'model_count': len(self.trained_models)
        }
        
        # Salvar
        with open(filename, 'wb') as f:
            pickle.dump(predictions, f)
        
        # Salvar também como JSON (para partes serializáveis)
        json_filename = os.path.join(path, f"predictions_{timestamp}.json")
        json_data = {
            'statistics': predictions.get('statistics', {}),
            'metadata': predictions.get('metadata', {}),
            'recommendations': predictions.get('recommendations', [])
        }
        
        with open(json_filename, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print(f"💾 Previsões salvas em {filename}")
    
    def load_predictions(self, path: str = "predictions") -> Dict[str, Any]:
        """Carrega previsões salvas"""
        import os
        import pickle
        import glob
        
        if not os.path.exists(path):
            print(f"⚠️ Diretório {path} não encontrado")
            return {}
        
        # Carregar previsão mais recente
        prediction_files = glob.glob(os.path.join(path, "predictions_*.pkl"))
        
        if not prediction_files:
            print(f"⚠️ Nenhuma previsão encontrada em {path}")
            return {}
        
        # Ordenar por data
        prediction_files.sort(reverse=True)
        latest_file = prediction_files[0]
        
        with open(latest_file, 'rb') as f:
            predictions = pickle.load(f)
        
        print(f"📂 Previsões carregadas de {latest_file}")
        print(f"   Data: {predictions.get('metadata', {}).get('saved_at', 'desconhecida')}")
        
        return predictions

# Instância global do motor de previsão
prediction_engine = PredictionEngine()