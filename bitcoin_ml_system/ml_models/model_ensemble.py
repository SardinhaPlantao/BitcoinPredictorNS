"""
SISTEMA DE ENSEMBLE AVANÇADO
Combina múltiplos modelos para previsões robustas
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy import stats

class AdvancedEnsemble(BaseEstimator, RegressorMixin):
    """
    Ensemble avançado com múltiplas estratégias de combinação
    """
    
    def __init__(self, 
                 models: List[Any] = None,
                 weights: List[float] = None,
                 combination_method: str = 'weighted_average',
                 use_dynamic_weights: bool = True):
        """
        Inicializa o ensemble
        
        Args:
            models: Lista de modelos
            weights: Pesos para cada modelo
            combination_method: Método de combinação
            use_dynamic_weights: Usar pesos dinâmicos baseados em performance
        """
        self.models = models if models else []
        self.weights = weights if weights else []
        self.combination_method = combination_method
        self.use_dynamic_weights = use_dynamic_weights
        self.model_performance = {}
        self.is_fitted_ = False
        
    def fit(self, X, y, sample_weight=None):
        """
        Treina todos os modelos do ensemble
        """
        X, y = check_X_y(X, y)
        
        if not self.models:
            self._initialize_default_models()
        
        # Treinar cada modelo
        for i, model in enumerate(self.models):
            print(f"  Ensemble: Treinando modelo {i+1}/{len(self.models)}")
            model.fit(X, y)
            
            # Avaliar performance no treino
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            self.model_performance[i] = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'model': model.__class__.__name__
            }
        
        # Calcular pesos iniciais
        if self.use_dynamic_weights:
            self._calculate_dynamic_weights()
        elif not self.weights:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        self.is_fitted_ = True
        return self
    
    def _initialize_default_models(self):
        """Inicializa modelos padrão se nenhum for fornecido"""
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from sklearn.linear_model import Ridge
        
        self.models = [
            RandomForestRegressor(n_estimators=100, random_state=42),
            XGBRegressor(n_estimators=100, random_state=42),
            LGBMRegressor(n_estimators=100, random_state=42),
            Ridge(alpha=1.0)
        ]
    
    def _calculate_dynamic_weights(self):
        """Calcula pesos dinâmicos baseados na performance"""
        if not self.model_performance:
            self.weights = [1.0 / len(self.models)] * len(self.models)
            return
        
        # Usar inverso do MSE como peso
        mses = [self.model_performance[i]['mse'] for i in range(len(self.models))]
        
        # Evitar divisão por zero
        mses = [max(mse, 1e-10) for mse in mses]
        
        # Pesos inversamente proporcionais ao MSE
        weights = [1.0 / mse for mse in mses]
        total = sum(weights)
        
        # Normalizar
        self.weights = [w / total for w in weights]
        
        print("Pesos dinâmicos calculados:")
        for i, weight in enumerate(self.weights):
            print(f"  Modelo {i+1}: {weight:.3f} (MSE={self.model_performance[i]['mse']:.6f})")
    
    def predict(self, X):
        """
        Faz previsões combinando todos os modelos
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        # Previsões individuais
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Combinar previsões
        if self.combination_method == 'weighted_average':
            return self._weighted_average(predictions)
        elif self.combination_method == 'median':
            return self._median_combination(predictions)
        elif self.combination_method == 'trimmed_mean':
            return self._trimmed_mean(predictions)
        elif self.combination_method == 'stacking':
            return self._stacking_combination(predictions, X)
        else:
            return self._weighted_average(predictions)
    
    def _weighted_average(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Média ponderada das previsões"""
        weighted_sum = np.zeros_like(predictions[0])
        
        for pred, weight in zip(predictions, self.weights):
            weighted_sum += pred * weight
        
        return weighted_sum
    
    def _median_combination(self, predictions: List[np.ndarray]) -> np.ndarray:
        """Mediana das previsões"""
        predictions_array = np.column_stack(predictions)
        return np.median(predictions_array, axis=1)
    
    def _trimmed_mean(self, predictions: List[np.ndarray], trim_ratio: float = 0.1) -> np.ndarray:
        """Média aparada (remove outliers)"""
        predictions_array = np.column_stack(predictions)
        n_trim = int(len(predictions) * trim_ratio)
        
        result = []
        for i in range(predictions_array.shape[0]):
            row = predictions_array[i]
            row_sorted = np.sort(row)
            trimmed = row_sorted[n_trim:-n_trim] if n_trim > 0 else row_sorted
            result.append(np.mean(trimmed))
        
        return np.array(result)
    
    def _stacking_combination(self, predictions: List[np.ndarray], X: np.ndarray) -> np.ndarray:
        """Stacking simplificado"""
        from sklearn.linear_model import LinearRegression
        
        # Meta-features
        meta_X = np.column_stack(predictions)
        
        # Para simplificar, usar média ponderada como fallback
        # Em uma implementação completa, treinaríamos um meta-modelo
        return self._weighted_average(predictions)
    
    def predict_with_uncertainty(self, X, confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Previsão com intervalo de confiança
        """
        check_is_fitted(self, 'is_fitted_')
        X = check_array(X)
        
        # Previsões individuais
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions_array = np.column_stack(predictions)
        
        # Calcular estatísticas
        mean_pred = np.mean(predictions_array, axis=1)
        std_pred = np.std(predictions_array, axis=1)
        
        # Intervalo de confiança
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        lower_bound = mean_pred - z_score * std_pred
        upper_bound = mean_pred + z_score * std_pred
        
        # Calcular concordância entre modelos
        agreement = self._calculate_agreement(predictions_array)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level,
            'agreement': agreement,
            'individual_predictions': predictions_array,
            'model_contributions': self._get_model_contributions(predictions_array)
        }
    
    def _calculate_agreement(self, predictions: np.ndarray) -> np.ndarray:
        """Calcula concordância entre modelos"""
        # Coeficiente de variação (menor = mais concordância)
        mean = np.mean(predictions, axis=1)
        std = np.std(predictions, axis=1)
        
        # Evitar divisão por zero
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = np.where(mean != 0, std / np.abs(mean), 0)
        
        # Concordância: 1 - CV normalizado
        agreement = 1 - np.clip(cv, 0, 1)
        return agreement
    
    def _get_model_contributions(self, predictions: np.ndarray) -> np.ndarray:
        """Calcula contribuição de cada modelo para a previsão final"""
        if self.combination_method == 'weighted_average':
            # Contribuição proporcional ao peso
            contributions = np.array(self.weights) * 100  # Em percentual
        else:
            # Para outros métodos, usar variância explicada
            mean_pred = np.mean(predictions, axis=1, keepdims=True)
            variances = np.var(predictions - mean_pred, axis=0)
            total_variance = np.sum(variances)
            
            if total_variance > 0:
                contributions = (variances / total_variance) * 100
            else:
                contributions = np.array([100 / len(self.models)] * len(self.models))
        
        return contributions
    
    def get_model_performance(self) -> pd.DataFrame:
        """Retorna performance de cada modelo"""
        if not self.model_performance:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(self.model_performance, orient='index')
        df['weight'] = self.weights
        df['contribution'] = self._get_model_contributions(
            np.zeros((1, len(self.models)))  # Dummy array para cálculo
        )
        
        return df
    
    def adapt_weights(self, X_val, y_val):
        """
        Adapta pesos baseados em performance na validação
        """
        if not self.is_fitted_:
            raise ValueError("Ensemble não treinado")
        
        # Avaliar cada modelo na validação
        val_performance = {}
        for i, model in enumerate(self.models):
            y_pred = model.predict(X_val)
            mse = np.mean((y_val - y_pred) ** 2)
            val_performance[i] = mse
        
        # Atualizar pesos baseados na nova performance
        mses = [val_performance[i] for i in range(len(self.models))]
        mses = [max(mse, 1e-10) for mse in mses]
        
        weights = [1.0 / mse for mse in mses]
        total = sum(weights)
        
        self.weights = [w / total for w in weights]
        
        print("Pesos adaptados com validação:")
        for i, weight in enumerate(self.weights):
            print(f"  Modelo {i+1}: {weight:.3f} (MSE={val_performance[i]:.6f})")

class BayesianEnsemble:
    """
    Ensemble bayesiano para previsão com incerteza
    """
    
    def __init__(self, n_models: int = 10):
        self.n_models = n_models
        self.models = []
        self.performance_history = []
        
    def fit(self, X, y):
        """Treina múltiplos modelos com bootstrap"""
        from sklearn.utils import resample
        
        for i in range(self.n_models):
            # Bootstrap sample
            X_sample, y_sample = resample(X, y, random_state=42 + i)
            
            # Criar e treinar modelo
            model = self._create_model(i)
            model.fit(X_sample, y_sample)
            
            # Avaliar no conjunto completo
            y_pred = model.predict(X)
            mse = np.mean((y - y_pred) ** 2)
            
            self.models.append(model)
            self.performance_history.append({
                'model_idx': i,
                'mse': mse,
                'model_type': model.__class__.__name__
            })
        
        print(f"✅ Ensemble bayesiano treinado com {len(self.models)} modelos")
    
    def _create_model(self, seed: int):
        """Cria modelo com random state único"""
        from sklearn.ensemble import RandomForestRegressor
        
        return RandomForestRegressor(
            n_estimators=50,
            max_depth=5,
            random_state=seed
        )
    
    def predict_distribution(self, X):
        """
        Retorna distribuição de previsões
        """
        predictions = []
        
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions_array = np.array(predictions)
        
        # Calcular estatísticas da distribuição
        mean = np.mean(predictions_array, axis=0)
        std = np.std(predictions_array, axis=0)
        
        # Ajustar distribuição (normal ou t-student)
        # Verificar normalidade
        from scipy import stats
        
        normality_test = []
        for i in range(predictions_array.shape[1]):
            # Teste de Shapiro-Wilk para normalidade
            if len(predictions_array[:, i]) >= 3:
                _, p_value = stats.shapiro(predictions_array[:, i])
                normality_test.append(p_value > 0.05)
        
        is_normal = np.mean(normality_test) > 0.5 if normality_test else True
        
        # Calcular intervalos de confiança
        if is_normal:
            # Intervalo normal
            z_score = stats.norm.ppf(0.975)  # 95% CI
            lower = mean - z_score * std
            upper = mean + z_score * std
            distribution = 'normal'
        else:
            # Intervalo t-student
            df = len(self.models) - 1
            t_score = stats.t.ppf(0.975, df)
            lower = mean - t_score * std
            upper = mean + t_score * std
            distribution = 't-student'
        
        return {
            'mean': mean,
            'std': std,
            'lower_95': lower,
            'upper_95': upper,
            'distribution': distribution,
            'all_predictions': predictions_array,
            'model_variance': np.var(predictions_array, axis=0),
            'prediction_entropy': self._calculate_entropy(predictions_array)
        }
    
    def _calculate_entropy(self, predictions: np.ndarray) -> np.ndarray:
        """Calcula entropia das previsões (incerteza)"""
        # Discretizar previsões para cálculo de entropia
        n_bins = 10
        entropy = []
        
        for i in range(predictions.shape[1]):
            hist, _ = np.histogram(predictions[:, i], bins=n_bins)
            prob = hist / hist.sum()
            prob = prob[prob > 0]  # Remover zeros
            
            # Entropia de Shannon
            e = -np.sum(prob * np.log2(prob))
            entropy.append(e)
        
        return np.array(entropy)
    
    def get_uncertainty_decomposition(self, X):
        """
        Decompõe incerteza em componentes
        """
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions_array = np.array(predictions)
        
        # Variância total
        total_variance = np.var(predictions_array, axis=0)
        
        # Variância entre modelos
        between_model_var = np.var(np.mean(predictions_array, axis=1))
        
        # Variância dentro do modelo (aproximada)
        within_model_var = total_variance - between_model_var
        
        return {
            'total_variance': total_variance,
            'between_model_variance': between_model_var,
            'within_model_variance': within_model_var,
            'variance_ratio': between_model_var / total_variance if total_variance > 0 else 0
        }

# Instância global do ensemble
model_ensemble = AdvancedEnsemble()