"""
TREINADOR DE MODELOS ML
Treina múltiplos modelos para previsão do Bitcoin
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
import pickle
import joblib
warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, 
    AdaBoostRegressor, ExtraTreesRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, 
    r2_score, explained_variance_score
)

from core.config import config
from utils.helpers import split_time_series

class ModelTrainer:
    """
    Sistema completo de treinamento de modelos ML
    """
    
    def __init__(self):
        """Inicializa o treinador de modelos"""
        self.models = {}
        self.model_metadata = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def train_all_models(self,
                        X_train: pd.DataFrame,
                        y_train: pd.Series,
                        X_test: pd.DataFrame,
                        y_test: pd.Series,
                        model_types: List[str] = None) -> Dict[str, Any]:
        """
        Treina múltiplos modelos simultaneamente
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_test: Features de teste
            y_test: Target de teste
            model_types: Tipos de modelos a treinar
        
        Returns:
            Dicionário com todos os modelos treinados
        """
        print("\n🤖 Treinando múltiplos modelos ML...")
        
        if model_types is None:
            model_types = [
                'linear', 'ridge', 'lasso', 'elasticnet',
                'random_forest', 'xgboost', 'lightgbm', 'catboost',
                'gradient_boosting', 'svr', 'knn', 'mlp'
            ]
        
        results = {}
        
        for model_type in model_types:
            try:
                print(f"  🔧 Treinando {model_type}...")
                
                model = self._create_model(model_type)
                
                # Treinar modelo
                model.fit(X_train, y_train)
                
                # Fazer previsões
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Avaliar
                train_metrics = self._calculate_metrics(y_train, y_pred_train)
                test_metrics = self._calculate_metrics(y_test, y_pred_test)
                
                # Calcular feature importance se disponível
                importance = self._get_feature_importance(model, X_train.columns)
                
                # Salvar resultados
                results[model_type] = {
                    'model': model,
                    'train_predictions': y_pred_train,
                    'test_predictions': y_pred_test,
                    'train_metrics': train_metrics,
                    'test_metrics': test_metrics,
                    'feature_importance': importance,
                    'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Salvar no cache
                self.models[model_type] = model
                self.feature_importance[model_type] = importance
                
                print(f"    ✅ {model_type}: RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}")
                
            except Exception as e:
                print(f"    ❌ Erro ao treinar {model_type}: {str(e)}")
                continue
        
        print(f"\n✅ {len(results)} modelos treinados com sucesso")
        return results
    
    def _create_model(self, model_type: str) -> Any:
        """Cria modelo baseado no tipo"""
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=config.RANDOM_STATE),
            'lasso': Lasso(alpha=0.1, random_state=config.RANDOM_STATE),
            'elasticnet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=config.RANDOM_STATE),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'xgboost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'lightgbm': LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'catboost': CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=config.RANDOM_STATE,
                verbose=False
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                random_state=config.RANDOM_STATE
            ),
            'svr': SVR(kernel='rbf', C=1.0, epsilon=0.1),
            'knn': KNeighborsRegressor(n_neighbors=5, weights='distance'),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=config.RANDOM_STATE
            ),
            'adaboost': AdaBoostRegressor(
                n_estimators=50,
                learning_rate=1.0,
                random_state=config.RANDOM_STATE
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            )
        }
        
        if model_type not in models:
            raise ValueError(f"Modelo {model_type} não suportado")
        
        return models[model_type]
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas de avaliação"""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred),
            'mape': self._calculate_mape(y_true, y_pred)
        }
        
        # Calcular direção correta (para previsão de preços)
        if len(y_true) > 1 and len(y_pred) > 1:
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(y_pred) > 0
            metrics['direction_accuracy'] = np.mean(direction_true == direction_pred)
        
        return metrics
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calcula Mean Absolute Percentage Error"""
        # Evitar divisão por zero
        mask = y_true != 0
        if np.sum(mask) > 0:
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return np.nan
    
    def _get_feature_importance(self, model, feature_names) -> Optional[pd.Series]:
        """Extrai importância das features do modelo"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_)
            elif hasattr(model, 'get_booster'):
                # XGBoost
                importance = model.get_booster().get_score(importance_type='weight')
                # Converter para array
                importance = np.array([importance.get(f, 0) for f in feature_names])
            else:
                return None
            
            # Normalizar
            if importance.sum() > 0:
                importance = importance / importance.sum()
            
            return pd.Series(importance, index=feature_names, name='importance')
            
        except Exception:
            return None
    
    def train_with_cross_validation(self,
                                   X: pd.DataFrame,
                                   y: pd.Series,
                                   model_type: str = 'random_forest',
                                   n_splits: int = 5) -> Dict[str, Any]:
        """
        Treina modelo com validação cruzada temporal
        
        Args:
            X: Features
            y: Target
            model_type: Tipo de modelo
            n_splits: Número de splits para validação cruzada
        
        Returns:
            Resultados do treinamento com CV
        """
        print(f"\n🔍 Treinando {model_type} com validação cruzada temporal...")
        
        model = self._create_model(model_type)
        
        # Validação cruzada temporal
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'rmse': [],
            'mae': [],
            'r2': [],
            'predictions': [],
            'train_sizes': []
        }
        
        fold = 1
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            print(f"  Fold {fold}/{n_splits}: Train={len(X_train)}, Test={len(X_test)}")
            
            # Treinar no fold
            model.fit(X_train, y_train)
            
            # Prever
            y_pred = model.predict(X_test)
            
            # Calcular métricas
            cv_scores['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
            cv_scores['mae'].append(mean_absolute_error(y_test, y_pred))
            cv_scores['r2'].append(r2_score(y_test, y_pred))
            cv_scores['train_sizes'].append(len(X_train))
            
            # Salvar previsões
            cv_scores['predictions'].append({
                'y_true': y_test.values,
                'y_pred': y_pred,
                'indices': test_idx
            })
            
            fold += 1
        
        # Estatísticas dos folds
        cv_results = {
            'model': model,
            'cv_scores': cv_scores,
            'mean_rmse': np.mean(cv_scores['rmse']),
            'std_rmse': np.std(cv_scores['rmse']),
            'mean_mae': np.mean(cv_scores['mae']),
            'std_mae': np.std(cv_scores['mae']),
            'mean_r2': np.mean(cv_scores['r2']),
            'std_r2': np.std(cv_scores['r2']),
            'n_splits': n_splits,
            'feature_importance': self._get_feature_importance(model, X.columns)
        }
        
        print(f"✅ CV concluído: RMSE={cv_results['mean_rmse']:.4f} ± {cv_results['std_rmse']:.4f}")
        
        return cv_results
    
    def optimize_hyperparameters(self,
                                X_train: pd.DataFrame,
                                y_train: pd.Series,
                                model_type: str = 'random_forest',
                                param_grid: Dict = None,
                                cv_splits: int = 3) -> Dict[str, Any]:
        """
        Otimiza hiperparâmetros usando validação cruzada
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            model_type: Tipo de modelo
            param_grid: Grid de parâmetros
            cv_splits: Número de splits para CV
        
        Returns:
            Melhor modelo e parâmetros
        """
        print(f"\n🎯 Otimizando hiperparâmetros para {model_type}...")
        
        from sklearn.model_selection import GridSearchCV
        
        # Grid de parâmetros padrão
        if param_grid is None:
            param_grid = self._get_default_param_grid(model_type)
        
        # Criar modelo base
        base_model = self._create_model(model_type)
        
        # Busca em grid com validação cruzada temporal
        tscv = TimeSeriesSplit(n_splits=cv_splits)
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Executar busca
        grid_search.fit(X_train, y_train)
        
        # Resultados
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = -grid_search.best_score_
        
        results = {
            'best_model': best_model,
            'best_params': best_params,
            'best_score': best_score,
            'cv_results': grid_search.cv_results_,
            'param_grid': param_grid
        }
        
        print(f"✅ Melhores parâmetros: {best_params}")
        print(f"   Melhor RMSE: {np.sqrt(best_score):.4f}")
        
        return results
    
    def _get_default_param_grid(self, model_type: str) -> Dict:
        """Retorna grid de parâmetros padrão para cada modelo"""
        grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1.0]
            }
        }
        
        return grids.get(model_type, {})
    
    def create_ensemble(self,
                       X_train: pd.DataFrame,
                       y_train: pd.Series,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       ensemble_method: str = 'voting') -> Dict[str, Any]:
        """
        Cria ensemble de múltiplos modelos
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_test: Features de teste
            y_test: Target de teste
            ensemble_method: Método de ensemble ('voting', 'stacking', 'averaging')
        
        Returns:
            Ensemble treinado e resultados
        """
        print(f"\n🎭 Criando ensemble ({ensemble_method})...")
        
        # Treinar modelos individuais
        base_models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=config.RANDOM_STATE),
            'xgb': XGBRegressor(n_estimators=100, random_state=config.RANDOM_STATE),
            'lgb': LGBMRegressor(n_estimators=100, random_state=config.RANDOM_STATE),
            'gb': GradientBoostingRegressor(n_estimators=100, random_state=config.RANDOM_STATE)
        }
        
        # Treinar cada modelo
        trained_models = {}
        for name, model in base_models.items():
            print(f"  Treinando {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        # Criar ensemble
        if ensemble_method == 'averaging':
            ensemble_pred = self._average_ensemble(trained_models, X_test)
        elif ensemble_method == 'voting':
            ensemble_pred = self._voting_ensemble(trained_models, X_test)
        elif ensemble_method == 'stacking':
            ensemble_pred = self._stacking_ensemble(trained_models, X_train, y_train, X_test)
        else:
            raise ValueError(f"Método {ensemble_method} não suportado")
        
        # Avaliar ensemble
        ensemble_metrics = self._calculate_metrics(y_test, ensemble_pred)
        
        # Avaliar modelos individuais
        individual_results = {}
        for name, model in trained_models.items():
            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            individual_results[name] = metrics
        
        results = {
            'ensemble_method': ensemble_method,
            'ensemble_predictions': ensemble_pred,
            'ensemble_metrics': ensemble_metrics,
            'individual_models': trained_models,
            'individual_results': individual_results,
            'feature_importance': self._get_ensemble_importance(trained_models, X_train.columns)
        }
        
        print(f"✅ Ensemble criado: RMSE={ensemble_metrics['rmse']:.4f}, R²={ensemble_metrics['r2']:.4f}")
        
        return results
    
    def _average_ensemble(self, models: Dict, X: pd.DataFrame) -> np.ndarray:
        """Ensemble por média simples"""
        predictions = []
        for model in models.values():
            predictions.append(model.predict(X))
        
        return np.mean(predictions, axis=0)
    
    def _voting_ensemble(self, models: Dict, X: pd.DataFrame) -> np.ndarray:
        """Ensemble por média ponderada (baseada em performance)"""
        # Usar pesos iguais por enquanto
        return self._average_ensemble(models, X)
    
    def _stacking_ensemble(self, models: Dict, 
                          X_train: pd.DataFrame, 
                          y_train: pd.Series,
                          X_test: pd.DataFrame) -> np.ndarray:
        """Ensemble por stacking com meta-modelo"""
        from sklearn.linear_model import LinearRegression
        
        # Gerar previsões de base
        base_predictions_train = []
        for name, model in models.items():
            pred = model.predict(X_train)
            base_predictions_train.append(pred)
        
        # Treinar meta-modelo
        X_meta_train = np.column_stack(base_predictions_train)
        meta_model = LinearRegression()
        meta_model.fit(X_meta_train, y_train)
        
        # Prever com meta-modelo
        base_predictions_test = []
        for name, model in models.items():
            pred = model.predict(X_test)
            base_predictions_test.append(pred)
        
        X_meta_test = np.column_stack(base_predictions_test)
        return meta_model.predict(X_meta_test)
    
    def _get_ensemble_importance(self, models: Dict, feature_names) -> pd.DataFrame:
        """Calcula importância das features para o ensemble"""
        importance_df = pd.DataFrame(index=feature_names)
        
        for name, model in models.items():
            importance = self._get_feature_importance(model, feature_names)
            if importance is not None:
                importance_df[name] = importance
        
        if not importance_df.empty:
            importance_df['ensemble_mean'] = importance_df.mean(axis=1)
            importance_df['ensemble_std'] = importance_df.std(axis=1)
        
        return importance_df
    
    def save_models(self, path: str = "models"):
        """Salva todos os modelos treinados"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            filename = os.path.join(path, f"{name}_model.pkl")
            with open(filename, 'wb') as f:
                pickle.dump(model, f)
            
            # Salvar metadados
            if name in self.feature_importance:
                importance_file = os.path.join(path, f"{name}_importance.pkl")
                with open(importance_file, 'wb') as f:
                    pickle.dump(self.feature_importance[name], f)
        
        # Salvar metadados gerais
        metadata_file = os.path.join(path, "training_metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(self.model_metadata, f)
        
        print(f"💾 Modelos salvos em {path}")
    
    def load_models(self, path: str = "models"):
        """Carrega modelos salvos"""
        import os
        import glob
        
        if not os.path.exists(path):
            print(f"⚠️ Diretório {path} não encontrado")
            return
        
        # Carregar modelos
        model_files = glob.glob(os.path.join(path, "*_model.pkl"))
        
        for file in model_files:
            with open(file, 'rb') as f:
                model = pickle.load(f)
                name = os.path.basename(file).replace("_model.pkl", "")
                self.models[name] = model
        
        # Carregar importância das features
        importance_files = glob.glob(os.path.join(path, "*_importance.pkl"))
        
        for file in importance_files:
            with open(file, 'rb') as f:
                importance = pickle.load(f)
                name = os.path.basename(file).replace("_importance.pkl", "")
                self.feature_importance[name] = importance
        
        # Carregar metadados
        metadata_file = os.path.join(path, "training_metadata.pkl")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                self.model_metadata = pickle.load(f)
        
        print(f"📂 {len(self.models)} modelos carregados")

# Instância global do treinador
model_trainer = ModelTrainer()