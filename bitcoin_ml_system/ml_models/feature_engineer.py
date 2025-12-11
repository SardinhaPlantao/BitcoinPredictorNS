"""
ENGENHARIA DE FEATURES
Cria features para modelos ML combinando todas as fontes de dados
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from core.config import config
from utils.helpers import calculate_returns, align_time_series, create_lagged_features

class FeatureEngineer:
    """
    Sistema completo de engenharia de features para previsão do Bitcoin
    """
    
    def __init__(self):
        """Inicializa o engenheiro de features"""
        self.scalers = {}
        self.feature_cache = {}
        self.feature_metadata = {}
        
    def create_features_pipeline(self,
                                btc_data: pd.DataFrame,
                                macro_data: pd.DataFrame,
                                technical_data: pd.DataFrame,
                                target_horizon: int = 30,
                                include_lags: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Pipeline completo de criação de features
        
        Args:
            btc_data: Dados Bitcoin
            macro_data: Dados macro
            technical_data: Dados técnicos
            target_horizon: Horizonte de previsão em dias
            include_lags: Incluir features com lag
        
        Returns:
            Tuple (features, target)
        """
        print(f"\n🔧 Criando features para horizonte {target_horizon} dias")
        
        # 1. Preparar dados básicos
        base_features = self._prepare_base_features(btc_data)
        
        # 2. Adicionar features macro
        if macro_data is not None and not macro_data.empty:
            macro_features = self._create_macro_features(macro_data)
            base_features = self._merge_features(base_features, macro_features)
        
        # 3. Adicionar features técnicas
        if technical_data is not None and not technical_data.empty:
            technical_features = self._create_technical_features(technical_data)
            base_features = self._merge_features(base_features, technical_features)
        
        # 4. Criar features derivadas
        derived_features = self._create_derived_features(base_features)
        base_features = self._merge_features(base_features, derived_features)
        
        # 5. Adicionar lags se solicitado
        if include_lags:
            lagged_features = self._create_lagged_features(base_features)
            base_features = self._merge_features(base_features, lagged_features)
        
        # 6. Criar target (preço futuro)
        target = self._create_target(btc_data, target_horizon)
        
        # 7. Alinhar features e target
        features, target_aligned = self._align_features_target(base_features, target)
        
        # 8. Processamento final
        features_processed = self._process_features(features)
        
        # Salvar metadados
        self.feature_metadata = {
            'n_features': len(features_processed.columns),
            'n_samples': len(features_processed),
            'feature_names': list(features_processed.columns),
            'target_horizon': target_horizon,
            'creation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        print(f"✅ {len(features_processed.columns)} features criadas, {len(features_processed)} amostras")
        return features_processed, target_aligned
    
    def _prepare_base_features(self, btc_data: pd.DataFrame) -> pd.DataFrame:
        """Prepara features básicas do Bitcoin"""
        features = pd.DataFrame(index=btc_data.index)
        
        # Preços
        if 'Close' in btc_data.columns:
            features['price'] = btc_data['Close']
            features['log_price'] = np.log(btc_data['Close'])
        
        # Retornos
        if 'Returns' in btc_data.columns:
            features['returns'] = btc_data['Returns']
            features['log_returns'] = np.log(1 + btc_data['Returns'])
        
        # Volatilidade
        if 'Volatility' in btc_data.columns:
            features['volatility'] = btc_data['Volatility']
        
        # Volume
        if 'Volume' in btc_data.columns:
            features['volume'] = btc_data['Volume']
            features['log_volume'] = np.log(btc_data['Volume'])
            
            if 'Volume_SMA_20' in btc_data.columns:
                features['volume_ratio'] = btc_data['Volume'] / btc_data['Volume_SMA_20']
        
        # Features temporais
        features['day_of_week'] = btc_data.index.dayofweek
        features['month'] = btc_data.index.month
        features['quarter'] = btc_data.index.quarter
        features['day_of_year'] = btc_data.index.dayofyear
        
        # Sazonalidade
        features['sin_day'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
        features['cos_day'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
        features['sin_week'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['cos_week'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        return features
    
    def _create_macro_features(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Cria features macroeconômicas"""
        features = pd.DataFrame(index=macro_data.index)
        
        # Features básicas do macro
        for col in macro_data.columns:
            # Normalizar nome da feature
            feature_name = f"macro_{col.lower()}"
            features[feature_name] = macro_data[col]
            
            # Adicionar mudanças percentuais
            if macro_data[col].dtype in [np.float64, np.int64]:
                for period in [7, 30, 90]:
                    features[f"{feature_name}_pct_{period}"] = macro_data[col].pct_change(period)
        
        # Features macro derivadas
        derived_features = self._create_macro_derived_features(macro_data)
        features = pd.concat([features, derived_features], axis=1)
        
        return features
    
    def _create_macro_derived_features(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Cria features macro derivadas"""
        derived = pd.DataFrame(index=macro_data.index)
        
        # 1. Liquidez
        if 'M2SL' in macro_data.columns:
            derived['liquidity_growth'] = macro_data['M2SL'].pct_change(365) * 100
        
        # 2. Yield curve
        if 'T10Y2Y' in macro_data.columns:
            derived['yield_curve'] = macro_data['T10Y2Y']
            derived['yield_curve_inverted'] = (macro_data['T10Y2Y'] < 0).astype(int)
        
        # 3. Inflação
        if 'CPIAUCSL' in macro_data.columns:
            derived['inflation_yoy'] = macro_data['CPIAUCSL'].pct_change(365) * 100
        
        # 4. Condições financeiras
        components = []
        
        if 'VIXCLS' in macro_data.columns:
            vix_norm = (macro_data['VIXCLS'] - macro_data['VIXCLS'].rolling(200).mean()) / \
                      macro_data['VIXCLS'].rolling(200).std()
            components.append(vix_norm)
        
        if 'T10Y2Y' in macro_data.columns:
            yc_norm = (macro_data['T10Y2Y'] - macro_data['T10Y2Y'].rolling(200).mean()) / \
                     macro_data['T10Y2Y'].rolling(200).std()
            components.append(yc_norm)
        
        if components:
            financial_conditions = pd.concat(components, axis=1).mean(axis=1)
            derived['financial_conditions'] = financial_conditions
        
        # 5. Difusão index (simplificado)
        indicator_cols = [col for col in macro_data.columns 
                         if col in config.MACRO_INDICATORS.values()]
        
        if indicator_cols:
            # Contar indicadores acima da média móvel
            above_ma = pd.DataFrame()
            for col in indicator_cols[:10]:  # Limitar a 10 indicadores
                ma = macro_data[col].rolling(200).mean()
                above_ma[col] = (macro_data[col] > ma).astype(int)
            
            derived['diffusion_index'] = above_ma.mean(axis=1) * 100
        
        return derived
    
    def _create_technical_features(self, technical_data: pd.DataFrame) -> pd.DataFrame:
        """Cria features técnicas"""
        features = pd.DataFrame(index=technical_data.index)
        
        # Selecionar colunas técnicas importantes
        tech_columns = [
            'rsi_14', 'macd_histogram', 'bb_position', 'volatility_30',
            'momentum_30', 'dist_sma_50', 'atr_percent'
        ]
        
        for col in tech_columns:
            if col in technical_data.columns:
                features[f"tech_{col}"] = technical_data[col]
        
        # Adicionar sinais compostos se disponíveis
        if 'composite_signal' in technical_data.columns:
            features['tech_signal'] = technical_data['composite_signal']
        
        if 'signal_strength' in technical_data.columns:
            # Codificar sinal strength
            strength_map = {
                'Strong Sell': -2,
                'Sell': -1,
                'Neutral': 0,
                'Buy': 1,
                'Strong Buy': 2
            }
            features['tech_strength'] = technical_data['signal_strength'].map(strength_map)
        
        return features
    
    def _create_derived_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Cria features derivadas complexas"""
        derived = pd.DataFrame(index=features.index)
        
        # 1. Interações entre features
        if 'price' in features.columns and 'volume' in features.columns:
            derived['price_volume_interaction'] = features['price'] * features['volume']
            derived['price_volume_ratio'] = features['price'] / features['volume'].replace(0, np.nan)
        
        # 2. Features estatísticas rolling
        if 'returns' in features.columns:
            for window in [7, 14, 30]:
                derived[f'returns_skew_{window}'] = features['returns'].rolling(window).skew()
                derived[f'returns_kurtosis_{window}'] = features['returns'].rolling(window).kurt()
                derived[f'returns_std_{window}'] = features['returns'].rolling(window).std()
        
        # 3. Ratio features
        tech_cols = [col for col in features.columns if col.startswith('tech_')]
        macro_cols = [col for col in features.columns if col.startswith('macro_')]
        
        if tech_cols and macro_cols:
            # Usar primeiras colunas de cada tipo
            tech_col = tech_cols[0] if tech_cols else 'returns'
            macro_col = macro_cols[0] if macro_cols else 'price'
            
            derived['tech_macro_ratio'] = features[tech_col] / features[macro_col].replace(0, np.nan)
        
        # 4. Regime features
        if 'volatility' in features.columns:
            # Identificar regimes de volatilidade
            vol_ma = features['volatility'].rolling(30).mean()
            vol_std = features['volatility'].rolling(30).std()
            
            derived['high_vol_regime'] = (features['volatility'] > vol_ma + vol_std).astype(int)
            derived['low_vol_regime'] = (features['volatility'] < vol_ma - vol_std).astype(int)
        
        # 5. Trend features
        if 'price' in features.columns:
            for window in [10, 20, 50]:
                # Slope da regressão linear
                slopes = []
                for i in range(window, len(features)):
                    y = features['price'].iloc[i-window:i].values
                    x = np.arange(len(y))
                    
                    if len(y) > 1:
                        slope = np.polyfit(x, y, 1)[0]
                        slopes.append(slope)
                    else:
                        slopes.append(0)
                
                # Preencher valores iniciais
                slopes = [0] * (window) + slopes
                derived[f'trend_slope_{window}'] = slopes[:len(features)]
        
        return derived
    
    def _create_lagged_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Cria features com lag"""
        lagged_features = pd.DataFrame(index=features.index)
        
        # Selecionar colunas para criar lags
        columns_to_lag = []
        
        # Preços e retornos
        for col in ['price', 'returns', 'log_returns', 'volume']:
            if col in features.columns:
                columns_to_lag.append(col)
        
        # Indicadores técnicos
        tech_cols = [col for col in features.columns if col.startswith('tech_')]
        columns_to_lag.extend(tech_cols[:5])  # Primeiros 5 indicadores técnicos
        
        # Indicadores macro
        macro_cols = [col for col in features.columns if col.startswith('macro_')]
        columns_to_lag.extend(macro_cols[:5])  # Primeiros 5 indicadores macro
        
        # Criar lags
        for col in columns_to_lag:
            for lag in [1, 2, 3, 5, 7, 14, 30]:
                lagged_features[f'{col}_lag_{lag}'] = features[col].shift(lag)
            
            # Diferenças
            for diff in [1, 3, 7]:
                lagged_features[f'{col}_diff_{diff}'] = features[col].diff(diff)
        
        return lagged_features
    
    def _create_target(self, btc_data: pd.DataFrame, horizon: int) -> pd.Series:
        """Cria target para previsão (preço futuro)"""
        if 'Close' not in btc_data.columns:
            raise ValueError("DataFrame deve conter coluna 'Close'")
        
        # Preço futuro (shift negativo)
        future_prices = btc_data['Close'].shift(-horizon)
        
        # Retorno futuro
        future_returns = (future_prices / btc_data['Close'] - 1) * 100
        
        # Classificação binária (alta/baixa)
        future_direction = (future_returns > 0).astype(int)
        
        # Para regressão, usar retorno futuro
        # Para classificação, usar direção futura
        target = future_returns
        
        # Nomear target
        target.name = f'target_{horizon}d'
        
        return target
    
    def _align_features_target(self, 
                              features: pd.DataFrame, 
                              target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Alinha features e target temporalmente"""
        # Remover NaN do target
        valid_mask = target.notna()
        
        if not valid_mask.any():
            raise ValueError("Target contém apenas valores NaN")
        
        # Alinhar features com target válido
        features_aligned = features[valid_mask]
        target_aligned = target[valid_mask]
        
        # Remover NaN das features
        features_aligned = features_aligned.ffill().bfill()
        
        # Garantir que não há NaN
        if features_aligned.isna().any().any() or target_aligned.isna().any():
            raise ValueError("Features ou target ainda contêm NaN após alinhamento")
        
        return features_aligned, target_aligned
    
    def _merge_features(self, 
                       base: pd.DataFrame, 
                       new: pd.DataFrame) -> pd.DataFrame:
        """Mescla features mantendo alinhamento temporal"""
        if base.empty:
            return new
        
        if new.empty:
            return base
        
        # Encontrar índice comum
        common_index = base.index.intersection(new.index)
        
        if len(common_index) < 10:
            print(f"⚠️ Poucos dados comuns ao mesclar features: {len(common_index)}")
            return base
        
        # Alinhar e mesclar
        base_aligned = base.loc[common_index]
        new_aligned = new.loc[common_index]
        
        # Remover colunas duplicadas
        result = pd.concat([base_aligned, new_aligned], axis=1)
        result = result.loc[:, ~result.columns.duplicated()]
        
        return result
    
    def _process_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Processamento final das features"""
        processed = features.copy()
        
        # 1. Remover colunas com muitos NaN
        nan_threshold = 0.3  # 30%
        cols_to_keep = processed.columns[processed.isna().mean() < nan_threshold]
        processed = processed[cols_to_keep]
        
        # 2. Preencher NaN restantes
        processed = processed.ffill().bfill()
        
        # 3. Remover colunas constantes
        constant_cols = processed.columns[processed.nunique() <= 1]
        if len(constant_cols) > 0:
            processed = processed.drop(columns=constant_cols)
        
        # 4. Normalizar features (exceto target se presente)
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Usar RobustScaler para reduzir impacto de outliers
            scaler = RobustScaler()
            processed[col] = scaler.fit_transform(processed[[col]])
            
            # Salvar scaler para uso futuro
            self.scalers[col] = scaler
        
        return processed
    
    def reduce_dimensionality(self,
                            features: pd.DataFrame,
                            n_components: Optional[int] = None,
                            variance_threshold: float = 0.95) -> pd.DataFrame:
        """
        Reduz dimensionalidade usando PCA
        
        Args:
            features: DataFrame com features
            n_components: Número de componentes (None para automático)
            variance_threshold: Variância mínima a explicar
        
        Returns:
            DataFrame com features reduzidas
        """
        print(f"\n📉 Reduzindo dimensionalidade (variance={variance_threshold})")
        
        if features.empty:
            return features
        
        # Selecionar apenas colunas numéricas
        numeric_features = features.select_dtypes(include=[np.number])
        
        if numeric_features.shape[1] < 2:
            print("⚠️ Não há features suficientes para PCA")
            return features
        
        # Aplicar PCA
        if n_components is None:
            # PCA automático baseado em variância
            pca = PCA(n_components=variance_threshold)
        else:
            pca = PCA(n_components=min(n_components, numeric_features.shape[1]))
        
        # Transformar features
        pca_features = pca.fit_transform(numeric_features)
        
        # Criar DataFrame com componentes
        if pca.n_components_ < numeric_features.shape[1]:
            component_names = [f'pca_component_{i+1}' for i in range(pca.n_components_)]
            pca_df = pd.DataFrame(pca_features, index=features.index, columns=component_names)
            
            print(f"✅ Dimensionalidade reduzida: {numeric_features.shape[1]} -> {pca.n_components_}")
            print(f"   Variância explicada: {pca.explained_variance_ratio_.sum():.3f}")
            
            # Adicionar features não numéricas de volta
            non_numeric = features.select_dtypes(exclude=[np.number])
            result = pd.concat([pca_df, non_numeric], axis=1)
            
            return result
        else:
            print("⚠️ PCA não reduziu dimensionalidade")
            return features
    
    def calculate_feature_importance(self,
                                   features: pd.DataFrame,
                                   target: pd.Series,
                                   method: str = 'mutual_info') -> pd.DataFrame:
        """
        Calcula importância das features
        
        Args:
            features: DataFrame com features
            target: Série com target
            method: Método ('mutual_info', 'correlation', 'variance')
        
        Returns:
            DataFrame com importância das features
        """
        print(f"\n🎯 Calculando importância das features ({method})")
        
        if features.empty or target.empty:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame(index=features.columns)
        importance_df['feature'] = features.columns
        
        if method == 'mutual_info':
            # Informação mútua
            mi_scores = mutual_info_regression(features, target, random_state=42)
            importance_df['importance'] = mi_scores
            importance_df['importance_type'] = 'mutual_info'
        
        elif method == 'correlation':
            # Correlação com target
            corr_scores = features.apply(lambda x: x.corr(target))
            importance_df['importance'] = corr_scores.abs()
            importance_df['importance_type'] = 'correlation'
        
        elif method == 'variance':
            # Variância das features
            var_scores = features.var()
            importance_df['importance'] = var_scores
            importance_df['importance_type'] = 'variance'
        
        else:
            raise ValueError(f"Método {method} não suportado")
        
        # Normalizar importância para 0-100
        if importance_df['importance'].max() > 0:
            importance_df['importance_normalized'] = (
                importance_df['importance'] / importance_df['importance'].max() * 100
            )
        else:
            importance_df['importance_normalized'] = 0
        
        # Ordenar por importância
        importance_df = importance_df.sort_values('importance_normalized', ascending=False)
        
        print(f"✅ {len(importance_df)} features avaliadas")
        print(f"   Top 5: {importance_df.head(5)['feature'].tolist()}")
        
        return importance_df
    
    def generate_feature_report(self,
                              features: pd.DataFrame,
                              target: pd.Series) -> Dict[str, Any]:
        """
        Gera relatório completo das features
        
        Args:
            features: DataFrame com features
            target: Série com target
        
        Returns:
            Relatório das features
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {},
            'feature_types': {},
            'statistics': {},
            'importance': {}
        }
        
        # Sumário
        report['summary'] = {
            'n_features': len(features.columns),
            'n_samples': len(features),
            'feature_categories': self._categorize_features(features),
            'missing_values': int(features.isna().sum().sum()),
            'missing_percentage': float(features.isna().sum().sum() / (len(features) * len(features.columns)) * 100)
        }
        
        # Tipos de dados
        dtypes = features.dtypes.value_counts().to_dict()
        report['feature_types'] = {str(k): int(v) for k, v in dtypes.items()}
        
        # Estatísticas
        numeric_features = features.select_dtypes(include=[np.number])
        
        if not numeric_features.empty:
            report['statistics'] = {
                'mean': numeric_features.mean().to_dict(),
                'std': numeric_features.std().to_dict(),
                'min': numeric_features.min().to_dict(),
                'max': numeric_features.max().to_dict(),
                'correlation_with_target': self._calculate_feature_target_correlation(numeric_features, target)
            }
        
        # Importância
        importance = self.calculate_feature_importance(features, target, 'mutual_info')
        
        if not importance.empty:
            report['importance'] = {
                'top_10': importance.head(10).to_dict('records'),
                'bottom_10': importance.tail(10).to_dict('records'),
                'distribution': self._calculate_importance_distribution(importance)
            }
        
        return report
    
    def _categorize_features(self, features: pd.DataFrame) -> Dict[str, int]:
        """Categoriza features por tipo"""
        categories = {
            'price_related': 0,
            'volume_related': 0,
            'technical': 0,
            'macro': 0,
            'temporal': 0,
            'derived': 0,
            'lagged': 0,
            'other': 0
        }
        
        for col in features.columns:
            col_lower = col.lower()
            
            if any(term in col_lower for term in ['price', 'return', 'close', 'open', 'high', 'low']):
                categories['price_related'] += 1
            elif any(term in col_lower for term in ['volume', 'obv', 'mfi']):
                categories['volume_related'] += 1
            elif any(term in col_lower for term in ['rsi', 'macd', 'bb', 'atr', 'stoch', 'tech_']):
                categories['technical'] += 1
            elif any(term in col_lower for term in ['macro_', 'fed', 'yield', 'cpi', 'unemploy']):
                categories['macro'] += 1
            elif any(term in col_lower for term in ['day', 'month', 'quarter', 'year', 'week', 'season']):
                categories['temporal'] += 1
            elif any(term in col_lower for term in ['lag_', 'diff_', 'shift']):
                categories['lagged'] += 1
            elif any(term in col_lower for term in ['interaction', 'ratio', 'derived', 'composite']):
                categories['derived'] += 1
            else:
                categories['other'] += 1
        
        return categories
    
    def _calculate_feature_target_correlation(self,
                                            features: pd.DataFrame,
                                            target: pd.Series) -> Dict[str, float]:
        """Calcula correlação entre features e target"""
        correlations = {}
        
        for col in features.columns:
            if features[col].dtype in [np.float64, np.int64]:
                corr = features[col].corr(target)
                if not pd.isna(corr):
                    correlations[col] = float(corr)
        
        # Ordenar por correlação absoluta
        correlations = dict(sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        ))
        
        return correlations
    
    def _calculate_importance_distribution(self,
                                         importance_df: pd.DataFrame) -> Dict[str, float]:
        """Calcula distribuição da importância"""
        if 'importance_normalized' not in importance_df.columns:
            return {}
        
        importance = importance_df['importance_normalized']
        
        distribution = {
            'mean': float(importance.mean()),
            'median': float(importance.median()),
            'std': float(importance.std()),
            'q1': float(importance.quantile(0.25)),
            'q3': float(importance.quantile(0.75)),
            'max': float(importance.max()),
            'min': float(importance.min()),
            'high_importance_count': int((importance > 50).sum()),  # > 50%
            'medium_importance_count': int(((importance >= 20) & (importance <= 50)).sum()),
            'low_importance_count': int((importance < 20).sum())
        }
        
        return distribution

# Instância global do engenheiro de features
feature_engineer = FeatureEngineer()