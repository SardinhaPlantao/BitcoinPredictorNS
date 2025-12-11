"""
VALIDADOR DE DADOS
Valida e limpa dados coletados de todas as fontes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class DataValidator:
    """
    Valida e limpa dados de múltiplas fontes
    """
    
    def __init__(self):
        """Inicializa o validador"""
        self.validation_rules = self._initialize_validation_rules()
    
    def _initialize_validation_rules(self) -> Dict[str, Dict]:
        """Inicializa regras de validação por tipo de dado"""
        return {
            'bitcoin': {
                'required_columns': ['Open', 'High', 'Low', 'Close', 'Volume'],
                'price_limits': (1, 1000000),  # Preços realistas do BTC
                'volume_limits': (0, 1e12),     # Volume realista
                'date_range': (datetime(2010, 1, 1), datetime.now()),
                'min_data_points': 100
            },
            'macro': {
                'required_coverage': 0.5,  # Mínimo 50% de cobertura
                'date_range': (datetime(1980, 1, 1), datetime.now()),
                'outlier_threshold': 5,    # Desvios padrão para outliers
                'min_indicators': 5
            },
            'related_assets': {
                'required_columns': ['Close'],
                'price_limits': (0, None),  # Sem limite superior
                'min_data_points': 50
            }
        }
    
    def validate_dataset(self, 
                        data: pd.DataFrame, 
                        data_type: str,
                        asset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Valida um dataset completo
        
        Args:
            data: DataFrame para validar
            data_type: Tipo de dados ('bitcoin', 'macro', 'related_assets')
            asset_name: Nome do ativo (para mensagens de erro)
        
        Returns:
            Dicionário com relatório de validação
        """
        if data_type not in self.validation_rules:
            raise ValueError(f"Tipo de dados desconhecido: {data_type}")
        
        rules = self.validation_rules[data_type]
        asset_name = asset_name or data_type
        
        report = {
            'asset': asset_name,
            'data_type': data_type,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'is_valid': True,
            'issues': [],
            'warnings': [],
            'statistics': {},
            'passed_checks': 0,
            'failed_checks': 0
        }
        
        # Verificação 1: DataFrame não vazio
        if data.empty:
            report['is_valid'] = False
            report['issues'].append('DataFrame vazio')
            report['failed_checks'] += 1
            return report
        
        report['statistics']['n_rows'] = len(data)
        report['statistics']['n_columns'] = len(data.columns)
        report['passed_checks'] += 1
        
        # Verificação 2: Número mínimo de pontos
        if 'min_data_points' in rules and len(data) < rules['min_data_points']:
            report['is_valid'] = False
            report['issues'].append(f'Poucos dados: {len(data)} < {rules["min_data_points"]}')
            report['failed_checks'] += 1
        else:
            report['passed_checks'] += 1
        
        # Verificação 3: Colunas requeridas
        if 'required_columns' in rules:
            missing_cols = [col for col in rules['required_columns'] 
                          if col not in data.columns]
            
            if missing_cols:
                report['issues'].append(f'Colunas faltando: {missing_cols}')
                report['failed_checks'] += 1
            else:
                report['passed_checks'] += 1
        
        # Verificação 4: Faixa de datas
        if 'date_range' in rules:
            min_date, max_date = rules['date_range']
            
            if data.index[0] < min_date:
                report['warnings'].append(f'Data inicial muito antiga: {data.index[0]}')
            
            if data.index[-1] > max_date:
                report['warnings'].append(f'Data final no futuro: {data.index[-1]}')
            
            report['passed_checks'] += 1
        
        # Verificação 5: Valores NaN
        nan_counts = data.isna().sum()
        total_nan = nan_counts.sum()
        nan_percentage = total_nan / (len(data) * len(data.columns)) * 100
        
        report['statistics']['total_nan'] = int(total_nan)
        report['statistics']['nan_percentage'] = float(nan_percentage)
        
        if nan_percentage > 30:  # Mais de 30% NaN
            report['is_valid'] = False
            report['issues'].append(f'Muitos valores NaN: {nan_percentage:.1f}%')
            report['failed_checks'] += 1
        elif nan_percentage > 10:  # 10-30% NaN
            report['warnings'].append(f'Valores NaN moderados: {nan_percentage:.1f}%')
            report['passed_checks'] += 1
        else:
            report['passed_checks'] += 1
        
        # Verificação 6: Valores extremos (apenas para colunas numéricas)
        if data_type in ['bitcoin', 'related_assets'] and 'price_limits' in rules:
            price_min, price_max = rules['price_limits']
            
            if 'Close' in data.columns:
                prices = data['Close']
                
                # Verificar preços não-positivos
                non_positive = (prices <= 0).sum()
                if non_positive > 0:
                    report['is_valid'] = False
                    report['issues'].append(f'{non_positive} preços não-positivos')
                    report['failed_checks'] += 1
                
                # Verificar limites de preço
                if price_min is not None:
                    below_min = (prices < price_min).sum()
                    if below_min > 0:
                        report['issues'].append(f'{below_min} preços abaixo do mínimo')
                        report['failed_checks'] += 1
                
                if price_max is not None:
                    above_max = (prices > price_max).sum()
                    if above_max > 0:
                        report['issues'].append(f'{above_max} preços acima do máximo')
                        report['failed_checks'] += 1
                
                report['passed_checks'] += 1
        
        # Verificação 7: Consistência temporal
        if len(data) > 1:
            time_diff = data.index.to_series().diff().dt.days
            
            # Verificar gaps grandes
            large_gaps = (time_diff > 7).sum()
            if large_gaps > 0:
                report['warnings'].append(f'{large_gaps} gaps temporais > 7 dias')
            
            # Verificar datas duplicadas
            duplicates = data.index.duplicated().sum()
            if duplicates > 0:
                report['issues'].append(f'{duplicates} datas duplicadas')
                report['failed_checks'] += 1
            
            report['passed_checks'] += 1
        
        # Resumo
        report['statistics']['pass_rate'] = report['passed_checks'] / (report['passed_checks'] + report['failed_checks'])
        
        if report['is_valid'] and report['failed_checks'] > 0:
            report['is_valid'] = False
        
        return report
    
    def clean_dataset(self, 
                     data: pd.DataFrame,
                     data_type: str) -> pd.DataFrame:
        """
        Limpa um dataset com base no tipo
        
        Args:
            data: DataFrame para limpar
            data_type: Tipo de dados
        
        Returns:
            DataFrame limpo
        """
        if data.empty:
            return data
        
        cleaned = data.copy()
        
        # 1. Remover duplicatas de índice
        cleaned = cleaned[~cleaned.index.duplicated(keep='first')]
        
        # 2. Ordenar por data
        cleaned = cleaned.sort_index()
        
        # 3. Preencher valores NaN
        # Primeiro forward fill, depois backward fill
        cleaned = cleaned.ffill().bfill()
        
        # 4. Para dados de preços: remover valores não-positivos
        if data_type in ['bitcoin', 'related_assets']:
            if 'Close' in cleaned.columns:
                # Substituir valores <= 0 pela média móvel
                mask = cleaned['Close'] <= 0
                if mask.any():
                    cleaned.loc[mask, 'Close'] = cleaned['Close'].rolling(20, min_periods=1).mean()
        
        # 5. Remover outliers extremos (para dados macro)
        if data_type == 'macro':
            for col in cleaned.columns:
                if cleaned[col].dtype in [np.float64, np.int64]:
                    # Usar método IQR
                    Q1 = cleaned[col].quantile(0.25)
                    Q3 = cleaned[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 5 * IQR  # 5x IQR para ser conservador
                    upper_bound = Q3 + 5 * IQR
                    
                    # Substituir outliers pela mediana
                    outlier_mask = (cleaned[col] < lower_bound) | (cleaned[col] > upper_bound)
                    if outlier_mask.any():
                        median_val = cleaned[col].median()
                        cleaned.loc[outlier_mask, col] = median_val
        
        return cleaned
    
    def align_datasets(self, 
                      *datasets: pd.DataFrame,
                      method: str = 'inner') -> List[pd.DataFrame]:
        """
        Alinha múltiplos datasets temporalmente
        
        Args:
            *datasets: DataFrames para alinhar
            method: Método de alinhamento ('inner', 'outer', 'left', 'right')
        
        Returns:
            Lista de DataFrames alinhados
        """
        if not datasets:
            return []
        
        # Encontrar índice comum
        common_index = datasets[0].index
        
        for dataset in datasets[1:]:
            if method == 'inner':
                common_index = common_index.intersection(dataset.index)
            elif method == 'outer':
                common_index = common_index.union(dataset.index)
            elif method == 'left':
                pass  # Manter índice do primeiro
            elif method == 'right':
                common_index = dataset.index
            else:
                raise ValueError(f"Método {method} não suportado")
        
        # Alinhar todos os datasets
        aligned_datasets = []
        
        for dataset in datasets:
            aligned = dataset.reindex(common_index)
            
            # Preencher valores NaN resultantes do reindex
            if method in ['outer', 'right']:
                aligned = aligned.ffill().bfill()
            
            aligned_datasets.append(aligned)
        
        return aligned_datasets
    
    def create_validation_report(self, 
                               validation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Cria relatório consolidado de validação
        
        Args:
            validation_results: Lista de resultados de validação
        
        Returns:
            Relatório consolidado
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_datasets': len(validation_results),
            'valid_datasets': 0,
            'invalid_datasets': 0,
            'total_issues': 0,
            'total_warnings': 0,
            'datasets': [],
            'summary': {}
        }
        
        all_issues = []
        all_warnings = []
        
        for result in validation_results:
            report['datasets'].append({
                'asset': result.get('asset', 'unknown'),
                'data_type': result.get('data_type', 'unknown'),
                'is_valid': result.get('is_valid', False),
                'n_issues': len(result.get('issues', [])),
                'n_warnings': len(result.get('warnings', []))
            })
            
            if result.get('is_valid', False):
                report['valid_datasets'] += 1
            else:
                report['invalid_datasets'] += 1
            
            all_issues.extend(result.get('issues', []))
            all_warnings.extend(result.get('warnings', []))
        
        report['total_issues'] = len(all_issues)
        report['total_warnings'] = len(all_warnings)
        
        # Resumo por tipo de problema
        issue_types = {}
        for issue in all_issues:
            issue_type = issue.split(':')[0] if ':' in issue else issue
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        warning_types = {}
        for warning in all_warnings:
            warning_type = warning.split(':')[0] if ':' in warning else warning
            warning_types[warning_type] = warning_types.get(warning_type, 0) + 1
        
        report['summary'] = {
            'issue_types': issue_types,
            'warning_types': warning_types,
            'validity_rate': report['valid_datasets'] / report['total_datasets'] * 100 if report['total_datasets'] > 0 else 0
        }
        
        return report

# Instância global do validador
data_validator = DataValidator()