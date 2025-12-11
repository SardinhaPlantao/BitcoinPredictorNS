"""
MODELOS DE DEEP LEARNING
LSTM, GRU e outras redes neurais para séries temporais
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    LSTM, GRU, Dense, Dropout, BatchNormalization,
    Conv1D, MaxPooling1D, Flatten, Bidirectional,
    Attention, MultiHeadAttention, LayerNormalization
)

from sklearn.preprocessing import MinMaxScaler

class DeepLearningModels:
    """
    Sistema completo de deep learning para previsão de séries temporais
    """
    
    def __init__(self):
        """Inicializa o sistema de deep learning"""
        self.models = {}
        self.scalers = {}
        self.histories = {}
        self.sequence_length = 60
        
    def prepare_sequences(self,
                         data: pd.DataFrame,
                         target_column: str = 'price',
                         sequence_length: int = 60,
                         test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara sequências para modelos de deep learning
        
        Args:
            data: DataFrame com dados
            target_column: Coluna alvo
            sequence_length: Comprimento da sequência
            test_size: Proporção de teste
        
        Returns:
            X_train, y_train, X_test, y_test
        """
        print(f"\n🧬 Preparando sequências (length={sequence_length})...")
        
        if target_column not in data.columns:
            raise ValueError(f"Coluna {target_column} não encontrada")
        
        # Normalizar dados
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[[target_column]])
        
        # Criar sequências
        X, y = [], []
        
        for i in range(sequence_length, len(scaled_data)):
            X.append(scaled_data[i-sequence_length:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape para [samples, time_steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Split train/test
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        self.scalers[target_column] = scaler
        self.sequence_length = sequence_length
        
        print(f"✅ Sequências preparadas: {X_train.shape[0]} treino, {X_test.shape[0]} teste")
        
        return X_train, y_train, X_test, y_test
    
    def create_lstm_model(self,
                         input_shape: Tuple[int, int],
                         units: List[int] = [50, 50, 50],
                         dropout_rate: float = 0.2,
                         learning_rate: float = 0.001) -> keras.Model:
        """
        Cria modelo LSTM
        
        Args:
            input_shape: Shape dos dados de entrada
            units: Número de unidades em cada camada LSTM
            dropout_rate: Taxa de dropout
            learning_rate: Taxa de aprendizado
        
        Returns:
            Modelo LSTM compilado
        """
        model = models.Sequential()
        
        # Primeira camada LSTM
        model.add(LSTM(
            units=units[0],
            return_sequences=True if len(units) > 1 else False,
            input_shape=input_shape
        ))
        model.add(Dropout(dropout_rate))
        
        # Camadas LSTM adicionais
        for i in range(1, len(units)):
            return_seq = i < len(units) - 1
            model.add(LSTM(units=units[i], return_sequences=return_seq))
            model.add(Dropout(dropout_rate))
        
        # Camadas densas
        model.add(Dense(units[0] // 2, activation='relu'))
        model.add(Dropout(dropout_rate / 2))
        model.add(Dense(units[0] // 4, activation='relu'))
        model.add(Dense(1))
        
        # Compilar
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def create_gru_model(self,
                        input_shape: Tuple[int, int],
                        units: List[int] = [50, 50, 50],
                        dropout_rate: float = 0.2,
                        learning_rate: float = 0.001) -> keras.Model:
        """
        Cria modelo GRU
        
        Args:
            input_shape: Shape dos dados de entrada
            units: Número de unidades em cada camada GRU
            dropout_rate: Taxa de dropout
            learning_rate: Taxa de aprendizado
        
        Returns:
            Modelo GRU compilado
        """
        model = models.Sequential()
        
        # Primeira camada GRU
        model.add(GRU(
            units=units[0],
            return_sequences=True if len(units) > 1 else False,
            input_shape=input_shape
        ))
        model.add(Dropout(dropout_rate))
        
        # Camadas GRU adicionais
        for i in range(1, len(units)):
            return_seq = i < len(units) - 1
            model.add(GRU(units=units[i], return_sequences=return_seq))
            model.add(Dropout(dropout_rate))
        
        # Camadas densas
        model.add(Dense(units[0] // 2, activation='relu'))
        model.add(Dropout(dropout_rate / 2))
        model.add(Dense(units[0] // 4, activation='relu'))
        model.add(Dense(1))
        
        # Compilar
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def create_cnn_lstm_model(self,
                             input_shape: Tuple[int, int],
                             cnn_filters: List[int] = [64, 128],
                             lstm_units: List[int] = [50, 50],
                             dropout_rate: float = 0.2,
                             learning_rate: float = 0.001) -> keras.Model:
        """
        Cria modelo CNN-LSTM híbrido
        
        Args:
            input_shape: Shape dos dados de entrada
            cnn_filters: Filtros CNN
            lstm_units: Unidades LSTM
            dropout_rate: Taxa de dropout
            learning_rate: Taxa de aprendizado
        
        Returns:
            Modelo CNN-LSTM compilado
        """
        model = models.Sequential()
        
        # Camadas CNN
        model.add(Conv1D(
            filters=cnn_filters[0],
            kernel_size=3,
            activation='relu',
            input_shape=input_shape
        ))
        model.add(MaxPooling1D(pool_size=2))
        
        if len(cnn_filters) > 1:
            model.add(Conv1D(filters=cnn_filters[1], kernel_size=3, activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
        
        # Camadas LSTM
        for i, units in enumerate(lstm_units):
            return_seq = i < len(lstm_units) - 1
            model.add(LSTM(units=units, return_sequences=return_seq))
            model.add(Dropout(dropout_rate))
        
        # Camadas densas
        model.add(Dense(lstm_units[-1] // 2, activation='relu'))
        model.add(Dropout(dropout_rate / 2))
        model.add(Dense(lstm_units[-1] // 4, activation='relu'))
        model.add(Dense(1))
        
        # Compilar
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def create_attention_model(self,
                              input_shape: Tuple[int, int],
                              units: int = 50,
                              num_heads: int = 4,
                              dropout_rate: float = 0.2,
                              learning_rate: float = 0.001) -> keras.Model:
        """
        Cria modelo com mecanismo de atenção
        
        Args:
            input_shape: Shape dos dados de entrada
            units: Número de unidades
            num_heads: Número de heads de atenção
            dropout_rate: Taxa de dropout
            learning_rate: Taxa de aprendizado
        
        Returns:
            Modelo com atenção compilado
        """
        inputs = layers.Input(shape=input_shape)
        
        # Camada LSTM
        lstm_out = LSTM(units, return_sequences=True)(inputs)
        lstm_out = Dropout(dropout_rate)(lstm_out)
        
        # Mecanismo de atenção
        attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=units // num_heads
        )(lstm_out, lstm_out)
        
        # Normalização e dropout
        attention = LayerNormalization()(attention + lstm_out)
        attention = Dropout(dropout_rate)(attention)
        
        # Global pooling
        pooled = layers.GlobalAveragePooling1D()(attention)
        
        # Camadas densas
        dense = Dense(units // 2, activation='relu')(pooled)
        dense = Dropout(dropout_rate / 2)(dense)
        dense = Dense(units // 4, activation='relu')(dense)
        
        # Saída
        outputs = Dense(1)(dense)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compilar
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return model
    
    def train_model(self,
                   model: keras.Model,
                   X_train: np.ndarray,
                   y_train: np.ndarray,
                   X_val: np.ndarray,
                   y_val: np.ndarray,
                   epochs: int = 100,
                   batch_size: int = 32,
                   model_name: str = 'lstm') -> Dict[str, Any]:
        """
        Treina modelo de deep learning
        
        Args:
            model: Modelo a treinar
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação
            y_val: Target de validação
            epochs: Número de épocas
            batch_size: Tamanho do batch
            model_name: Nome do modelo
        
        Returns:
            Histórico de treinamento e modelo treinado
        """
        print(f"\n🏋️‍♂️ Treinando {model_name}...")
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                f'best_{model_name}_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Treinar
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Salvar
        self.models[model_name] = model
        self.histories[model_name] = history.history
        
        # Avaliar
        train_loss = model.evaluate(X_train, y_train, verbose=0)
        val_loss = model.evaluate(X_val, y_val, verbose=0)
        
        print(f"✅ {model_name} treinado")
        print(f"   Loss treino: {train_loss[0]:.6f}, validação: {val_loss[0]:.6f}")
        
        return {
            'model': model,
            'history': history.history,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_epoch': np.argmin(history.history['val_loss']) + 1
        }
    
    def predict_with_model(self,
                          model: keras.Model,
                          X: np.ndarray,
                          scaler: MinMaxScaler) -> np.ndarray:
        """
        Faz previsões e inverte normalização
        
        Args:
            model: Modelo treinado
            X: Dados de entrada
            scaler: Scaler usado na normalização
        
        Returns:
            Previsões desnormalizadas
        """
        # Fazer previsões
        predictions_scaled = model.predict(X, verbose=0)
        
        # Inverter normalização
        # Criar array dummy para inverse_transform
        dummy_array = np.zeros((len(predictions_scaled), 1))
        dummy_array[:, 0] = predictions_scaled[:, 0]
        
        predictions = scaler.inverse_transform(dummy_array)[:, 0]
        
        return predictions
    
    def predict_sequence(self,
                        model: keras.Model,
                        last_sequence: np.ndarray,
                        scaler: MinMaxScaler,
                        steps: int = 30) -> np.ndarray:
        """
        Previsão multi-step usando auto-regressão
        
        Args:
            model: Modelo treinado
            last_sequence: Última sequência conhecida
            scaler: Scaler usado na normalização
            steps: Número de passos à frente
        
        Returns:
            Previsões para os próximos steps
        """
        predictions = []
        current_sequence = last_sequence.copy()
        
        for step in range(steps):
            # Fazer previsão
            prediction_scaled = model.predict(
                current_sequence.reshape(1, -1, 1),
                verbose=0
            )[0, 0]
            
            # Inverter normalização
            dummy_array = np.array([[prediction_scaled]])
            prediction = scaler.inverse_transform(dummy_array)[0, 0]
            predictions.append(prediction)
            
            # Atualizar sequência
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = prediction_scaled
        
        return np.array(predictions)
    
    def create_ensemble_dl(self,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          input_shape: Tuple[int, int]) -> Dict[str, Any]:
        """
        Cria ensemble de modelos de deep learning
        
        Args:
            X_train: Features de treino
            y_train: Target de treino
            X_val: Features de validação
            y_val: Target de validação
            input_shape: Shape dos dados de entrada
        
        Returns:
            Ensemble de modelos DL
        """
        print("\n🎭 Criando ensemble de modelos DL...")
        
        # Criar diferentes arquiteturas
        models_config = [
            ('lstm_simple', self.create_lstm_model(input_shape, [50, 50])),
            ('lstm_deep', self.create_lstm_model(input_shape, [100, 100, 50])),
            ('gru', self.create_gru_model(input_shape, [50, 50, 50])),
            ('cnn_lstm', self.create_cnn_lstm_model(input_shape)),
            ('attention', self.create_attention_model(input_shape))
        ]
        
        trained_models = {}
        
        for name, model in models_config:
            print(f"  Treinando {name}...")
            
            result = self.train_model(
                model, X_train, y_train, X_val, y_val,
                epochs=50,  # Menos épocas para ensemble
                batch_size=32,
                model_name=name
            )
            
            trained_models[name] = {
                'model': result['model'],
                'train_loss': result['train_loss'][0],
                'val_loss': result['val_loss'][0]
            }
        
        # Criar ensemble (média das previsões)
        def ensemble_predict(X):
            predictions = []
            for name, model_info in trained_models.items():
                pred = model_info['model'].predict(X, verbose=0)
                predictions.append(pred)
            
            return np.mean(predictions, axis=0)
        
        # Avaliar ensemble
        ensemble_train_pred = ensemble_predict(X_train)
        ensemble_val_pred = ensemble_predict(X_val)
        
        train_metrics = self._calculate_dl_metrics(y_train, ensemble_train_pred)
        val_metrics = self._calculate_dl_metrics(y_val, ensemble_val_pred)
        
        results = {
            'trained_models': trained_models,
            'ensemble_predictor': ensemble_predict,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'model_performance': {
                name: info['val_loss'] 
                for name, info in trained_models.items()
            }
        }
        
        print(f"✅ Ensemble DL criado com {len(trained_models)} modelos")
        print(f"   RMSE validação: {val_metrics['rmse']:.6f}")
        
        return results
    
    def _calculate_dl_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula métricas para modelos DL"""
        mse = np.mean((y_true - y_pred) ** 2)
        mae = np.mean(np.abs(y_true - y_pred))
        
        # R² score
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mae),
            'r2': float(r2)
        }
    
    def visualize_training(self, history: Dict, model_name: str = "Model"):
        """Visualiza histórico de treinamento"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss
        axes[0].plot(history['loss'], label='Train Loss')
        axes[0].plot(history['val_loss'], label='Val Loss')
        axes[0].set_title(f'{model_name} - Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE
        axes[1].plot(history['mae'], label='Train MAE')
        axes[1].plot(history['val_mae'], label='Val MAE')
        axes[1].set_title(f'{model_name} - MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def save_models(self, path: str = "dl_models"):
        """Salva modelos de deep learning"""
        import os
        os.makedirs(path, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = os.path.join(path, f"{name}_model.h5")
            model.save(model_path)
        
        # Salvar scalers
        scalers_path = os.path.join(path, "scalers.pkl")
        with open(scalers_path, 'wb') as f:
            import pickle
            pickle.dump(self.scalers, f)
        
        print(f"💾 Modelos DL salvos em {path}")
    
    def load_models(self, path: str = "dl_models"):
        """Carrega modelos de deep learning"""
        import os
        import glob
        import pickle
        
        if not os.path.exists(path):
            print(f"⚠️ Diretório {path} não encontrado")
            return
        
        # Carregar modelos
        model_files = glob.glob(os.path.join(path, "*_model.h5"))
        
        for file in model_files:
            model = keras.models.load_model(file)
            name = os.path.basename(file).replace("_model.h5", "")
            self.models[name] = model
        
        # Carregar scalers
        scalers_file = os.path.join(path, "scalers.pkl")
        if os.path.exists(scalers_file):
            with open(scalers_file, 'rb') as f:
                self.scalers = pickle.load(f)
        
        print(f"📂 {len(self.models)} modelos DL carregados")

# Instância global de modelos DL
deep_learning = DeepLearningModels()