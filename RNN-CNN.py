import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten, Dense, Input

# 1. Verileri Yükle
train_df = pd.read_csv('DailyDelhiClimateTrain.csv', parse_dates=['date'])
test_df = pd.read_csv('DailyDelhiClimateTest.csv', parse_dates=['date'])

# 2. Girdi olarak tüm sütunları kullan
feature_cols = ['meantemp', 'humidity', 'wind_speed', 'meanpressure']
target_col = 'meantemp'

train_features = train_df[feature_cols].copy()
test_features = test_df[feature_cols].copy()

# 3. Normalize Et
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_features)
test_scaled = scaler.transform(test_features)

# 4. Sekans oluşturma fonksiyonu (y target sadece sıcaklık olacak)
def create_sequences_multivariate(data, target_index, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size][target_index])  # sadece sıcaklık
    return np.array(X), np.array(y)

window_size = 10
target_index = feature_cols.index('meantemp')
X_train, y_train = create_sequences_multivariate(train_scaled, target_index, window_size)
X_test, y_test = create_sequences_multivariate(test_scaled, target_index, window_size)

# --- RNN (LSTM) Modeli --- (Sadece sıcaklık kullanır)
rnn_train_temp = train_df[['meantemp']]
test_temp = test_df[['meantemp']]
rnn_train_scaled = MinMaxScaler().fit_transform(rnn_train_temp)
rnn_test_scaled = MinMaxScaler().fit_transform(test_temp)

# Tek değişkenli sekans oluşturma fonksiyonu (RNN için)
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

X_train_rnn, y_train_rnn = create_sequences(rnn_train_scaled, window_size)
X_test_rnn, y_test_rnn = create_sequences(rnn_test_scaled, window_size)

rnn_model = Sequential([
    Input(shape=(window_size, 1)),
    LSTM(64),
    Dense(1)
])
rnn_model.compile(optimizer='adam', loss='mse')
rnn_model.fit(X_train_rnn, y_train_rnn, epochs=100, batch_size=16, validation_split=0.1, verbose=1)

# --- CNN Modeli (tüm özellikler kullanılır) ---
cnn_model = Sequential([
    Input(shape=(window_size, len(feature_cols))),
    Conv1D(64, kernel_size=2, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)
])
cnn_model.compile(optimizer='adam', loss='mse')
cnn_model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.1, verbose=1)

# --- Tahminler ---
rnn_pred = rnn_model.predict(X_test_rnn)
cnn_pred = cnn_model.predict(X_test)

# --- Geri ölçekleme
rnn_scaler = MinMaxScaler()
rnn_scaler.fit(rnn_train_temp)
rnn_pred_inv = rnn_scaler.inverse_transform(rnn_pred)
cnn_pred_inv = scaler.inverse_transform(np.concatenate([cnn_pred, np.zeros((cnn_pred.shape[0], len(feature_cols) - 1))], axis=1))[:, 0]
actual_temp = test_df['meantemp'].values[window_size:]

# --- Performans Metriği ---
def print_metrics(name, actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    print(f"{name} - MAE: {mae:.3f} | RMSE: {rmse:.3f}")

print_metrics("RNN", actual_temp, rnn_pred_inv.flatten())
print_metrics("CNN (Tüm Özelliklerle)", actual_temp, cnn_pred_inv.flatten())

# --- Grafik Karşılaştırma ---
plt.figure(figsize=(14, 6))
plt.plot(actual_temp, label='Gerçek Sıcaklık', color='black')
plt.plot(rnn_pred_inv, label='RNN Tahmini', linestyle='--')
plt.plot(cnn_pred_inv, label='CNN Tahmini', linestyle=':')
plt.title('Sıcaklık Tahmini Karşılaştırması (RNN vs CNN - Çoklu Girdi)')
plt.xlabel('Gün')
plt.ylabel('Sıcaklık (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
