import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras import Input
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. Verileri Yükle
train_df = pd.read_csv('DailyDelhiClimateTrain.csv', parse_dates=['date'])
test_df = pd.read_csv('DailyDelhiClimateTest.csv', parse_dates=['date'])

print(train_df.columns)
# 2. Sadece sıcaklık kolonu kullanılacak (target: mean_temperature)
train_temp = train_df[['meantemp']].copy()
test_temp = test_df[['meantemp']].copy()

# 3. Normalize Et
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_temp)
test_scaled = scaler.transform(test_temp)

# 4. RNN Girişi İçin Veri Hazırlığı
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 10
X_train, y_train = create_sequences(train_scaled, window_size)
X_test, y_test = create_sequences(test_scaled, window_size)

# 5. RNN Modeli Oluştur
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# 6. Modeli Eğit
history = model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.1, verbose=1)

# 7. Test Verisi ile Tahmin
predicted = model.predict(X_test)
predicted_temp = scaler.inverse_transform(predicted)
actual_temp = scaler.inverse_transform(y_test)

mse = mean_squared_error(actual_temp, predicted_temp)
mae = mean_absolute_error(actual_temp, predicted_temp)
rmse = np.sqrt(mse)

print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f}")

# 8. Sonuçları Görselleştir
plt.figure(figsize=(12, 6))
plt.plot(actual_temp, label='Gerçek Sıcaklık')
plt.plot(predicted_temp, label='Tahmin Edilen Sıcaklık')
plt.title('Sıcaklık Tahmini (RNN)')
plt.xlabel('Gün')
plt.ylabel('Sıcaklık (°C)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
