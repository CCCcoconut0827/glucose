import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("data.csv")
features = data[['SystolicBP', 'HeartRate', 'BodyTemperature', 'Absorbance']].values
labels = data['BloodGlucose'].values

# 数据标准化
scaler = MinMaxScaler()
features = scaler.fit_transform(features)
labels = labels.reshape(-1, 1)
label_scaler = MinMaxScaler()
labels = label_scaler.fit_transform(labels)

# 转换为时间序列数据
def create_sequences(data, labels, seq_length=1):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(labels[i + seq_length - 1])
    return np.array(X), np.array(y)

seq_length = 3  # 时间序列长度
X, y = create_sequences(features, labels, seq_length)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 创建数据集和数据加载器
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 定义BP模型
class BPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BPModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# 超参数
hidden_size = 64
output_size = 1
num_epochs = 200
learning_rate = 0.001

# 训练BP模型
bp_model = BPModel(X.shape[2] * seq_length, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(bp_model.parameters(), lr=learning_rate)

# 将数据扁平化用于BP模型
X_bp = X.view(X.size(0), -1)
for epoch in range(num_epochs):
    outputs = bp_model(X_bp)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# BP模型评估
bp_model.eval()
with torch.no_grad():
    bp_preds = bp_model(X_bp).numpy()
    bp_preds = label_scaler.inverse_transform(bp_preds)
    true_values = label_scaler.inverse_transform(y.numpy())

# 训练LSTM模型
lstm_model = LSTMModel(X.shape[2], hidden_size, output_size)
optimizer = torch.optim.Adam(lstm_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        outputs = lstm_model(batch_x)
        loss = criterion(outputs, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# LSTM模型评估
lstm_model.eval()
with torch.no_grad():
    lstm_preds = lstm_model(X).numpy()
    lstm_preds = label_scaler.inverse_transform(lstm_preds)

# SVM模型
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# 超参数调整
param_grid = {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.5]}
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X_train, y_train.ravel())
svm_model = grid_search.best_estimator_

# SVM模型评估
svm_preds = svm_model.predict(X_test)
svm_preds = label_scaler.inverse_transform(svm_preds.reshape(-1, 1))

# 性能指标计算函数
def evaluate_model(predictions, true_values):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    accuracy = np.mean(np.abs(predictions - true_values) < 1)
    return mse, rmse, mae, r2, accuracy

# BP模型性能评估
bp_mse, bp_rmse, bp_mae, bp_r2, bp_accuracy = evaluate_model(bp_preds, true_values)

# LSTM模型性能评估
lstm_mse, lstm_rmse, lstm_mae, lstm_r2, lstm_accuracy = evaluate_model(lstm_preds, true_values)

# SVM模型性能评估
svm_mse, svm_rmse, svm_mae, svm_r2, svm_accuracy = evaluate_model(svm_preds, label_scaler.inverse_transform(y_test))

# 打印性能评估结果
print("BP模型评估结果:")
print(f"BP MSE: {bp_mse:.4f}")
print(f"BP RMSE: {bp_rmse:.4f}")
print(f"BP MAE: {bp_mae:.4f}")
print(f"BP R²: {bp_r2:.4f}")
print(f"BP Accuracy: {bp_accuracy:.4f}")

print("\nLSTM模型评估结果:")
print(f"LSTM MSE: {lstm_mse:.4f}")
print(f"LSTM RMSE: {lstm_rmse:.4f}")
print(f"LSTM MAE: {lstm_mae:.4f}")
print(f"LSTM R²: {lstm_r2:.4f}")
print(f"LSTM Accuracy: {lstm_accuracy:.4f}")

print("\nSVM模型评估结果:")
print(f"SVM MSE: {svm_mse:.4f}")
print(f"SVM RMSE: {svm_rmse:.4f}")
print(f"SVM MAE: {svm_mae:.4f}")
print(f"SVM R²: {svm_r2:.4f}")
print(f"SVM Accuracy: {svm_accuracy:.4f}")

# 性能指标比较柱状图
labels = ['MSE', 'RMSE', 'MAE', 'R²', 'Accuracy']
bp_metrics = [bp_mse, bp_rmse, bp_mae, bp_r2, bp_accuracy]
lstm_metrics = [lstm_mse, lstm_rmse, lstm_mae, lstm_r2, lstm_accuracy]
svm_metrics = [svm_mse, svm_rmse, svm_mae, svm_r2, svm_accuracy]

x = np.arange(len(labels))
width = 0.25  # 柱状图宽度

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, bp_metrics, width, label='BP', color='gray')  # 设置为黑色
rects2 = ax.bar(x, lstm_metrics, width, label='LSTM', color='black')  # 设置为灰色
rects3 = ax.bar(x + width, svm_metrics, width, label='SVM', color='darkgray')  # 设置为深灰色

ax.set_xlabel('Comparison')
ax.set_title('BP, LSTM and SVM Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

plt.show()
