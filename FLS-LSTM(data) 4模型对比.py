import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
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


# 定义模糊层
class FuzzyLayer(nn.Module):
    def __init__(self, n_features, n_rules):
        super(FuzzyLayer, self).__init__()
        self.n_rules = n_rules
        self.weights = nn.Parameter(torch.randn(n_features, n_rules))

    def forward(self, x):
        return torch.matmul(x, self.weights)


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


# 定义FLS-LSTM模型
class FLSLSTMModel(nn.Module):
    def __init__(self, n_features, n_rules, hidden_size, output_size):
        super(FLSLSTMModel, self).__init__()
        self.fuzzy_layer = FuzzyLayer(n_features, n_rules)
        self.lstm = nn.LSTM(n_rules, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        fuzzy_out = self.fuzzy_layer(x)
        fuzzy_out = fuzzy_out.view(fuzzy_out.size(0), -1, self.fuzzy_layer.n_rules)
        _, (hn, _) = self.lstm(fuzzy_out)
        out = self.fc(hn[-1])
        return out

# 定义仅使用注意力机制和LSTM的模型
class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        weighted_out = torch.sum(attention_weights * lstm_out, dim=1)
        out = self.fc(weighted_out)
        return out


# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)  # 用于计算注意力权重

    def _generate_positional_encoding(self, d_model, seq_len):
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(device)  # 确保位置编码在同一设备上
        return pe

    def forward(self, lstm_out):
        seq_len = lstm_out.size(1)
        batch_size = lstm_out.size(0)
        positional_encoding = self._generate_positional_encoding(lstm_out.size(-1), seq_len)
        lstm_out = lstm_out + positional_encoding.repeat(batch_size, 1, 1)
        attention_weights = torch.softmax(self.linear(lstm_out), dim=1)  # 计算注意力权重
        weighted_out = torch.sum(attention_weights * lstm_out, dim=1)  # 加权求和
        return weighted_out

# 定义带注意力机制的FLS-LSTM模型
class MyModel(nn.Module):
    def __init__(self, n_features, n_rules, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.fuzzy_layer = FuzzyLayer(n_features, n_rules)  # 模糊层
        self.lstm = nn.LSTM(n_rules, hidden_size, batch_first=True, num_layers=2)  # 堆叠的LSTM层
        self.attention = AttentionLayer(hidden_size)  # 使用单独的注意力层
        self.fc = nn.Linear(hidden_size, output_size)  # 全连接层
        self.dropout = nn.Dropout(0.5)  # Dropout层

    def forward(self, x):
        fuzzy_out = self.fuzzy_layer(x)  # 通过模糊层获得模糊输出
        fuzzy_out = fuzzy_out.view(fuzzy_out.size(0), -1, self.fuzzy_layer.n_rules)  # 调整形状
        lstm_out, _ = self.lstm(fuzzy_out)  # 输入到LSTM
        weighted_out = self.attention(lstm_out)  # 使用注意力层
        out = self.fc(weighted_out + lstm_out[:, -1, :])  # 残差连接
        return out
# 超参数
hidden_size = 64
output_size = 1
num_epochs = 200
learning_rate = 0.001
n_rules = 3


# 训练模型的通用函数
def train_model(model, dataloader, optimizer, num_epochs):
    print(f"\n开始训练 {model.__class__.__name__} 模型...")
    for epoch in range(num_epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = nn.MSELoss()(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"{model.__class__.__name__} Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")


# 模型评估函数
def evaluate_model(predictions, true_values):
    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predictions)
    r2 = r2_score(true_values, predictions)
    accuracy = np.mean(np.abs(predictions - true_values) < 1)
    return mse, rmse, mae, r2, accuracy


# 创建模型、优化器并进行训练和评估
models = {
    "LSTM": LSTMModel(X.shape[2], hidden_size, output_size),
    "FLS-LSTM": FLSLSTMModel(X.shape[2], n_rules, hidden_size, output_size),
    "AttentionLSTMModel": AttentionLSTMModel(X.shape[2], hidden_size, output_size),
    "MyModel": MyModel(X.shape[2], n_rules, hidden_size, output_size)
}

metrics_results = {}
import time  # 导入时间库

# 增加用于存储每个模型运行时间的字典
runtime_results = {}

# 修改训练循环，记录每个模型的运行时间
for model_name, model in models.items():
    start_time = time.time()  # 开始计时
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_model(model, dataloader, optimizer, num_epochs)
    end_time = time.time()  # 结束计时

    # 计算运行时间并存储
    runtime_results[model_name] = end_time - start_time

    # 评估模型
    model.eval()
    with torch.no_grad():
        preds = model(X).numpy()
        preds = label_scaler.inverse_transform(preds)
        metrics_results[model_name] = evaluate_model(preds, label_scaler.inverse_transform(y.numpy()))

# 控制台打印所有模型的运行时间在一行
runtime_str = ', '.join([f"{model}: {round(runtime, 2)}s" for model, runtime in runtime_results.items()])
print("模型运行时间:", runtime_str)
# 绘制运行时间对比图
plt.figure(figsize=(8, 5))
plt.bar(runtime_results.keys(), runtime_results.values())
plt.ylabel("Runtime (seconds)")
plt.title("Model Runtime Comparison")
plt.show()

# 打印性能评估结果
for model_name, metrics in metrics_results.items():
    mse, rmse, mae, r2, accuracy = metrics
    print(f"{model_name} 模型评估结果: MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, Accuracy: {accuracy:.4f}")

# 性能指标比较柱状图
labels = ['MSE', 'RMSE', 'MAE', 'R²', 'Accuracy']
model_metrics = [metrics_results[model] for model in models.keys()]

x = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots()
colors = ['gray', 'darkgray', 'lightgray', 'black']  # 使用不同的灰色调
for i, (model_name, metrics) in enumerate(metrics_results.items()):
    rects = ax.bar(x + i * width, metrics, width, label=model_name, color=colors[i])

ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison')
ax.set_xticks(x + width / 2)
ax.set_xticklabels(labels)
ax.legend()

plt.show()


# 克拉克误差网格图
def clark_error_grid(predictions, true_values, model_name="Model"):
    plt.figure(figsize=(8, 8))
    plt.title(f"{model_name} Clark Error Grid")
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.plot([0, 20], [0, 20], 'k--')  # 45度线

    # 计算区域
    zone_a, zone_b, zone_c, zone_d, zone_e = 0, 0, 0, 0, 0
    for pred, true in zip(predictions, true_values):
        if true < 3.9:  # 低血糖
            if pred < 3.9:
                zone_a += 1
            elif 3.9 <= pred < 4.9:
                zone_b += 1
            elif 4.9 <= pred < 10:
                zone_c += 1
            else:
                zone_d += 1
        elif 3.9 <= true <= 6.1:  # 正常血糖范围
            if true - 0.2 <= pred <= true + 0.2:
                zone_a += 1
            elif pred < true - 0.2:
                zone_b += 1
            elif true + 0.2 < pred < true + 3:
                zone_c += 1
            else:
                zone_d += 1
        else:  # 高血糖
            if pred > 6.1:
                zone_a += 1
            elif 6.1 < pred <= 7.1:
                zone_b += 1
            elif 7.1 < pred <= 8.1:
                zone_c += 1
            else:
                zone_d += 1

    total = len(predictions)
    zone_a_perc = (zone_a / total) * 100
    zone_b_perc = (zone_b / total) * 100
    zone_c_perc = (zone_c / total) * 100
    zone_d_perc = (zone_d / total) * 100

    plt.scatter(predictions, true_values, label='Predictions', color='black', alpha=0.5)
    plt.xlabel("Predicted Blood Glucose (mmol/L)", color='black')
    plt.ylabel("True Blood Glucose (mmol/L)", color='black')
    plt.grid()
    plt.text(1, 18, f"Zone A: {zone_a_perc:.1f}%", fontsize=12, color='black')
    plt.text(1, 16, f"Zone B: {zone_b_perc:.1f}%", fontsize=12, color='darkgray')
    plt.text(1, 14, f"Zone C: {zone_c_perc:.1f}%", fontsize=12, color='gray')
    plt.text(1, 12, f"Zone D: {zone_d_perc:.1f}%", fontsize=12, color='lightgray')
    plt.legend()
    plt.show()


# 评估每个模型并绘制克拉克误差网格
with torch.no_grad():
    for model_name, model in models.items():
        preds = model(X).numpy()
        preds = label_scaler.inverse_transform(preds)
        clark_error_grid(preds, label_scaler.inverse_transform(y.numpy()), model_name=model_name)

# 在FLS-LSTM模型训练完成后添加以下代码
# torch.save(models["FLS-LSTM"].state_dict(), "FLS_LSTM_model.pt")