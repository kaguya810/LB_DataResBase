import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE  # 引入SMOTE进行过采样
import numpy as np
import matplotlib.pyplot as plt
import torch.onnx
from captum.attr import IntegratedGradients
from captum.attr import visualization as viz

#Github GUN3.0 —— KAGUYA810

# Load the dataset from Excel file
file_path = 'test1.xlsx'  # Update the path according to your file location
df = pd.read_excel(file_path)

def save_history_to_csv(history, filename="res1.csv"):
    # 将历史数据转换为DataFrame
    history_df = pd.DataFrame(history)
    # 保存到CSV文件
    history_df.to_csv(filename, index=False)
    print(f"历史评估数据已保存到 {filename}")

def fraction_to_decimal(fraction_str):
    try:
        numerator, denominator = fraction_str.split('/')
        return float(numerator) / float(denominator) if denominator != '0' else 0
    except:
        return 0

# Apply the fraction_to_decimal function to the 'EM' and 'EN' columns
df['EM'] = df['EM'].apply(fraction_to_decimal)
df['EN'] = df['EN'].apply(fraction_to_decimal)

# Preprocess the dataset
X = df.drop(['GROUP', 'GRADE'], axis=1)  # Assuming 'GROUP' is the target variable and 'GRADE' is not used as a feature
y = df['GROUP']
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Encode target labels

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 对B、C类数据进行重采样 # #By BY7030SWL
smote = SMOTE(sampling_strategy='auto')
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Normalize the feature data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Convert arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float)
y_train_tensor = torch.tensor(y_train_resampled, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader for both training and testing sets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

class SimulatedAnnealingLR:
    def __init__(self, initial_lr, min_lr, initial_temp, min_temp, alpha):
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.current_temp = initial_temp
        self.min_temp = min_temp
        self.alpha = alpha  # 冷却率

    def update_lr(self, epoch):
        # 计算新的学习率
        if self.current_temp > self.min_temp:
            self.current_temp *= self.alpha
            self.current_lr = max(self.min_lr, self.current_lr * self.alpha)
        return self.current_lr


# Define the neural network model with an additional layer
class EnhancedNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(EnhancedNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 225)
        self.bn1 = nn.BatchNorm1d(225)  # 添加批标准化层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(225, 128)
        self.bn2 = nn.BatchNorm1d(128)  # 添加批标准化层
        self.layer_back = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)  # 添加批标准化层
        self.layer_back2 = nn.Linear(64, 64)
        self.bn4 = nn.BatchNorm1d(64)  # 添加批标准化层
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # 应用批标准化
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)  # 应用批标准化
        x = self.relu(x)
        x = self.layer_back(x)
        x = self.bn3(x)  # 应用批标准化
        x = self.relu(x)
        x = self.layer_back2(x)
        x = self.bn4(x)  # 应用批标准化
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
model = EnhancedNN(input_size=X_train.shape[1], num_classes=len(le.classes_))
num_classes = len(le.classes_)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0096)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.001, patience=5)
sa_lr_scheduler = SimulatedAnnealingLR(initial_lr=0.01, min_lr=0.0001, initial_temp=100, min_temp=1, alpha=0.9)

# Function to plot training metrics

def plot_history(history):
    plt.figure(figsize=(12, 8))

    # 绘制训练损失和验证损失
    plt.subplot(2, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.legend()

    # 绘制准确率
    plt.subplot(2, 3, 2)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.title('Accuracy')
    plt.legend()

    # 绘制AUC
    plt.subplot(2, 3, 3)
    plt.plot(history['auc'], label='AUC')
    plt.title('AUC')
    plt.legend()

    # 绘制特异性
    plt.subplot(2, 3, 4)
    plt.plot(history['specificity'], label='Specificity')
    plt.title('Specificity')
    plt.legend()

    # 绘制召回率
    plt.subplot(2, 3, 5)
    plt.plot(history['recall'], label='Recall')
    plt.title('Recall')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Function to train the model
def compute_validation_loss(model, val_loader, criterion):
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    with torch.no_grad():  # 在计算验证损失时不需要计算梯度
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    return avg_loss


def train_model(model, criterion, optimizer, train_loader, val_loader, sa_lr_scheduler, num_epochs=30):
    # 初始化存储训练过程数据的字典
    history = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': [],
        'auc': [],
        'specificity': [],
        'recall': []
    }

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        total_loss = 0.0

        # 更新学习率
        lr = sa_lr_scheduler.update_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()  # 清空之前的梯度
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)  # 使用正确的方式更新字典中的列表

        # 在这里调用evaluate_model来计算验证集上的性能指标
        eval_metrics = evaluate_model(model, val_loader, num_classes, le, return_metrics=True)
        history['val_loss'].append(eval_metrics["val_loss"])
        history['accuracy'].append(eval_metrics["accuracy"])
        history['auc'].append(eval_metrics["auc"])
        history['specificity'].append(eval_metrics["specificity"])
        history['recall'].append(eval_metrics["recall"])

        # 打印当前epoch的信息，包括学习率
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], LR: {lr:.6f}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {eval_metrics["val_loss"]:.4f}')

    return history


def analyze_model_decisions(model, input_tensor, target=0, n_steps=50):
    """
    使用Integrated Gradients分析模型决策。

    参数:
    - model: 要分析的PyTorch模型。
    - input_tensor: 输入数据，一个PyTorch张量，包含一个或多个样本。
    - target: 分析的目标类别的索引。
    - n_steps: 计算综合梯度时的步数。

    返回:
    - 属性分数和绘制的条形图。
    """

    input_tensor.requires_grad = True

    # 初始化Integrated Gradients
    ig = IntegratedGradients(model)

    # 计算属性分数
    attributions_ig = ig.attribute(input_tensor, target=target, n_steps=n_steps)

    # 可视化
    attributions_ig_np = attributions_ig.detach().numpy().flatten()
    features = range(input_tensor.shape[1])

    plt.figure(figsize=(30, 6))
    plt.bar(features, attributions_ig_np, color='skyblue')
    plt.xlabel('Features')
    plt.ylabel('Attribution Score')
    plt.title('Feature Importances Using Integrated Gradients')
    plt.xticks(features)
    plt.show()

    return attributions_ig


# Function to export the model to ONNX format
def export_model_to_onnx(model, sample_input, filename="model.onnx"):
    torch.onnx.export(model, sample_input, filename)
    print(f"模型已保存到 {filename}")

# Function to evaluate the model
def evaluate_model(model, test_loader, num_classes, le,return_metrics=False):
    model.eval()
    y_true = []
    y_pred = []
    y_pred_proba = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(y_batch.tolist())
            y_pred.extend(predicted.tolist())
            y_pred_proba.extend(outputs.softmax(dim=1).numpy())

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True)
    conf_matrix = confusion_matrix(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovo')


    # 计算特异性
    specificity_list = []
    for i in range(num_classes):
        tn = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
        fp = conf_matrix[:, i].sum() - conf_matrix[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_list.append(specificity)
    overall_specificity = np.mean(specificity_list)

    # 计算总召回率（平均召回率）
    total_recall = np.mean([report[label]['recall'] for label in le.classes_])

    if return_metrics:
        avg_val_loss = compute_validation_loss(model, test_loader, criterion)
        return {
            "val_loss": avg_val_loss,
            "accuracy": accuracy,
            "auc": auc,
            "specificity": overall_specificity,
            "recall": total_recall
        }

    print(f'Accuracy: {accuracy:.4f}')
    print(f'AUC (One-vs-One): {auc:.4f}')
    print(f'Overall Specificity: {overall_specificity:.4f}')
    print(f'Total Recall: {total_recall:.4f}')
    print('Classification Report:')
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    print('Confusion Matrix:')
    print(conf_matrix)


# Train the model
history = train_model(model, criterion, optimizer, train_loader, test_loader, sa_lr_scheduler, num_epochs=100)
# Evaluate the model
evaluate_model(model, test_loader, num_classes, le)
# Plot training metrics
plot_history(history)
# Save the history to a CSV file
save_history_to_csv(history, filename="res1.csv")
# Export the model to ONNX format
sample_input = X_train_tensor[0].unsqueeze(0)  # 使用unsqueeze(0)来添加一个额外的维度，因为模型期望的输入是批量的
export_model_to_onnx(model, sample_input)
input_tensor = X_test_tensor[0].unsqueeze(0)
target_class = 0
attributions = analyze_model_decisions(model, input_tensor, target=target_class, n_steps=100)
