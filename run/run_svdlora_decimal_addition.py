import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from peft import LoraConfig, get_peft_model


# 定义用于十进制加法的模型
class DecimalAdditionModel(nn.Module):
    def __init__(self, input_size=21, hidden_size=256, output_size=10):  # 输出0-9（个位数）
        super().__init__()
        self.embedding = nn.Embedding(11, 32)  # 0-9数字 + 1个PAD符号(索引10)
        self.fc1 = nn.Linear(input_size * 32, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, input_size) - 包含两个数字的序列
        x = self.embedding(x)  # (batch_size, input_size, 32)
        x = x.view(x.size(0), -1)  # (batch_size, input_size * 32)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# 生成十进制加法训练数据
def generate_addition_data(num_samples=10000, max_digits=10):
    """
    生成加法数据集
    每个样本包含两个数字序列，格式为:
    [digit1_digit2_..._digitN, PAD, digit1_digit2_..._digitN]
    """
    data = []
    labels = []
    
    for _ in range(num_samples):
        # 生成两个随机数（最多max_digits位数字）
        num_digits1 = torch.randint(1, max_digits + 1, (1,)).item()
        num_digits2 = torch.randint(1, max_digits + 1, (1,)).item()
        
        # 生成指定位数的数字
        min_val1 = 10**(num_digits1-1) if num_digits1 > 1 else 0
        max_val1 = 10**num_digits1 - 1
        num1 = torch.randint(min_val1, max_val1 + 1, (1,)).item() if max_val1 >= min_val1 else 0
        
        min_val2 = 10**(num_digits2-1) if num_digits2 > 1 else 0
        max_val2 = 10**num_digits2 - 1
        num2 = torch.randint(min_val2, max_val2 + 1, (1,)).item() if max_val2 >= min_val2 else 0
        
        result = num1 + num2
        
        # 转换为数字序列
        str_num1 = str(num1).zfill(max_digits)  # 补齐到max_digits位
        str_num2 = str(num2).zfill(max_digits)  # 补齐到max_digits位
        
        # 转换为数字列表
        digits1 = [int(d) for d in str_num1]
        digits2 = [int(d) for d in str_num2]
        
        # 构造输入序列（两个数字序列用PAD分隔）
        PAD = 10  # 用10作为填充符（不在0-9范围内）
        input_seq = digits1 + [PAD] + digits2  # 总长度为2*max_digits + 1
        
        # 预测结果的个位数
        label = result % 10
        
        data.append(input_seq)
        labels.append(label)
    
    return torch.tensor(data), torch.tensor(labels)

# 检查是否有可用的GPU设备
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 配置SVD LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["fc1", "fc2", "fc3"],
    use_svdlora=True,
    bias="none"
)

# 初始化模型和数据
MAX_DIGITS = 10
model = DecimalAdditionModel(input_size=2*MAX_DIGITS + 1)  # 两个数字序列 + 1个PAD = 21
import os
os.makedirs("./svdlora_addition_model", exist_ok=True)
torch.save(model.state_dict(), "./svdlora_addition_model/decimal_addition_model.pt")

# 将模型移动到指定设备
model = model.to(device)

# 生成训练数据
print("Generating training data...")
train_data, train_labels = generate_addition_data(num_samples=10000, max_digits=MAX_DIGITS)
eval_data, eval_labels = generate_addition_data(num_samples=2000, max_digits=MAX_DIGITS)

print(f"Training data shape: {train_data.shape}")
print(f"Evaluation data shape: {eval_data.shape}")

# 应用SVD LoRA到模型
peft_model = get_peft_model(model, lora_config)
peft_model = peft_model.to(device)  # 确保PEFT模型也在正确的设备上
peft_model.train()

# 设置优化器和损失函数
optimizer = torch.optim.AdamW(peft_model.parameters(), lr=5e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
criterion = nn.CrossEntropyLoss()

# 训练模型并记录loss
train_losses = []
eval_losses = []
eval_accuracies = []
epochs = 10

print("Starting training...")
for epoch in range(epochs):
    epoch_loss = 0.0
    # 训练阶段
    peft_model.train()
    for i in range(0, len(train_data), 32):  # 批量处理
        batch_data = train_data[i:i+32].to(device)
        batch_labels = train_labels[i:i+32].to(device)
        
        optimizer.zero_grad()
        outputs = peft_model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(peft_model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    avg_epoch_loss = epoch_loss / (len(train_data) // 32)
    train_losses.append(avg_epoch_loss)
    
    # 评估阶段
    peft_model.eval()
    eval_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(eval_data), 32):
            batch_data = eval_data[i:i+32].to(device)
            batch_labels = eval_labels[i:i+32].to(device)
            
            outputs = peft_model(batch_data)
            loss = criterion(outputs, batch_labels)
            eval_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    avg_eval_loss = eval_loss / (len(eval_data) // 32)
    eval_losses.append(avg_eval_loss)
    accuracy = 100 * correct / total
    eval_accuracies.append(accuracy)
    

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.2f}%")

# 绘制loss和accuracy曲线
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, label='Train Loss', color='blue')
plt.plot(range(1, epochs+1), eval_losses, label='Eval Loss', color='red')
plt.title('Training and Evaluation Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), eval_accuracies, color='green')
plt.title('Evaluation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)

plt.tight_layout()
plt.savefig('./svdlora_addition_model/svdlora_addition_curves.png')
# plt.show()

# 保存模型
peft_model.save_pretrained("./svdlora_addition_model")
print("Model saved to ./svdlora_addition_model")

# 测试几个具体的加法例子
print("\nTesting specific addition examples:")



def test(peft_model):
    # 创建测试用例
    def create_test_case(num1_str, num2_str, max_digits=10):
        # 将字符串转换为数字列表并补齐
        digits1 = [int(d) for d in num1_str.zfill(max_digits)]
        digits2 = [int(d) for d in num2_str.zfill(max_digits)]
        PAD = 10
        input_seq = digits1 + [PAD] + digits2
        result = int(num1_str) + int(num2_str)
        return input_seq, f"{num1_str} + {num2_str}", result

    test_cases = [
        create_test_case("12345", "67890", MAX_DIGITS),
        create_test_case("9999999999", "1", MAX_DIGITS),
        create_test_case("5555555555", "4444444444", MAX_DIGITS),
        create_test_case("123", "456789", MAX_DIGITS),
        create_test_case("0", "9999999999", MAX_DIGITS),
    ]

    with torch.no_grad():
        for test_input, description, actual_result in test_cases:
            test_tensor = torch.tensor([test_input]).to(device)
            output = peft_model(test_tensor)
            predicted = torch.argmax(output, dim=1).item()
            actual_last_digit = actual_result % 10
            status = "✓" if predicted == actual_last_digit else "✗"
            print(f"{description} = ...{actual_last_digit} (actual), predicted: {predicted} {status}")

print("Test without SVD LoRA loading:")
base_model = DecimalAdditionModel(input_size=2*MAX_DIGITS + 1)
base_model.load_state_dict(torch.load("./svdlora_addition_model/decimal_addition_model.pt"))
from peft import PeftModel
peft_model = PeftModel.from_pretrained(base_model, "./svdlora_addition_model")
peft_model = peft_model.to(device)

test(peft_model)

print("Test SVD LoRA loading:")
base_model = DecimalAdditionModel(input_size=2*MAX_DIGITS + 1)
base_model.load_state_dict(torch.load("./svdlora_addition_model/decimal_addition_model.pt"))

from peft import PeftModel
peft_model = PeftModel.from_pretrained(base_model, "./svdlora_addition_model", use_svdlora=True)
peft_model = peft_model.to(device)
print(f"{peft_model.peft_config['default'].use_svdlora=}")
test(peft_model)