import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import matplotlib.pyplot as plt
import numpy as np
from peft import LoraConfig, get_peft_model
import os
import argparse
import functools
from peft.utils.other import fsdp_auto_wrap_policy

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
def get_fsdp_wrap_policy(module, config=None, is_lora=True):
    """Get FSDP wrap policy for the module.

    Args:
        module: The module to get wrap policy for
        config: Configuration for wrap policy
        is_lora: Whether to enable lambda policy for LoRA modules
    """
    if config is None:
        config = {}

    # NOTE: This is a temporary workaround to be compatible with the OmegaConf & dataclass. We will remove this
    # once we have make all config in verl from OmegaConf to data class.
    def _get_attr(attr_name, default_value=None):
        if hasattr(config, "get"):
            return config.get(attr_name, default_value)
        else:
            return config.__getattribute__(attr_name)

    if _get_attr("disable", False):
        return None

    default_transformer_cls_names_to_wrap = getattr(module, "_no_split_modules", None)
    fsdp_transformer_layer_cls_to_wrap = _get_attr(
        "transformer_layer_cls_to_wrap", default_transformer_cls_names_to_wrap
    )
    min_num_params = _get_attr("min_num_params", 0)
    auto_wrap_policy = None

    policies = []

    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy

    # Add lambda policy for LoRA modules if is_lora is True
    if is_lora:
        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        policies.append(lambda_policy)

    if min_num_params > 0:
        size_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
        policies.append(size_policy)
    elif fsdp_transformer_layer_cls_to_wrap is not None:
        transformer_cls_to_wrap = set()
        for layer_class in fsdp_transformer_layer_cls_to_wrap:
            transformer_cls = get_module_class_from_name(module, layer_class)
            if transformer_cls is None:
                raise Exception("Could not find the transformer layer class to wrap in the model.")
            else:
                transformer_cls_to_wrap.add(transformer_cls)

        transformer_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=transformer_cls_to_wrap,
        )
        policies.append(transformer_policy)

    if len(policies) > 0:
        auto_wrap_policy = functools.partial(_or_policy, policies=policies)

    return auto_wrap_policy
def lambda_policy_fn(module):
    return bool(
        len(list(module.named_children())) == 0
        and getattr(module, "weight", None) is not None
        and module.weight.requires_grad
    )
def print_module_recursively(module, indent=0):
    for name, child in module.named_children():
        print("  " * indent + f"{name}: {type(child)} {lambda_policy_fn(child)}")
        print_module_recursively(child, indent + 1)


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

def setup(rank, world_size):
    """初始化分布式环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # 初始化分布式进程组
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", rank=rank, world_size=world_size)

def cleanup():
    """清理分布式环境"""
    dist.destroy_process_group()
def print_fsdp_shards(model: nn.Module):
    """
    打印一个 FSDP 封装后的模型中，哪些参数被分到一组（即被同一个 FSDP 实例管理）。

    注意：此函数是基于 FSDP 结构推断的，它会打印出 FSDP 实例及其直接管理的参数。
    """
    
    # 存储 FSDP 实例路径及其所管理的参数名称
    fsdp_groups: Dict[str, List[str]] = {}
    
    # 遍历模型的所有命名子模块
    for name, module in model.named_modules():
        # FSDP 实例通常是 FullyShardedDataParallel 的一个子类或实例
        # 实际检查取决于您的 PyTorch 版本和 FSDP 的确切导入方式。
        # 这里我们使用一个通用的方法：检查类型名是否包含 'FSDP'
        # 在真实的 PyTorch 环境中，应检查 `isinstance(module, FSDP)`
        is_fsdp = False
        try:
            # 尝试导入 FSDP 类并进行精确检查 (需要安装 torch.distributed)
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP_Class
            if isinstance(module, FSDP_Class):
                is_fsdp = True
        except ImportError:
            # 如果没有安装 distributed，则退回到基于名称的检查
            if "fsdp" in module.__class__.__name__.lower():
                is_fsdp = True

        if is_fsdp:
            # 找到一个 FSDP 实例。现在，我们需要找到它直接管理的参数。
            # 这些参数通常是该 FSDP 实例的直接子模块中未被再次 FSDP 封装的参数。
            
            # 使用 FSDP 实例的完整名称作为组的键
            group_key = name if name else "TopLevelFSDP"
            fsdp_groups[group_key] = []
            
            # 遍历当前 FSDP 实例 (module) 的直接命名参数 (immediate parameters)
            for param_name, param in module.named_parameters(recurse=False):
                # FSDP 实例直接管理的参数（通常是该模块的叶子参数）
                # 它们的名字是相对于当前 FSDP 实例 (module) 的
                full_param_name = f"{name}.{param_name}" if name else param_name
                fsdp_groups[group_key].append(full_param_name)
            
            # 打印 FSDP 组的名称
            print(f"--- FSDP Group: **{group_key}** ---")
            
            # 打印该组管理的参数列表
            if fsdp_groups[group_key]:
                print(f"  Managed Parameters ({len(fsdp_groups[group_key])}):")
                for p_name in fsdp_groups[group_key]:
                    print(f"    - {p_name}")
            else:
                print("  This FSDP instance is likely a wrapper for sub-FSDP modules (auto-wrapping).")

    if not fsdp_groups:
        print("Model does not appear to contain any FSDP instances.")
from peft.utils import get_peft_model_state_dict
def main(rank, world_size, args):
    """主训练函数"""
    print(f"Running on rank {rank} of {world_size}")
    
    # 设置分布式环境
    setup(rank, world_size)
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    print(f"Using device: {device}")
    
    # 配置SVD LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["fc1", "fc2", "fc3"],
        use_svdlora=True,
        bias="none"
    )
    print(lora_config)

    # 初始化模型和数据
    MAX_DIGITS = 10
    model = DecimalAdditionModel(input_size=2*MAX_DIGITS + 1)  # 两个数字序列 + 1个PAD = 21
    
    # 生成训练数据
    print("Generating training data...")
    train_data, train_labels = generate_addition_data(num_samples=10000, max_digits=MAX_DIGITS)
    eval_data, eval_labels = generate_addition_data(num_samples=2000, max_digits=MAX_DIGITS)

    print(f"Training data shape: {train_data.shape}")
    print(f"Evaluation data shape: {eval_data.shape}")

    # 应用SVD LoRA到模型
    peft_model = get_peft_model(model, lora_config)
    if rank == 0:
        print(peft_model)
        print_module_recursively(peft_model)
        
    # 使用FSDP包装模型
    fsdp_model = FSDP(
        peft_model,
        device_id=device,
        auto_wrap_policy=get_fsdp_wrap_policy(peft_model),
    )
    if rank == 0:
        print("\n========== FSDP Wrap 结果（普通 LoRA）==========")
        print_fsdp_shards(fsdp_model)
    
    # 设置优化器和损失函数
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=5e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    # 训练模型并记录loss
    train_losses = []
    eval_losses = []
    eval_accuracies = []
    epochs = args.epochs

    print("Starting training...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        # 训练阶段
        fsdp_model.train()
        for i in range(0, len(train_data), args.batch_size):  # 批量处理
            batch_data = train_data[i:i+args.batch_size].to(device)
            batch_labels = train_labels[i:i+args.batch_size].to(device)
            
            optimizer.zero_grad()
            outputs = fsdp_model(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        scheduler.step()
        avg_epoch_loss = epoch_loss / (len(train_data) // args.batch_size)
        train_losses.append(avg_epoch_loss)
        
        # 评估阶段
        fsdp_model.eval()
        eval_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for i in range(0, len(eval_data), args.batch_size):
                batch_data = eval_data[i:i+args.batch_size].to(device)
                batch_labels = eval_labels[i:i+args.batch_size].to(device)
                
                outputs = fsdp_model(batch_data)
                loss = criterion(outputs, batch_labels)
                eval_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        avg_eval_loss = eval_loss / (len(eval_data) // args.batch_size)
        eval_losses.append(avg_eval_loss)
        accuracy = 100 * correct / total
        eval_accuracies.append(accuracy)
        
        if rank == 0:  # 只在主进程中打印日志
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_epoch_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # 在主进程中绘制loss和accuracy曲线
    if rank == 0:
        os.makedirs("./fsdp_svdlora_addition_model", exist_ok=True)
        
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
        plt.savefig('./fsdp_svdlora_addition_model/fsdp_svdlora_addition_curves.png')
        # plt.show()
        

        # 测试几个具体的加法例子
        print("\nTesting specific addition examples:")
        
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
                print("well...")
                output = fsdp_model(test_tensor)
                predicted = torch.argmax(output, dim=1).item()
                actual_last_digit = actual_result % 10
                status = "✓" if predicted == actual_last_digit else "✗"
                print(f"{description} = ...{actual_last_digit} (actual), predicted: {predicted} {status}")

    # 清理分布式环境
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FSDP SVDLora Decimal Addition Training")
    parser.add_argument("--world_size", type=int, default=2, help="Number of GPUs/processes to use")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    
    args = parser.parse_args()
    
    # 启动多进程训练
    world_size = args.world_size
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)