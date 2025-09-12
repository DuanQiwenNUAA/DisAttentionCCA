import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


def load_multi_view_data(view1_data, view2_data):
    """
    加载多视图数据
    
    参数:
        view1_data: 第一个视图的数据，可以是numpy数组或文件路径
        view2_data: 第二个视图的数据，可以是numpy数组或文件路径
    
    返回:
        tuple: (view1, view2)，两个视图的数据
    """
    # 如果是文件路径，则加载数据
    if isinstance(view1_data, str):
        view1 = np.loadtxt(view1_data, delimiter=',')
    else:
        view1 = np.array(view1_data)
        
    if isinstance(view2_data, str):
        view2 = np.loadtxt(view2_data, delimiter=',')
    else:
        view2 = np.array(view2_data)
        
    return view1, view2


def normalize_data(view_data):
    """
    对视图数据进行标准化处理
    
    参数:
        view_data: 视图数据，形状为 [n_samples, n_features]
    
    返回:
        normalized_data: 标准化后的数据
        scaler: 用于标准化的转换器，可以用于反向转换
    """
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(view_data)
    return normalized_data, scaler


def prepare_for_attention(view_data, sequence_length=None):
    """
    准备数据以适应自注意力机制的输入格式
    
    参数:
        view_data: 视图数据，形状为 [n_samples, n_features]
        sequence_length: 序列长度，如果为None则将每个样本视为一个序列
    
    返回:
        prepared_data: 处理后的数据，形状为 [n_samples, seq_len, feature_dim]
    """
    # # 确保数据是numpy数组
    # if isinstance(view_data, torch.Tensor):
    #     view_data = view_data.detach().numpy()
    # else:
    #     view_data = np.array(view_data)
    
    # # 获取数据形状
    # if len(view_data.shape) == 1:
    #     # 一维数据转换为二维
    #     view_data = view_data.reshape(-1, 1)
    # elif len(view_data.shape) > 2:
    #     # 对于高维数据，将其展平为二维
    #     view_data = view_data.reshape(view_data.shape[0], -1)
        
    # 获取样本数量和特征维度
    n_samples, n_features = view_data.shape
    
    if sequence_length is None:
        # 将每个样本视为一个长度为1的序列
        prepared_data = view_data.reshape(n_samples, 1, n_features)
    else:
        # 检查是否需要重塑数据
        if n_features % sequence_length != 0:
            raise ValueError(f"特征维度{n_features}必须能被序列长度{sequence_length}整除")
        
        # 重塑数据为序列形式
        feature_dim = n_features // sequence_length
        prepared_data = view_data.reshape(n_samples, sequence_length, feature_dim)
        
    return prepared_data


def convert_to_tensor(data, dtype=torch.float32):
    """
    将数据转换为PyTorch张量
    
    参数:
        data: 输入数据，可以是numpy数组或列表
        dtype: 张量的数据类型
    
    返回:
        tensor: 转换后的PyTorch张量
    """
    if isinstance(data, torch.Tensor):
        return data.to(dtype)
    else:
        return torch.tensor(data, dtype=dtype)


def split_train_test(view1, view2, test_ratio=0.2, random_state=None):
    """
    分割训练集和测试集
    
    参数:
        view1: 第一个视图的数据
        view2: 第二个视图的数据
        test_ratio: 测试集比例
        random_state: 随机种子
    
    返回:
        tuple: (view1_train, view1_test, view2_train, view2_test)
    """
    # 设置随机种子
    if random_state is not None:
        np.random.seed(random_state)
    
    # 获取样本数量
    n_samples = view1.shape[0]
    
    # 生成随机索引
    indices = np.random.permutation(n_samples)
    
    # 计算分割点
    split_point = int(n_samples * (1 - test_ratio))
    
    # 分割数据
    train_indices, test_indices = indices[:split_point], indices[split_point:]
    
    view1_train, view1_test = view1[train_indices], view1[test_indices]
    view2_train, view2_test = view2[train_indices], view2[test_indices]
    
    return view1_train, view1_test, view2_train, view2_test


def batch_data(view1, view2, batch_size, shuffle=True):
    """
    将数据分批次，方便模型训练
    
    参数:
        view1: 第一个视图的数据
        view2: 第二个视图的数据
        batch_size: 批次大小
        shuffle: 是否打乱数据
    
    返回:
        generator: 生成批次数据的生成器
    """
    # 获取样本数量
    n_samples = view1.shape[0]
    
    # 生成索引
    indices = np.arange(n_samples)
    
    # 打乱索引
    if shuffle:
        np.random.shuffle(indices)
    
    # 分批次生成数据
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_indices = indices[start_idx:end_idx]
        
        yield view1[batch_indices], view2[batch_indices]
