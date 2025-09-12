import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """
    自注意力机制模块，用于处理单视图数据
    将输入的特征向量通过自注意力计算，得到新的特征表示
    """
    def __init__(self, input_dim, hidden_dim=None, output_dim=None):
        """
        初始化自注意力模块
        
        参数:
            input_dim: 输入特征的维度
            hidden_dim: 隐藏层维度，默认与输入维度相同
            output_dim: 输出特征的维度，默认与输入维度相同
        """
        super(SelfAttention, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # 定义查询、键、值的线性变换层
        self.query = nn.Linear(input_dim, self.hidden_dim)
        self.key = nn.Linear(input_dim, self.hidden_dim)
        self.value = nn.Linear(input_dim, self.hidden_dim)
        
        # 输出层，将注意力加权后的特征映射到指定维度空间
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
        # 缩放因子，用于稳定注意力分数
        self.scale = torch.sqrt(torch.FloatTensor([self.hidden_dim]))
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征，形状为 [batch_size, seq_len, input_dim]
        
        返回:
            output: 自注意力处理后的特征，形状为 [batch_size, seq_len, output_dim]
        """
        
        # 计算查询、键、值
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.permute(0, 2, 1)) / self.scale.to(x.device)
        
        # 应用softmax得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 使用注意力权重对值进行加权求和
        attended_values = torch.matmul(attention_weights, value)
        
        # 通过输出层映射到指定维度空间
        output = self.output_layer(attended_values)
        
        return output


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制，通过多个独立的注意力头并行计算，然后将结果拼接
    能够捕捉不同子空间的特征信息
    """
    def __init__(self, input_dim, num_heads, hidden_dim=None, output_dim=None):
        """
        初始化多头自注意力模块
        
        参数:
            input_dim: 输入特征的维度
            num_heads: 注意力头的数量
            hidden_dim: 隐藏层总维度，默认与输入维度相同
            output_dim: 输出特征的维度，默认与输入维度相同
        """
        super(MultiHeadSelfAttention, self).__init__()
        
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # 确保隐藏层维度能被头数整除
        assert self.hidden_dim % self.num_heads == 0, "隐藏层维度必须能被头数整除"
        
        # 每个头的维度
        self.head_dim = self.hidden_dim // self.num_heads
        
        # 定义查询、键、值的线性变换层
        self.query = nn.Linear(input_dim, self.hidden_dim)
        self.key = nn.Linear(input_dim, self.hidden_dim)
        self.value = nn.Linear(input_dim, self.hidden_dim)
        
        # 输出层，将特征映射到指定维度空间
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
        # 缩放因子
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x: 输入特征，形状为 [batch_size, seq_len, input_dim]
        
        返回:
            output: 多头自注意力处理后的特征，形状为 [batch_size, seq_len, output_dim]
        """
        batch_size = x.shape[0]
        
        # 计算查询、键、值
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # 重塑张量以适应多头计算
        # 形状变为 [batch_size, num_heads, seq_len, head_dim]
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.permute(0, 1, 3, 2)) / self.scale.to(x.device)
        
        # 应用softmax得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 使用注意力权重对值进行加权求和
        attended_values = torch.matmul(attention_weights, value)
        
        # 重塑回原始形状
        attended_values = attended_values.permute(0, 2, 1, 3).contiguous()
        attended_values = attended_values.view(batch_size, -1, self.hidden_dim)
        
        # 通过输出层映射到指定维度空间
        output = self.output_layer(attended_values)
        
        return output


def apply_self_attention(view_data, model, device=None, train_mode=False):
    """
    应用自注意力模型处理视图数据的函数
    
    参数:
        view_data: 视图数据，形状为 [batch_size, seq_len, feature_dim]
        model: 自注意力模型实例
        device: 计算设备，默认为None
        train_mode: 是否为训练模式，训练模式下会启用梯度计算并设置为训练状态
    
    返回:
        processed_data: 处理后的视图数据
    """
    # 如果数据不是torch张量，转换为张量
    if not isinstance(view_data, torch.Tensor):
        view_data = torch.tensor(view_data, dtype=torch.float32)
    
    # 移动到指定设备
    if device is not None:
        view_data = view_data.to(device)
        model = model.to(device)
    
    # 根据模式设置模型状态
    if train_mode:
        model.train()
        # 训练模式下启用梯度计算
        processed_data = model(view_data)
    else:
        # 评估模式
        model.eval()
        with torch.no_grad():
            processed_data = model(view_data)
    
    return processed_data