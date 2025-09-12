import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttention(nn.Module):
    """
    交叉注意力机制模块，用于处理两个不同视图的数据
    计算视图间的注意力权重并生成新的特征表示
    """
    def __init__(self, input_dim1, input_dim2, hidden_dim=None, output_dim=None):
        """
        初始化交叉注意力模块
        
        参数:
            input_dim1: 第一个视图的输入维度
            input_dim2: 第二个视图的输入维度
            hidden_dim: 隐藏层维度，默认与input_dim1相同
            output_dim: 输出特征的维度，默认与input_dim1相同
        """
        super(CrossAttention, self).__init__()
        
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim1
        self.output_dim = output_dim if output_dim is not None else input_dim1
        
        # 定义查询、键、值的线性变换层
        self.query = nn.Linear(input_dim1, self.hidden_dim)
        self.key = nn.Linear(input_dim2, self.hidden_dim)
        self.value = nn.Linear(input_dim2, self.hidden_dim)
        
        # 输出层，将注意力加权后的特征映射到指定维度空间
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)
        
        # 缩放因子，用于稳定注意力分数
        self.scale = torch.sqrt(torch.FloatTensor([self.hidden_dim]))
    
    def forward(self, x1, x2):
        """
        前向传播函数
        
        参数:
            x1: 第一个视图的特征，形状为 [batch_size, seq_len1, input_dim1]
            x2: 第二个视图的特征，形状为 [batch_size, seq_len2, input_dim2]
        
        返回:
            output: 交叉注意力处理后的特征，形状为 [batch_size, seq_len1, output_dim]
        """
        # 计算查询、键、值
        query = self.query(x1)
        key = self.key(x2)
        value = self.value(x2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.permute(0, 2, 1)) / self.scale.to(x1.device)
        
        # 应用softmax得到注意力权重
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 使用注意力权重对值进行加权求和
        attended_values = torch.matmul(attention_weights, value)
        
        # 通过输出层映射到指定维度空间
        output = self.output_layer(attended_values)
        
        return output


def apply_cross_attention(view1_data, view2_data, model, device=None, train_mode=False):
    """
    应用交叉注意力模型处理两个视图数据的函数
    
    参数:
        view1_data: 第一个视图数据，形状为 [batch_size, seq_len, feature_dim]
        view2_data: 第二个视图数据，形状为 [batch_size, seq_len, feature_dim]
        model: 交叉注意力模型实例
        device: 计算设备，默认为None
        train_mode: 是否为训练模式
    
    返回:
        processed_data: 处理后的视图数据
    """
    # 如果数据不是torch张量，转换为张量
    if not isinstance(view1_data, torch.Tensor):
        view1_data = torch.tensor(view1_data, dtype=torch.float32)
    if not isinstance(view2_data, torch.Tensor):
        view2_data = torch.tensor(view2_data, dtype=torch.float32)
    
    # 移动到指定设备
    if device is not None:
        view1_data = view1_data.to(device)
        view2_data = view2_data.to(device)
        model = model.to(device)
    
    # 根据模式设置模型状态
    if train_mode:
        model.train()
        # 训练模式下启用梯度计算
        processed_data = model(view1_data, view2_data)
    else:
        # 评估模式
        model.eval()
        with torch.no_grad():
            processed_data = model(view1_data, view2_data)
    
    return processed_data