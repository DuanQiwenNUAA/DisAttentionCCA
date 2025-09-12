import torch
import torch.optim as optim
import numpy as np
from self_attention import SelfAttention, MultiHeadSelfAttention, apply_self_attention
from cross_attention import CrossAttention, apply_cross_attention
from data_preprocessing import (
    load_multi_view_data,
    normalize_data,
    prepare_for_attention,
    convert_to_tensor,
    split_train_test,
    batch_data
)
from evaluation import evaluate_attention_effect, evaluate_kmeans_clustering
from complexity_metrics import compute_pds, compute_mnc, calculate_dataset_complexity


class AttentionCCA:
    """
    注意力机制结合CCA的主类
    先对每个视图数据进行自注意力机制处理，得到新的向量表示
    然后可以进行后续的CCA处理
    """
    def __init__(self, config=None):
        """
        初始化AttentionCCA模型
        
        参数:
            config: 配置字典，包含模型参数
        """
        # 默认配置
        self.config = {
            'view1_input_dim': 100,  # 第一个视图的输入维度
            'view2_input_dim': 100,  # 第二个视图的输入维度
            'view1_output_dim': None,  # 第一个视图的输出维度，默认为输入维度
            'view2_output_dim': None,  # 第二个视图的输出维度，默认为输入维度
            'attention_type': 'multihead',  # 'single' 或 'multihead'
            'num_heads': 4,  # 多头自注意力的头数
            'hidden_dim': 128,  # 隐藏层维度
            'use_gpu': False,  # 是否使用GPU
            'enable_cross_attention': True,  # 是否执行交叉注意力环节
        }
        
        # 更新配置
        if config is not None:
            self.config.update(config)
        
        # 初始化设备
        self.device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        
        # 初始化自注意力模型
        self._init_attention_models()
        
    def _init_attention_models(self):
        """
        初始化注意力模型
        """
        if self.config['attention_type'] == 'single':
            # 单头自注意力
            self.view1_attention = SelfAttention(
                input_dim=self.config['view1_input_dim'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['view1_output_dim']
            )
            self.view2_attention = SelfAttention(
                input_dim=self.config['view2_input_dim'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['view2_output_dim']
            )
        else:
            # 多头自注意力
            self.view1_attention = MultiHeadSelfAttention(
                input_dim=self.config['view1_input_dim'],
                num_heads=self.config['num_heads'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['view1_output_dim']
            )
            self.view2_attention = MultiHeadSelfAttention(
                input_dim=self.config['view2_input_dim'],
                num_heads=self.config['num_heads'],
                hidden_dim=self.config['hidden_dim'],
                output_dim=self.config['view2_output_dim']
            )
            
        # 初始化交叉注意力模型
        self.cross_attention1 = CrossAttention(
            input_dim1=self.config['view1_output_dim'] or self.config['view1_input_dim'],
            input_dim2=self.config['view2_output_dim'] or self.config['view2_input_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['view1_output_dim'] or self.config['view1_input_dim']
        )
        self.cross_attention2 = CrossAttention(
            input_dim1=self.config['view2_output_dim'] or self.config['view2_input_dim'],
            input_dim2=self.config['view1_output_dim'] or self.config['view1_input_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['view2_output_dim'] or self.config['view2_input_dim']
        )
            
    def process_views(self, view1_data, view2_data, sequence_length1=None, sequence_length2=None):
        """
        处理两个视图数据，应用自注意力机制和交叉注意力机制
        
        参数:
            view1_data: 第一个视图的数据
            view2_data: 第二个视图的数据
            sequence_length1: 第一个视图的序列长度
            sequence_length2: 第二个视图的序列长度
        
        返回:
            tuple: (processed_view1, processed_view2)，处理后的两个视图数据
        """
        # 准备数据格式
        prepared_view1 = prepare_for_attention(view1_data, sequence_length1)
        prepared_view2 = prepare_for_attention(view2_data, sequence_length2)
        
        # 转换为张量
        tensor_view1 = convert_to_tensor(prepared_view1)
        tensor_view2 = convert_to_tensor(prepared_view2)
        
        # 应用自注意力机制
        processed_view1 = apply_self_attention(tensor_view1, self.view1_attention, self.device)
        processed_view2 = apply_self_attention(tensor_view2, self.view2_attention, self.device)
        
        # 应用交叉注意力机制（如果启用）
        if self.config['enable_cross_attention']:
            cross_view1 = apply_cross_attention(processed_view1, processed_view2, self.cross_attention1, self.device)
            cross_view2 = apply_cross_attention(processed_view2, processed_view1, self.cross_attention2, self.device)
            
            # 将结果转换回numpy数组（如果需要）
            if not isinstance(view1_data, torch.Tensor):
                cross_view1 = cross_view1.cpu().numpy()
                cross_view2 = cross_view2.cpu().numpy()
            
            return cross_view1, cross_view2
        else:
            # 如果不启用交叉注意力，直接返回自注意力处理结果
            if not isinstance(view1_data, torch.Tensor):
                processed_view1 = processed_view1.cpu().numpy()
                processed_view2 = processed_view2.cpu().numpy()
            
            return processed_view1, processed_view2

    def save_models(self, view1_path, view2_path):
        """
        保存自注意力模型
        
        参数:
            view1_path: 第一个视图的模型保存路径
            view2_path: 第二个视图的模型保存路径
        """
        torch.save(self.view1_attention.state_dict(), view1_path)
        torch.save(self.view2_attention.state_dict(), view2_path)
        
    def load_models(self, view1_path, view2_path):
        """
        加载自注意力模型
        
        参数:
            view1_path: 第一个视图的模型加载路径
            view2_path: 第二个视图的模型加载路径
        """
        self.view1_attention.load_state_dict(torch.load(view1_path, map_location=self.device))
        self.view2_attention.load_state_dict(torch.load(view2_path, map_location=self.device))
        
        # 设置为评估模式
        self.view1_attention.eval()
        self.view2_attention.eval()
        
    def _correlation_loss(self, view1_features, view2_features):
        """
        计算两个视图特征之间的相关性损失
        目标是最大化或最小化两个视图之间的相关性
        
        参数:
            view1_features: 第一个视图处理后的特征
            view2_features: 第二个视图处理后的特征
        
        返回:
            loss: 相关性损失值
        """
        # 对特征进行平均池化，减少序列维度
        view1_mean = torch.mean(view1_features, dim=1)  # [batch_size, output_dim]
        view2_mean = torch.mean(view2_features, dim=1)  # [batch_size, output_dim]
        
        # 计算协方差矩阵
        batch_size = view1_mean.size(0)
        centered_view1 = view1_mean - torch.mean(view1_mean, dim=0, keepdim=True)
        centered_view2 = view2_mean - torch.mean(view2_mean, dim=0, keepdim=True)
        
        # 归一化特征以计算相关性
        view1_norm = torch.norm(centered_view1, dim=1, keepdim=True) + 1e-8
        view2_norm = torch.norm(centered_view2, dim=1, keepdim=True) + 1e-8
        
        normalized_view1 = centered_view1 / view1_norm
        normalized_view2 = centered_view2 / view2_norm
        
        # 计算视图间的相关性
        correlation = torch.mean(torch.sum(normalized_view1 * normalized_view2, dim=1))
        
        # 如果我们想最大化相关性，使用1 - correlation作为损失
        # 如果我们想最小化相关性，直接使用correlation作为损失
        # 这里我们选择最大化视图间的相关性
        loss = 1 - correlation
        
        return loss
        
    def train_model(self, train_data, num_epochs=100, batch_size=32, learning_rate=0.001, train_phase='self_attention'):
        """
        训练AttentionCCA模型
        
        参数:
            train_data: 训练数据，包含(view1_data, view2_data)元组
            num_epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            train_phase: 训练阶段，'self_attention'或'cross_attention'
        
        返回:
            loss_history: 训练过程中的损失历史
            processed_view1: 视图1处理后的特征
            processed_view2: 视图2处理后的特征
        """
        # 解包训练数据
        view1_train, view2_train = train_data
        
        # 准备数据格式
        view1_data = prepare_for_attention(view1_train)
        view2_data = prepare_for_attention(view2_train)
        
        # 转换为张量
        tensor_view1 = convert_to_tensor(view1_data)
        tensor_view2 = convert_to_tensor(view2_data)
        
        # 创建批次数据并转换为列表以便获取长度
        train_batches = list(batch_data(tensor_view1, tensor_view2, batch_size))
        
        # 根据训练阶段设置优化器参数
        if train_phase == 'self_attention':
            params = list(self.view1_attention.parameters()) + list(self.view2_attention.parameters())
        elif train_phase == 'cross_attention' and self.config['enable_cross_attention']:
            params = list(self.cross_attention1.parameters()) + list(self.cross_attention2.parameters())
        else:
            raise ValueError("Invalid train_phase or cross attention not enabled")
            
        optimizer = optim.Adam(params, lr=learning_rate)
        
        # 记录损失历史
        loss_history = []
        
        # 开始训练循环
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch in train_batches:
                # 解包批次数据
                batch_view1, batch_view2 = batch
                
                # 移动到指定设备
                batch_view1 = batch_view1.to(self.device)
                batch_view2 = batch_view2.to(self.device)
                
                # 前向传播 - 使用训练模式
                if train_phase == 'self_attention':
                    processed_view1 = apply_self_attention(batch_view1, self.view1_attention, self.device, train_mode=True)
                    processed_view2 = apply_self_attention(batch_view2, self.view2_attention, self.device, train_mode=True)
                elif train_phase == 'cross_attention':
                    # 应用交叉注意力
                    processed_view1 = apply_cross_attention(batch_view1, batch_view2, self.cross_attention1, self.device, train_mode=True)
                    processed_view2 = apply_cross_attention(batch_view2, batch_view1, self.cross_attention2, self.device, train_mode=True)
                
                # 计算损失
                loss = self._correlation_loss(processed_view1, processed_view2)
                
                # 反向传播和参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 累计损失
                epoch_loss += loss.item()
            
            # 计算平均损失
            avg_epoch_loss = epoch_loss / len(train_batches)
            loss_history.append(avg_epoch_loss)
            
            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                if train_phase == 'self_attention':
                    print(f"[自注意力] Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
                elif train_phase == 'cross_attention':
                    print(f"[交叉注意力] Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
        
        return loss_history, processed_view1, processed_view2

# 示例用法函数
def demo_attention_cca():
    """
    演示AttentionCCA的使用方法，包括模型训练过程
    """
    # 创建模拟数据
    np.random.seed(42)
    view1_data = np.random.rand(100, 100)  # 100个样本，每个样本100维
    view2_data = np.random.rand(100, 100)  # 100个样本，每个样本100维
    
    # 创建配置
    config = {
        'view1_input_dim': 100,
        'view2_input_dim': 100,
        'view1_output_dim': 50,  # 指定降维后的输出维度
        'view2_output_dim': 50,  # 指定降维后的输出维度
        'attention_type': 'multihead',
        'num_heads': 4,
        'hidden_dim': 128,
        'use_gpu': False
    }
    
    # 初始化模型
    model = AttentionCCA(config)
    
    # 使用未训练的模型处理数据
    print("===== 未训练模型的处理结果 =====")
    untrained_view1, untrained_view2 = model.process_views(view1_data, view2_data)
    print(f"\n测试数据形状:")
    print(f"  视图1形状: {view1_data.shape}")
    print(f"  视图2形状: {view2_data.shape}")
    print(f"训练后处理结果形状:")
    print(f"  视图1形状: {untrained_view1.shape}")
    print(f"  视图2形状: {untrained_view2.shape}")

    # 评估处理前后视图之间的相关性
    print("\n评估处理前后视图之间的相关性:")
    print("=========================")
    
    # 先确保数据是numpy数组
    if isinstance(view1_data, torch.Tensor):
        view1_data = view1_data.cpu().numpy()
    if isinstance(view2_data, torch.Tensor):
        view2_data = view2_data.cpu().numpy()
    if isinstance(untrained_view1, torch.Tensor):
        untrained_view1 = untrained_view1.cpu().numpy()
    if isinstance(untrained_view2, torch.Tensor):
        untrained_view2 = untrained_view2.cpu().numpy()
    
    # 评估原始视图间相关性
    original_cross_correlation_result = evaluate_attention_effect(view1_data, view2_data)
    original_cross_correlation = original_cross_correlation_result['cross_correlation']

    # 评估处理后视图间相关性
    processed_cross_correlation_result = evaluate_attention_effect(untrained_view1, untrained_view2)
    processed_cross_correlation = processed_cross_correlation_result['cross_correlation']

    print("\n原始视图间相关性:")
    print(f"  视图1形状: {original_cross_correlation_result['view1_shape']}")
    print(f"  视图2形状: {original_cross_correlation_result['view2_shape']}")
    print(f"  视图1均值: {original_cross_correlation_result['view1_mean']:.4f}")
    print(f"  视图2均值: {original_cross_correlation_result['view2_mean']:.4f}")
    print(f"  视图1方差: {original_cross_correlation_result['view1_variance']:.4f}")
    print(f"  视图2方差: {original_cross_correlation_result['view2_variance']:.4f}")
    print(f"  视图间相关性: {original_cross_correlation:.4f}")
    
    print("\n处理后视图间相关性:")
    print(f"  视图1形状: {processed_cross_correlation_result['view1_shape']}")
    print(f"  视图2形状: {processed_cross_correlation_result['view2_shape']}")
    print(f"  视图1均值: {processed_cross_correlation_result['view1_mean']:.4f}")
    print(f"  视图2均值: {processed_cross_correlation_result['view2_mean']:.4f}")
    print(f"  视图1方差: {processed_cross_correlation_result['view1_variance']:.4f}")
    print(f"  视图2方差: {processed_cross_correlation_result['view2_variance']:.4f}")
    print(f"  视图间相关性: {processed_cross_correlation:.4f}")
    
    print("\n相关性比较:")
    print(f"  相关性变化: {processed_cross_correlation - original_cross_correlation:.4f} ({(processed_cross_correlation - original_cross_correlation) / original_cross_correlation * 100:.2f}%)")
    
    # 准备训练数据
    print("\n===== 开始训练模型 =====")
    # 分割训练和测试数据
    view1_train, view1_test, view2_train, view2_test = split_train_test(view1_data, view2_data, test_ratio=0.2)
    train_data = (view1_train, view2_train)
    test_data = (view1_test, view2_test)
    
    # 训练自注意力模型
    print("===== 训练自注意力模型 =====")
    self_loss_history, processed_view1, processed_view2 = model.train_model(
        train_data=train_data,
        num_epochs=50,  # 训练轮数
        batch_size=view1_train.shape[0],  # 批次大小
        learning_rate=0.001,  # 学习率
        train_phase='self_attention'
    )
    
    # 保存训练后的自注意力模型
    model.save_models('view1_attention_model.pth', 'view2_attention_model.pth')

    # 计算自注意力模型输出数据的结构复杂度
    print("\n===== 计算自注意力模型输出数据的结构复杂度 =====")
    #计算PDS分数
    pds_view1 = compute_pds(torch.squeeze(processed_view1,dim = 1).detach().numpy())
    pds_view2 = compute_pds(torch.squeeze(processed_view2,dim = 1).detach().numpy())
    print(f"  视图1 PDS分数: {pds_view1:.4f}")
    print(f"  视图2 PDS分数: {pds_view2:.4f}")
    
    # 计算MNC分数
    mnc_view1 = compute_mnc(torch.squeeze(processed_view1,dim = 1).detach().numpy())
    mnc_view2 = compute_mnc(torch.squeeze(processed_view2,dim = 1).detach().numpy())
    print(f"  视图1 MNC分数: {mnc_view1:.4f}")
    print(f"  视图2 MNC分数: {mnc_view2:.4f}")
    
    # 训练交叉注意力模型
    print("\n===== 训练交叉注意力模型 =====")
    model.config['enable_cross_attention'] = True

    # 使用自注意力模型的输出，使用加权后的输出作为交叉注意力的输入
    train_data = (pds_view1 * torch.squeeze(processed_view1,dim = 1).detach().numpy(), pds_view2 * torch.squeeze(processed_view2,dim = 1).detach().numpy())
    cross_loss_history, processed_view1, processed_view2 = model.train_model(
        train_data=train_data,
        num_epochs=50,  # 训练轮数
        batch_size=view1_train.shape[0],  # 批次大小
        learning_rate=0.001,  # 学习率
        train_phase='cross_attention'
    )

    # 保存训练后的交叉注意力模型
    model.save_models('view1_attention_model.pth', 'view2_attention_model.pth')
    print("\n模型已保存到view1_attention_model.pth和view2_attention_model.pth")
    
    # 使用训练后的模型处理数据
    print("\n===== 训练后模型的处理结果 =====")
    trained_view1, trained_view2 = model.process_views(view1_test, view2_test)
    
    # 打印结果形状
    print(f"\n测试数据形状:")
    print(f"  视图1形状: {view1_test.shape}")
    print(f"  视图2形状: {view2_test.shape}")
    print(f"训练后处理结果形状:")
    print(f"  视图1形状: {trained_view1.shape}")
    print(f"  视图2形状: {trained_view2.shape}")
    
    # 评估处理前后视图之间的相关性
    print("\n评估处理前后视图之间的相关性:")
    print("=========================")
    
    # 先确保数据是numpy数组
    if isinstance(view1_test, torch.Tensor):
        view1_test = view1_test.cpu().numpy()
    if isinstance(view2_test, torch.Tensor):
        view2_test = view2_test.cpu().numpy()
    if isinstance(trained_view1, torch.Tensor):
        trained_view1 = trained_view1.cpu().numpy()
    if isinstance(trained_view2, torch.Tensor):
        trained_view2 = trained_view2.cpu().numpy()
    
    # 评估原始视图间相关性
    original_cross_correlation_result = evaluate_attention_effect(view1_test, view2_test)
    original_cross_correlation = original_cross_correlation_result['cross_correlation']
    
    # 评估原始视图的Kmeans聚类效果
    print("\n原始视图的Kmeans聚类效果:")
    original_kmeans_result = evaluate_kmeans_clustering(view1_test, view2_test)
    print(f"  视图1轮廓系数: {original_kmeans_result['view1_silhouette']:.4f}")
    print(f"  视图2轮廓系数: {original_kmeans_result['view2_silhouette']:.4f}")
    print(f"  联合轮廓系数: {original_kmeans_result['joint_silhouette']:.4f}")
    
    # 评估处理后视图间相关性
    processed_cross_correlation_result = evaluate_attention_effect(trained_view1, trained_view2)
    processed_cross_correlation = processed_cross_correlation_result['cross_correlation']
    
    # 评估处理后视图的Kmeans聚类效果
    print("\n处理后视图的Kmeans聚类效果:")
    processed_kmeans_result = evaluate_kmeans_clustering(trained_view1, trained_view2)
    print(f"  视图1轮廓系数: {processed_kmeans_result['view1_silhouette']:.4f}")
    print(f"  视图2轮廓系数: {processed_kmeans_result['view2_silhouette']:.4f}")
    print(f"  联合轮廓系数: {processed_kmeans_result['joint_silhouette']:.4f}")
    
    print("\n原始视图间相关性:")
    print(f"  视图1形状: {original_cross_correlation_result['view1_shape']}")
    print(f"  视图2形状: {original_cross_correlation_result['view2_shape']}")
    print(f"  视图1均值: {original_cross_correlation_result['view1_mean']:.4f}")
    print(f"  视图2均值: {original_cross_correlation_result['view2_mean']:.4f}")
    print(f"  视图1方差: {original_cross_correlation_result['view1_variance']:.4f}")
    print(f"  视图2方差: {original_cross_correlation_result['view2_variance']:.4f}")
    print(f"  视图间相关性: {original_cross_correlation:.4f}")
    
    print("\n处理后视图间相关性:")
    print(f"  视图1形状: {processed_cross_correlation_result['view1_shape']}")
    print(f"  视图2形状: {processed_cross_correlation_result['view2_shape']}")
    print(f"  视图1均值: {processed_cross_correlation_result['view1_mean']:.4f}")
    print(f"  视图2均值: {processed_cross_correlation_result['view2_mean']:.4f}")
    print(f"  视图1方差: {processed_cross_correlation_result['view1_variance']:.4f}")
    print(f"  视图2方差: {processed_cross_correlation_result['view2_variance']:.4f}")
    print(f"  视图间相关性: {processed_cross_correlation:.4f}")
    
    print("\n相关性比较:")
    print(f"  相关性变化: {processed_cross_correlation - original_cross_correlation:.4f} ({(processed_cross_correlation - original_cross_correlation) / original_cross_correlation * 100:.2f}%)")
    
    return trained_view1, trained_view2


if __name__ == "__main__":
    # 运行演示
    demo_attention_cca()