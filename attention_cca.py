import torch
import torch.nn as nn
import torch.nn.functional as F
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
import scipy.io as sio


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
            'num_classes': 40  # 分类类别数，设置为40以匹配1-40的标签范围
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
            output_dim=self.config['view1_output_dim'] or self.config['view1_input_dim'],
        )
        self.cross_attention2 = CrossAttention(
            input_dim1=self.config['view2_output_dim'] or self.config['view2_input_dim'],
            input_dim2=self.config['view1_output_dim'] or self.config['view1_input_dim'],
            hidden_dim=self.config['hidden_dim'],
            output_dim=self.config['view2_output_dim'] or self.config['view2_input_dim'],
        )
        
        # 初始化分类器
        combined_dim = (self.config['view1_output_dim'] or self.config['view1_input_dim']) + \
                      (self.config['view2_output_dim'] or self.config['view2_input_dim'])
        self.classifier = nn.Linear(combined_dim, self.config.get('num_classes', 40))
            
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
        
        # 设置模型为评估模式
        self.view1_attention.eval()
        self.view2_attention.eval()
        # 应用自注意力机制
        with torch.no_grad():
            processed_view1 = apply_self_attention(tensor_view1, self.view1_attention, self.device)
            processed_view2 = apply_self_attention(tensor_view2, self.view2_attention, self.device)

        # 计算自注意力模型输出数据的结构复杂度
        #计算PDS分数
        pds_view1 = compute_pds(torch.squeeze(processed_view1,dim = 1).detach().cpu().numpy())
        pds_view2 = compute_pds(torch.squeeze(processed_view2,dim = 1).detach().cpu().numpy())
        
        # 应用交叉注意力机制（如果启用）
        if self.config['enable_cross_attention']:
            self.cross_attention1.eval()
            self.cross_attention2.eval()
            with torch.no_grad():
                # cross_view1 = apply_cross_attention(abs(pds_view1) * processed_view1, abs(pds_view2) * processed_view2, self.cross_attention1, self.device)
                # cross_view2 = apply_cross_attention(abs(pds_view2) * processed_view2, abs(pds_view1) * processed_view1, self.cross_attention2, self.device)
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

    def save_models(self, view1_path, view2_path, cross1_path=None, cross2_path=None, classifier_path=None):
        """
        保存模型
        
        参数:
            view1_path: 第一个视图的自注意力模型保存路径
            view2_path: 第二个视图的自注意力模型保存路径
            cross1_path: 第一个视图的交叉注意力模型保存路径(可选)
            cross2_path: 第二个视图的交叉注意力模型保存路径(可选)
            classifier_path: 分类器模型保存路径(可选)
        """
        torch.save(self.view1_attention.state_dict(), view1_path)
        torch.save(self.view2_attention.state_dict(), view2_path)
        
        if cross1_path and cross2_path and hasattr(self, 'cross_attention1') and hasattr(self, 'cross_attention2'):
            torch.save(self.cross_attention1.state_dict(), cross1_path)
            torch.save(self.cross_attention2.state_dict(), cross2_path)
            
        if classifier_path and hasattr(self, 'classifier'):
            torch.save(self.classifier.state_dict(), classifier_path)
        
    def load_models(self, view1_path, view2_path, cross1_path=None, cross2_path=None, classifier_path=None):
        """
        加载模型
        
        参数:
            view1_path: 第一个视图的自注意力模型加载路径
            view2_path: 第二个视图的自注意力模型加载路径
            cross1_path: 第一个视图的交叉注意力模型加载路径(可选)
            cross2_path: 第二个视图的交叉注意力模型加载路径(可选)
            classifier_path: 分类器模型加载路径(可选)
        """
        self.view1_attention.load_state_dict(torch.load(view1_path, map_location=self.device))
        self.view2_attention.load_state_dict(torch.load(view2_path, map_location=self.device))
        
        # 加载交叉注意力模型(如果提供路径)
        if cross1_path and cross2_path and hasattr(self, 'cross_attention1') and hasattr(self, 'cross_attention2'):
            self.cross_attention1.load_state_dict(torch.load(cross1_path, map_location=self.device))
            self.cross_attention2.load_state_dict(torch.load(cross2_path, map_location=self.device))
            
        # 加载分类器模型(如果提供路径)
        if classifier_path and hasattr(self, 'classifier'):
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        
        # 设置为评估模式
        self.view1_attention.eval()
        self.view2_attention.eval()
        if hasattr(self, 'cross_attention1') and hasattr(self, 'cross_attention2'):
            self.cross_attention1.eval()
            self.cross_attention2.eval()
        if hasattr(self, 'classifier'):
            self.classifier.eval()
        
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

    def cca_loss(self, H1, H2):
        """

        It is the loss function of CCA as introduced in the original paper. There can be other formulations.

        """

        r1 = 1e-3
        r2 = 1e-3
        eps = 1e-9

        H1, H2 = H1.squeeze().t(), H2.squeeze().t()

        o1 = o2 = H1.size(0)

        m = H1.size(1)

        H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1)
        H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1)

        SigmaHat12 = (1.0 / (m - 1)) * torch.matmul(H1bar, H2bar.t())
        SigmaHat11 = (1.0 / (m - 1)) * torch.matmul(H1bar,
                                                    H1bar.t()) + r1 * torch.eye(o1, device=self.device)
        SigmaHat11 = SigmaHat11 + 1e-8 * torch.randn_like(SigmaHat11)
        SigmaHat22 = (1.0 / (m - 1)) * torch.matmul(H2bar,
                                                    H2bar.t()) + r2 * torch.eye(o2, device=self.device)
        SigmaHat22 = SigmaHat22 + 1e-8 * torch.randn_like(SigmaHat22)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        UPLO = "U"

        # [D1, V1] = torch.linalg.eigh(SigmaHat11, UPLO=UPLO)
        [D2, V2] = torch.linalg.eigh(SigmaHat22, UPLO=UPLO)
        [D1, V1] = torch.linalg.eigh(SigmaHat11, UPLO=UPLO)
        
        # Added to increase stability
        posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
        D1 = D1[posInd1]
        V1 = V1[:, posInd1]
        posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
        D2 = D2[posInd2]
        V2 = V2[:, posInd2]
        
        # Ensure eigenvalues are positive and not too small
        D1 = torch.clamp(D1, min=eps)
        D2 = torch.clamp(D2, min=eps)     

        SigmaHat11RootInv = torch.matmul(
            torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
        SigmaHat22RootInv = torch.matmul(
            torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())

        Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,
                                         SigmaHat12), SigmaHat22RootInv)

        # if self.use_all_singular_values:
            # all singular values are used to calculate the correlation
        tmp = torch.matmul(Tval.t(), Tval)
        corr = torch.trace(torch.sqrt(tmp))
            # assert torch.isnan(corr).item() == 0
        # else:
        #     # just the top self.outdim_size singular values are used
        #     trace_TT = torch.matmul(Tval.t(), Tval)
        #     trace_TT = torch.add(trace_TT, (torch.eye(trace_TT.shape[0])*r1).to(self.device)) # regularization for more stability
        #     U, V = torch.linalg.eigh(trace_TT, UPLO=UPLO)
        #     U = torch.where(U>eps, U, (torch.ones(U.shape).double()*eps).to(self.device))
        #     U = U.topk(self.outdim_size)[0]
        #     corr = torch.sum(torch.sqrt(U))
        return -corr
        
    def train_model(self, train_data, test_data, labels_test=None, labels=None, num_epochs=100, batch_size=32, learning_rate=0.01, train_phase='self_attention'):
        """
        训练AttentionCCA模型
        
        参数:
            train_data: 训练数据，包含(view1_data, view2_data)元组
            labels: 标签数据，形状为[batch_size]
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
        
        # 如果有标签数据，转换为张量
        if labels is not None:
            labels = torch.tensor(labels, dtype=torch.long).to(self.device)
        
        # 准备数据格式
        view1_data = prepare_for_attention(view1_train)
        view2_data = prepare_for_attention(view2_train)
        
        # 转换为张量
        tensor_view1 = convert_to_tensor(view1_data)
        tensor_view2 = convert_to_tensor(view2_data)
        
        # 创建批次数据并转换为列表以便获取长度
        if labels is not None:
            # 当有标签时，batch_data返回的是(view1_batch, view2_batch, label_batch)
            train_batches = list(batch_data(tensor_view1, tensor_view2, labels, batch_size))
        else:
            # 当无标签时，batch_data返回的是(view1_batch, view2_batch)
            train_batches = list(batch_data(tensor_view1, tensor_view2, None, batch_size))
        
        # 根据训练阶段设置优化器参数
        if train_phase == 'self_attention':
            params = list(self.view1_attention.parameters()) + list(self.view2_attention.parameters())
        elif train_phase == 'cross_attention' and self.config['enable_cross_attention']:
            params = list(self.cross_attention1.parameters()) + list(self.cross_attention2.parameters()) + \
                     list(self.classifier.parameters())
        else:
            raise ValueError("Invalid train_phase or cross attention not enabled")
            
        optimizer = optim.Adam(params, lr=learning_rate)
        
        # 记录损失历史
        loss_history = []
        
        # 设置模型为训练模式
        if train_phase == 'self_attention':
            self.view1_attention.train()
            self.view2_attention.train()
        elif train_phase == 'cross_attention' and self.config['enable_cross_attention']:
            self.cross_attention1.train()
            self.cross_attention2.train()
            self.classifier.train()
            
        # 开始训练循环
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            for batch in train_batches:
                # 解包批次数据
                if len(batch) == 2:
                    batch_view1, batch_view2 = batch
                else:
                    batch_view1, batch_view2, batch_labels = batch
                
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
                if train_phase == 'self_attention':
                    loss = self._correlation_loss(processed_view1, processed_view2)
                elif train_phase == 'cross_attention' and labels is not None:
                    # 拼接两个视图的特征并确保在正确设备上
                    combined_features = torch.cat([processed_view1.mean(dim=1), processed_view2.mean(dim=1)], dim=1).to(self.device)
                    # 确保标签也在相同设备上
                    batch_labels = batch_labels.to(self.device)
                    # 确保分类器也在相同设备上
                    self.classifier = self.classifier.to(self.device)
                    # 计算分类损失
                    classification_loss = F.cross_entropy(self.classifier(combined_features), batch_labels - 1)  # 将1-40标签转换为0-39范围
                    # 组合相关性损失和分类损失，使用加权和
                    # loss = 0.1 * self._correlation_loss(processed_view1, processed_view2) + 0.9 * classification_loss
                    loss = classification_loss

                    predictions = self.classifier(combined_features)
                    # 得到预测标签
                    predicted_labels = torch.argmax(predictions, dim=1)
                    # 计算准确率
                    accuracy = torch.mean((predicted_labels == batch_labels - 1).float())
                    print(f"  训练集准确率: {accuracy:.4f}")

                    # 每10轮评估一次测试集性能
                    if hasattr(self, 'classifier') and test_data is not None:
                        view1_test, view2_test = test_data
                        test_view1 = prepare_for_attention(view1_test)
                        test_view2 = prepare_for_attention(view2_test)
                        tensor_view1 = convert_to_tensor(test_view1)
                        tensor_view2 = convert_to_tensor(test_view2)
                        
                        with torch.no_grad():
                            processed_view1 = apply_self_attention(tensor_view1, self.view1_attention, self.device)
                            processed_view2 = apply_self_attention(tensor_view2, self.view2_attention, self.device)
                            
                            # 应用交叉注意力
                            cross_view1 = apply_cross_attention(processed_view1, processed_view2, self.cross_attention1, self.device)
                            cross_view2 = apply_cross_attention(processed_view2, processed_view1, self.cross_attention2, self.device)
                            
                            # 拼接特征并分类
                            combined_features = torch.cat([cross_view1.mean(dim=1), cross_view2.mean(dim=1)], dim=1)
                            predictions = self.classifier(combined_features)
                            predicted_labels = torch.argmax(predictions, dim=1)
                            
                            # 计算测试集准确率
                            if labels_test is not None:
                                test_labels = torch.tensor(labels_test, dtype=torch.long).to(self.device)
                                accuracy = torch.mean((predicted_labels == test_labels - 1).float())
                                print(f"  测试集准确率: {accuracy:.4f}")

                else:
                    loss = self._correlation_loss(processed_view1, processed_view2)
                
                # 反向传播和参数更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 累计损失
                epoch_loss += loss.item()
            
            # 计算平均损失
            if len(train_batches) == 0:
                print("Warning: No batches available for training, skipping this epoch")
                continue
            avg_epoch_loss = epoch_loss / len(train_batches)
            loss_history.append(avg_epoch_loss)
            
            # 打印训练进度
            if (epoch + 1) % 10 == 0:
                if train_phase == 'self_attention':
                    print(f"[自注意力] Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
                elif train_phase == 'cross_attention':
                    print(f"[交叉注意力] Epoch {epoch+1}/{num_epochs}, Loss: {avg_epoch_loss:.6f}")
                    
                    # # 每10轮评估一次测试集性能
                    # if hasattr(self, 'classifier') and test_data is not None:
                    #     view1_test, view2_test = test_data
                    #     test_view1 = prepare_for_attention(view1_test)
                    #     test_view2 = prepare_for_attention(view2_test)
                    #     tensor_view1 = convert_to_tensor(test_view1)
                    #     tensor_view2 = convert_to_tensor(test_view2)
                        
                    #     with torch.no_grad():
                    #         processed_view1 = apply_self_attention(tensor_view1, self.view1_attention, self.device)
                    #         processed_view2 = apply_self_attention(tensor_view2, self.view2_attention, self.device)
                            
                    #         # 应用交叉注意力
                    #         cross_view1 = apply_cross_attention(processed_view1, processed_view2, self.cross_attention1, self.device)
                    #         cross_view2 = apply_cross_attention(processed_view2, processed_view1, self.cross_attention2, self.device)
                            
                    #         # 拼接特征并分类
                    #         combined_features = torch.cat([cross_view1.mean(dim=1), cross_view2.mean(dim=1)], dim=1)
                    #         predictions = self.classifier(combined_features)
                    #         predicted_labels = torch.argmax(predictions, dim=1)
                            
                    #         # 计算测试集准确率
                    #         if labels_test is not None:
                    #             test_labels = torch.tensor(labels_test, dtype=torch.long).to(self.device)
                    #             accuracy = torch.mean((predicted_labels == test_labels - 1).float())
                    #             print(f"  测试集准确率: {accuracy:.4f}")
                    
        
        return loss_history, processed_view1, processed_view2

# 示例用法函数
def demo_attention_cca():
    """
    演示AttentionCCA的使用方法，包括模型训练过程和标签分类
    """
    # 四个视图，分别为：（400，512）、（400，59）、（400，864）、（400，254）
    mat_data = sio.loadmat("D:\本科毕业设计\Python_Projects\DataSets\数据集\ORL.mat")
    view1_data = mat_data['fea'][0][0]
    view2_data = mat_data['fea'][0][1]

    labels = mat_data['gt'].squeeze()
    
    # 创建配置
    config = {
        'view1_input_dim': view1_data.shape[1],
        'view2_input_dim': view2_data.shape[1],
        'view1_output_dim': 50,  # 指定降维后的输出维度
        'view2_output_dim': 50,  # 指定降维后的输出维度
        'attention_type': 'multihead',
        'num_heads': 4,
        'hidden_dim': 128,
        'use_gpu': True,
        'num_classes': 40,  # 40个类别
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
    print(f"  视图1形状: {untrained_view1.squeeze().shape}")
    print(f"  视图2形状: {untrained_view2.squeeze().shape}")

    # 准备训练数据
    print("\n===== 开始训练模型 =====")
    # 分割训练和测试数据
    view1_train, view1_test, view2_train, view2_test, labels_train, labels_test = split_train_test(view1_data, view2_data, labels, test_ratio=0.2)
    train_data = (view1_train, view2_train)
    test_data = (view1_test, view2_test)
    
    # 训练自注意力模型
    print("===== 训练自注意力模型 =====")
    self_loss_history, processed_view1, processed_view2 = model.train_model(
        train_data=train_data,
        test_data=test_data,
        labels_test=labels_test,
        labels=labels_train,
        num_epochs=50,  # 训练轮数
        batch_size=view1_train.shape[0],  # 批次大小
        learning_rate=0.001,  # 学习率
        train_phase='self_attention'
    )

    # 计算自注意力模型输出数据的结构复杂度
    print("\n===== 计算自注意力模型输出数据的结构复杂度 =====")
    #计算PDS分数
    pds_view1 = compute_pds(torch.squeeze(processed_view1,dim = 1).detach().cpu().numpy())
    pds_view2 = compute_pds(torch.squeeze(processed_view2,dim = 1).detach().cpu().numpy())
    print(f"  视图1 PDS分数: {pds_view1:.4f}")
    print(f"  视图2 PDS分数: {pds_view2:.4f}")
    
    # # 计算MNC分数
    # mnc_view1 = compute_mnc(torch.squeeze(processed_view1,dim = 1).detach().cpu().numpy())
    # mnc_view2 = compute_mnc(torch.squeeze(processed_view2,dim = 1).detach().cpu().numpy())
    # print(f"  视图1 MNC分数: {mnc_view1:.4f}")
    # print(f"  视图2 MNC分数: {mnc_view2:.4f}")
    
    # 初始化分类器
    print("\n===== 初始化分类器 =====")
    combined_dim = (model.config['view1_output_dim'] or model.config['view1_input_dim']) + \
                      (model.config['view2_output_dim'] or model.config['view2_input_dim'])
    model.classifier = nn.Linear(combined_dim, model.config.get('num_classes', 40))

    # 训练交叉注意力模型
    print("\n===== 训练交叉注意力模型和分类器 =====")
    model.config['enable_cross_attention'] = True
    
    # 使用自注意力模型的输出，使用加权后的输出作为交叉注意力的输入
    train_data = (abs(pds_view1) * torch.squeeze(processed_view1,dim = 1).detach().cpu().numpy(), abs(pds_view2) * torch.squeeze(processed_view2,dim = 1).detach().cpu().numpy())
    train_data = (torch.squeeze(processed_view1,dim = 1).detach().cpu().numpy(), torch.squeeze(processed_view2,dim = 1).detach().cpu().numpy())
    cross_loss_history, processed_view1, processed_view2 = model.train_model(
        train_data=train_data,
        test_data=test_data,
        labels_test=labels_test,
        labels=labels_train,
        num_epochs=500,  # 训练轮数
        batch_size=view1_train.shape[0],  # 批次大小
        learning_rate=0.001,  # 学习率
        train_phase='cross_attention'
    )

    # 保存所有模型
    model.save_models(
        'view1_attention_model.pth', 
        'view2_attention_model.pth',
        'cross_attention1_model.pth',
        'cross_attention2_model.pth',
        'classifier_model.pth'
    )
    print("\n所有模型已保存")
    
    # 使用训练后的模型处理数据
    print("\n===== 训练后模型的处理结果 =====")
    trained_view1, trained_view2 = model.process_views(view1_test, view2_test)

    # 打印结果形状
    print(f"\n测试数据形状:")
    print(f"  视图1形状: {view1_test.shape}")
    print(f"  视图2形状: {view2_test.shape}")
    print(f"训练后处理结果形状:")
    print(f"  视图1形状: {trained_view1.squeeze().shape}")
    print(f"  视图2形状: {trained_view2.squeeze().shape}")

    # 使用训练后的模型，得到测试集的输出结果，进而可以使用训练好的全连接分类器，得到测试集的预测标签
    print("\n===== 测试集的预测标签 =====")
    # 先拼接，再分类
    if not isinstance(trained_view1, torch.Tensor):
        trained_view1 = torch.tensor(trained_view1, dtype=torch.float32).to(model.device)
    if not isinstance(trained_view2, torch.Tensor):
        trained_view2 = torch.tensor(trained_view2, dtype=torch.float32).to(model.device)
    combined_features = torch.cat([torch.squeeze(trained_view1,dim = 1), torch.squeeze(trained_view2,dim = 1)], dim=1).to(model.device)
    
    predictions = model.classifier(combined_features)
    
    # 得到预测标签
    predicted_labels = torch.argmax(predictions, dim=1)
    # 计算准确率
    accuracy = torch.mean((predicted_labels == torch.tensor(labels_test, dtype=torch.long).to(model.device) - 1).float())
    print(f"  测试集准确率: {accuracy:.4f}")

    return trained_view1, trained_view2


if __name__ == "__main__":
    # 运行演示
    demo_attention_cca()