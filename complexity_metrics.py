import numpy as np
import torch
from typing import Union, Tuple
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity


def calculate_feature_dimension(data: Union[np.ndarray, torch.Tensor]) -> int:
    """
    计算数据集的维度（特征数量）
    
    参数:
        data: 输入数据，可以是numpy数组或PyTorch张量
        
    返回:
        int: 数据的特征维度
    """
    if isinstance(data, np.ndarray):
        return data.shape[-1]
    elif isinstance(data, torch.Tensor):
        return data.size(-1)
    else:
        raise ValueError("输入数据类型必须是numpy数组或PyTorch张量")


def calculate_sample_density(data: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算样本密度（样本数量/特征数量）
    
    参数:
        data: 输入数据，可以是numpy数组或PyTorch张量
        
    返回:
        float: 样本密度值
    """
    num_samples = data.shape[0] if len(data.shape) > 1 else 1
    num_features = calculate_feature_dimension(data)
    return num_samples / num_features


def calculate_class_balance(labels: Union[np.ndarray, torch.Tensor]) -> float:
    """
    计算类别平衡度（熵值）
    
    参数:
        labels: 类别标签
        
    返回:
        float: 类别平衡度（熵值），值越大表示类别分布越平衡
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    
    unique, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    return entropy


def calculate_dataset_complexity(data: Union[np.ndarray, torch.Tensor], 
                                labels: Union[np.ndarray, torch.Tensor] = None) -> Tuple[float, float, float]:
    """
    计算数据集的综合复杂度指标
    
    参数:
        data: 输入数据
        labels: 可选，类别标签
        
    返回:
        tuple: (维度, 样本密度, 类别平衡度)
    """
    dim = calculate_feature_dimension(data)
    density = calculate_sample_density(data)
    
    if labels is not None:
        balance = calculate_class_balance(labels)
    else:
        balance = 0.0
    
    return dim, density, balance

def compute_pds(X):
    """
    计算数据集的全局结构复杂度PDS
    
    参数:
    X : numpy数组, 形状为(n_samples, n_features)
        输入的高维数据集
    
    返回:
    pds : float
        PDS分数，表示数据集的全局结构复杂度
    """
    # Step 1: 计算成对距离矩阵
    pairwise_distances = pdist(X)
    
    # Step 2: 计算距离的标准差和均值
    std_dev = np.std(pairwise_distances)
    mean_val = np.mean(pairwise_distances)
    
    # 计算PDS = log(标准差 / 均值)
    pds = np.log(std_dev / mean_val)
    
    return pds

def compute_mnc(X, k=5):
    """
    计算数据集的局部结构复杂度MNC
    
    参数:
    X : numpy数组, 形状为(n_samples, n_features)
        输入的高维数据集
    k : int, 默认=5
        k近邻参数
    
    返回:
    mnc_score : float
        MNC分数，表示数据集的局部结构复杂度
    """
    n_samples = X.shape[0]
    
    # Step 1: 计算kNN相似矩阵 M_kNN
    M_kNN = np.zeros((n_samples, n_samples))
    
    # 使用k+1因为包含自身
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    # 构建kNN矩阵
    for i in range(n_samples):
        # 跳过自身 (indices[i, 0] 是自身)
        for rank, j in enumerate(indices[i, 1:], start=1):  # rank从1开始
            M_kNN[i, j] = max(0, k - rank + 1)
    
    # Step 2: 计算SNN相似矩阵 M_SNN
    M_SNN = np.zeros((n_samples, n_samples))
    
    # 为每个点存储其k近邻的索引和排名
    neighbor_ranks = {}
    for i in range(n_samples):
        # 存储每个邻居在i的kNN中的排名
        for rank, j in enumerate(indices[i, 1:], start=1):
            if j not in neighbor_ranks:
                neighbor_ranks[j] = {}
            neighbor_ranks[j][i] = rank
    
    # 计算SNN相似度
    for i in range(n_samples):
        for j in range(i + 1, n_samples):  # 只计算上三角，然后对称复制
            if i == j:
                continue
                
            snn_value = 0
            # 检查所有共同的邻居
            if i in neighbor_ranks and j in neighbor_ranks:
                common_neighbors = set(neighbor_ranks[i].keys()) & set(neighbor_ranks[j].keys())
                
                for m in common_neighbors:
                    if m != i and m != j:  # 排除自身
                        rank_i = neighbor_ranks[i].get(m, 0)
                        rank_j = neighbor_ranks[j].get(m, 0)
                        
                        if rank_i > 0 and rank_j > 0:  # 确保m在两者的kNN中
                            snn_value += (k + 1 - rank_i) * (k + 1 - rank_j)
            
            M_SNN[i, j] = snn_value
            M_SNN[j, i] = snn_value  # 对称矩阵
    
    # Step 3: 计算kNN和SNN矩阵之间的差异（余弦相似度）
    cosine_similarities = []
    
    for i in range(n_samples):
        # 获取第i行
        kNN_row = M_kNN[i, :]
        SNN_row = M_SNN[i, :]
        
        # 计算余弦相似度
        # 使用1 - cosine()得到相似度，然后确保在[0,1]范围内
        cos_sim = 1 - cosine(kNN_row, SNN_row)
        cos_sim = max(0, min(1, cos_sim))  # 确保在[0,1]范围内
        
        cosine_similarities.append(cos_sim)
    
    # 计算平均MNC分数
    mnc_score = np.mean(cosine_similarities)
    
    return mnc_score

# # 示例用法
# if __name__ == "__main__":
#     # 生成示例数据
#     np.random.seed(42)
#     X = np.random.randn(100, 10)  # 100个样本，10维特征
    
#     # 计算MNC
#     mnc_score = compute_mnc(X, k=5)
#     print(f"MNC score: {mnc_score:.4f}")
    
#     # 测试不同k值
#     for k in [3, 5, 7, 10]:
#         mnc_score = compute_mnc(X, k=k)
#         print(f"MNC score (k={k}): {mnc_score:.4f}")

# # 示例用法
# if __name__ == "__main__":
#     # 生成示例数据
#     np.random.seed(42)
#     X = np.random.randn(100, 10)  # 100个样本，10维特征
    
#     # 计算PDS
#     pds_score = compute_pds(X)
#     print(f"PDS score: {pds_score:.4f}")