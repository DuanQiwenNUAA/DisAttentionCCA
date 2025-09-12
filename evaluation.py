import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def evaluate_kmeans_clustering(view1_data, view2_data, n_clusters=3):
    """
    评估KMeans聚类效果，计算聚类指标
    
    参数:
        view1_data: 第一个视图的数据
        view2_data: 第二个视图的数据
        n_clusters: 聚类数量，默认为3
    
    返回:
        dict: 包含聚类评估指标的字典
    """
    # 确保数据是numpy数组
    if isinstance(view1_data, torch.Tensor):
        view1_data = view1_data.cpu().numpy()
    if isinstance(view2_data, torch.Tensor):
        view2_data = view2_data.cpu().numpy()
    
    # 如果数据是三维的，展平为二维数组
    if view1_data.ndim > 2:
        view1_data = view1_data.reshape(view1_data.shape[0], -1)
    if view2_data.ndim > 2:
        view2_data = view2_data.reshape(view2_data.shape[0], -1)
    
    # 执行KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    # 分别计算每个视图的聚类指标
    view1_labels = kmeans.fit_predict(view1_data)
    view1_silhouette = silhouette_score(view1_data, view1_labels)
    view1_calinski = calinski_harabasz_score(view1_data, view1_labels)
    
    view2_labels = kmeans.fit_predict(view2_data)
    view2_silhouette = silhouette_score(view2_data, view2_labels)
    view2_calinski = calinski_harabasz_score(view2_data, view2_labels)
    
    # 计算联合聚类指标
    joint_data = np.concatenate([view1_data, view2_data], axis=1)
    joint_labels = kmeans.fit_predict(joint_data)
    joint_silhouette = silhouette_score(joint_data, joint_labels)
    
    # 返回评估结果
    return {
        'n_clusters': n_clusters,
        'view1_silhouette': view1_silhouette,
        'view2_silhouette': view2_silhouette,
        'joint_silhouette': joint_silhouette,
        'view1_calinski_harabasz_score': view1_calinski,
        'view2_calinski_harabasz_score': view2_calinski
    }


def evaluate_attention_effect(view1_data, view2_data):
    """
    评估自注意力机制的效果，计算两个视图之间的相关性
    
    参数:
        view1_data: 第一个视图的数据
        view2_data: 第二个视图的数据
    
    返回:
        dict: 包含视图间相关性评估指标的字典
    """
    # 确保数据是numpy数组
    if isinstance(view1_data, torch.Tensor):
        view1_data = view1_data.cpu().numpy()
    if isinstance(view2_data, torch.Tensor):
        view2_data = view2_data.cpu().numpy()
        
    # 计算数据形状
    view1_shape = view1_data.shape
    view2_shape = view2_data.shape
    
    # 计算数据统计量
    view1_mean = np.mean(view1_data)
    view2_mean = np.mean(view2_data)
    view1_var = np.var(view1_data)
    view2_var = np.var(view2_data)
    
    # 计算两个视图之间的相关性
    # 展平数据以计算相关性
    view1_flat = view1_data.flatten()
    view2_flat = view2_data.flatten()
    cross_correlation = np.corrcoef(view1_flat, view2_flat)[0, 1]
    
    # 返回评估结果
    return {
        'view1_shape': view1_shape,
        'view2_shape': view2_shape,
        'view1_mean': view1_mean,
        'view2_mean': view2_mean,
        'view1_variance': view1_var,
        'view2_variance': view2_var,
        'cross_correlation': cross_correlation
    }