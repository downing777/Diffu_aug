import numpy as np
from sklearn.model_selection import train_test_split
from collections import defaultdict

def load_data(file_path):
    """加载交互数据"""
    user_ids, item_ids = [], []
    with open(file_path, 'r') as f:
        for line in f:
            u, i = line.strip().split()
            user_ids.append(int(u))
            item_ids.append(int(i))
    return np.array(user_ids), np.array(item_ids)

def split_train_test(user_ids, item_ids, test_size=0.2):
    """划分训练集和测试集"""
    train_u, test_u, train_i, test_i = train_test_split(
        user_ids, item_ids, test_size=test_size, random_state=42
    )
    return train_u, train_i, test_u, test_i

def build_interaction_matrix(user_ids, item_ids, n_users, n_items):
    """构建用户-物品交互矩阵"""
    mat = np.zeros((n_users, n_items))
    for u, i in zip(user_ids, item_ids):
        mat[u][i] = 1
    return mat

def get_user_items_dict(user_ids, item_ids):
    """返回列表形式的历史记录而非集合"""
    user_items = defaultdict(list)
    for u, i in zip(user_ids, item_ids):
        user_items[u].append(i)
    return user_items