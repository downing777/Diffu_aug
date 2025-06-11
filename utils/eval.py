import numpy as np
from collections import defaultdict

def evaluate(scores, test_u, test_i, train_u, train_i, topk=10):
    """评估推荐性能"""
    # 构建训练集用户-物品字典
    train_user_items = defaultdict(set)
    for u, i in zip(train_u, train_i):
        train_user_items[u].add(i)
    
    # 计算指标
    hit, ndcg = 0, 0
    test_user_items = defaultdict(set)
    for u, i in zip(test_u, test_i):
        test_user_items[u].add(i)
    
    for u in test_user_items:
        # 排除训练集中已交互的物品
        rated = train_user_items.get(u, set())
        candidates = [i for i in range(scores.shape[1]) if i not in rated]
        
        # 获取预测分数
        user_score = scores[u, candidates]
        item_idx = np.argsort(-user_score)[:topk]
        
        # 计算HR和NDCG
        pred_items = np.array(candidates)[item_idx]
        real_items = test_user_items[u]
        
        for i, item in enumerate(pred_items):
            if item in real_items:
                hit += 1
                ndcg += 1 / np.log2(i + 2)
    
    hit_rate = hit / len(test_u)
    ndcg = ndcg / len(test_u)
    
    print(f"HR@{topk}: {hit_rate:.4f}, NDCG@{topk}: {ndcg:.4f}")
    return hit_rate, ndcg