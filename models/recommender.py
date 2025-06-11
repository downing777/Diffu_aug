import torch
import torch.nn as nn
import torch.nn.functional as F

class BPR(nn.Module):
    """Bayesian Personalized Ranking"""
    def __init__(self, user_num, item_num, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(user_num, dim)
        self.item_emb = nn.Embedding(item_num, dim)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        
    def forward(self, u, i, j):
        """u: user ids, i: positive items, j: negative items"""
        u_emb = self.user_emb(u)
        i_emb = self.item_emb(i)
        j_emb = self.item_emb(j)
        
        pos_score = (u_emb * i_emb).sum(dim=-1)
        neg_score = (u_emb * j_emb).sum(dim=-1)
        
        return pos_score, neg_score
    
    def predict(self, u):
        u_emb = self.user_emb(u)
        all_item_emb = self.item_emb.weight
        return torch.matmul(u_emb, all_item_emb.T)

class GCN(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, user_num, item_num, dim=64, layers=3):
        super().__init__()
        self.user_emb = nn.Embedding(user_num, dim)
        self.item_emb = nn.Embedding(item_num, dim)
        self.layers = layers
        self.W = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layers)])
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        
    def forward(self, adj):
        # 初始特征
        features = torch.cat([self.user_emb.weight, self.item_emb.weight])
        
        # GCN传播
        for i in range(self.layers):
            features = torch.spmm(adj, features)  # 稀疏矩阵乘法
            features = self.W[i](features)
            if i != self.layers - 1:
                features = F.relu(features)
                
        user_emb, item_emb = torch.split(features, [self.user_emb.num_embeddings, 
                                                    self.item_emb.num_embeddings])
        return user_emb, item_emb
    
    def predict(self, u, user_emb, item_emb):
        return torch.matmul(user_emb[u], item_emb.T)