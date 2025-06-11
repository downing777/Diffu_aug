import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.recommender import BPR, GCN
from utils.dataloader import load_data, split_train_test, build_interaction_matrix
from utils.eval import evaluate
import numpy as np
from tqdm import tqdm

def normalize_adj(adj):
    """归一化邻接矩阵"""
    rowsum = np.array(adj.sum(1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = np.diag(d_inv)
    return adj.dot(d_mat).transpose().dot(d_mat)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """将scipy稀疏矩阵转换为torch稀疏张量"""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def train_bpr(train_u, train_i, test_u, test_i, n_users, n_items):
    # 准备负样本
    train_dataset = TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_i))
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BPR(n_users, n_items).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练循环
    for epoch in range(30):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            u, i = batch
            u, i = u.to(device), i.to(device)
            
            # 采样负样本
            j = torch.randint(1, n_items, i.size(), device=device)
            
            # 计算BPR损失
            optimizer.zero_grad()
            pos_score, neg_score = model(u, i, j)
            loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
        
        # 评估
        model.eval()
        with torch.no_grad():
            scores = model.predict(torch.arange(n_users).to(device))
            evaluate(scores.cpu().numpy(), test_u, test_i, train_u, train_i)
    
    return model

def train_gcn(train_u, train_i, test_u, test_i, n_users, n_items):
    # 构建邻接矩阵
    adj = build_interaction_matrix(train_u, train_i, n_users, n_items)
    adj = normalize_adj(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GCN(n_users, n_items).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练循环
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        
        # GCN前向传播
        user_emb, item_emb = model(adj)
        
        # 采样正负样本对
        u = torch.LongTensor(np.random.choice(train_u, 1024)).to(device)
        i = torch.LongTensor(np.random.choice(train_i, 1024)).to(device)
        j = torch.LongTensor(np.random.randint(1, n_items, 1024)).to(device)
        
        # 计算BPR损失
        pos_score = (user_emb[u] * item_emb[i]).sum(dim=-1)
        neg_score = (user_emb[u] * item_emb[j]).sum(dim=-1)
        loss = -torch.log(torch.sigmoid(pos_score - neg_score)).mean()
        
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
        
        # 评估
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                user_emb, item_emb = model(adj)
                scores = model.predict(torch.arange(n_users).to(device), 
                                     user_emb, item_emb)
                evaluate(scores.cpu().numpy(), test_u, test_i, train_u, train_i)
    
    return model

def compare_models():
    # 加载原始数据
    orig_u, orig_i = load_data("data/interactions.txt")
    n_users, n_items = max(orig_u) + 1, max(orig_i) + 1
    train_u, train_i, test_u, test_i = split_train_test(orig_u, orig_i)
    
    print("=== Training on Original Data ===")
    orig_bpr = train_bpr(train_u, train_i, test_u, test_i, n_users, n_items)
    #orig_gcn = train_gcn(train_u, train_i, test_u, test_i, n_users, n_items)
    
    # 加载增强数据
    aug_u, aug_i = load_data("data/augmented_interactions.txt")
    aug_train_u, aug_train_i = np.concatenate([train_u, aug_u]), np.concatenate([train_i, aug_i])
    
    print("\n=== Training on Augmented Data ===")
    aug_bpr = train_bpr(aug_train_u, aug_train_i, test_u, test_i, n_users, n_items)
    #aug_gcn = train_gcn(aug_train_u, aug_train_i, test_u, test_i, n_users, n_items)

if __name__ == "__main__":
    compare_models()