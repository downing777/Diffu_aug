import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models.diffusion import DiffusionModel
from utils.dataloader import load_data, get_user_items_dict, split_train_test
import numpy as np
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset

parser = argparse.ArgumentParser()

class InteractionDataset(Dataset):
    def __init__(self, user_items, item_num, max_len):
        self.user_items = list(user_items.items())  # 转换为(user, items)列表
        self.item_num = item_num
        self.max_len = max_len
        
    def __len__(self):
        return len(self.user_items)
    
    def __getitem__(self, idx):
        user, items = self.user_items[idx]
        # 确保items是列表
        items = list(items)[:self.max_len] if isinstance(items, (set, list)) else [items]
        # 填充或截断
        items = items[:self.max_len] + [0] * (self.max_len - len(items))
        return torch.LongTensor([user]), torch.LongTensor(items)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/interactions.txt')
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_seq_len', type=int, default=20)
    parser.add_argument('--guidance_scale', type=float, default=2.0)
    parser.add_argument("--num_diffusion_timesteps",
                    default=1000,
                    type=int,
                    help="the number of timesteps")
    args = parser.parse_args()

    # 加载数据
    user_ids, item_ids = load_data(args.data_path)
    user_items = get_user_items_dict(user_ids, item_ids)
    n_users, n_items = max(user_ids)+1, max(item_ids)+1
    
    # 准备数据集
    dataset = InteractionDataset(user_items, n_items, args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DiffusionModel(
        user_num=n_users,
        item_num=n_items,
        device=device,
        args=args
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 训练循环
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for user_batch, item_batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            user_batch = user_batch.squeeze().to(device)
            item_batch = item_batch.to(device)
            
            # 随机时间步
            t = torch.randint(0, args.n_steps, (item_batch.size(0),), device=device)
            
            # 使用item_batch自身作为guidance
            loss = model(
                x=item_batch,
                t=t,
                guidance=item_batch  # 使用相同的物品序列作为guidance
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
    torch.save(model.state_dict(), "diffusion_guided.pth")

if __name__ == "__main__":
    train()