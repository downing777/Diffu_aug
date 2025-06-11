import torch
from models.diffusion import DiffusionModel
from utils.dataloader import load_data, get_user_items_dict
from tqdm import tqdm  # 添加这行

import torch
from tqdm import tqdm
import random

def guided_augmentation(model_path, data_path, output_path, n_aug=3, augment_ratio=0.2):
    # 加载数据
    user_ids, item_ids = load_data(data_path)
    user_items = get_user_items_dict(user_ids, item_ids)
    n_users, n_items = max(user_ids)+1, max(item_ids)+1
    all_items = torch.arange(1, n_items+1)
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = argparse.Namespace(
        hidden_size=64,
        num_diffusion_timesteps=1000,
        guidance_scale=2.0
    )
    model = DiffusionModel(n_users, n_items, device, args).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_items = all_items.to(device)
    
    # 随机选择20%的用户进行增强
    all_user_ids = list(user_items.keys())
    num_augment_users = int(len(all_user_ids) * augment_ratio)
    augment_users = random.sample(all_user_ids, num_augment_users)
    augment_users_set = set(augment_users)  # 用于快速查找
    
    # 生成增强数据
    with open(output_path, 'w') as f:
        # 首先写入所有原始数据
        # for u, items in user_items.items():
        #     for i in items:
        #         f.write(f"{u} {i}\n")
        
        # 只对选中的用户生成增强数据
        for user in tqdm(all_user_ids, desc="Augmenting users"):
            if user not in augment_users_set:
                continue
                
            hist = torch.LongTensor(list(user_items[user])).to(device)
            
            for _ in range(n_aug):
                with torch.no_grad():
                    try:
                        generated = model.generate(
                            user_id=user,
                            user_history=hist,
                            all_items=all_items,
                            steps=50,
                            topk=3
                        )
                        for item in generated.cpu().numpy():
                            f.write(f"{user} {item}\n")
                    except Exception as e:
                        print(f"Error generating for user {user}: {str(e)}")
                        continue

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='diffusion_guided.pth')
    parser.add_argument('--data_path', default='data/interactions.txt')
    parser.add_argument('--output_path', default='data/augmented_guided.txt')
    parser.add_argument('--n_aug', type=int, default=3)
    parser.add_argument('--augment_ratio', type=float, default=0.2)
    args = parser.parse_args()
    
    guided_augmentation(args.model_path, args.data_path, args.output_path, args.n_aug, args.augment_ratio)