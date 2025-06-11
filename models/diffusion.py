import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, user_num, item_num, device, args):
        super().__init__()
        self.device = device
        self.item_emb = nn.Embedding(item_num + 2, args.hidden_size, padding_idx=0)  # +2 for padding/mask
        self.user_emb = nn.Embedding(user_num, args.hidden_size)
        self.mask_token = item_num + 1
        
        # Guidance相关参数
        self.guidance_scale = args.guidance_scale  # 控制guidance强度
        self.exclude_historical = True  # 强制排除历史交互
        
        # 噪声预测网络（带guidance输入）
        self.noise_predictor = nn.Sequential(
            nn.Linear(args.hidden_size * 2, args.hidden_size * 2),  # 输入维度修正
            nn.SiLU(),
            nn.Linear(args.hidden_size * 2, args.hidden_size)      # 保持输出维度
        )
        
        # 扩散参数
        self.n_steps = args.num_diffusion_timesteps
        self.register_buffer('betas', torch.linspace(1e-4, 0.02, self.n_steps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))

    def forward(self, x, t, guidance):
        """x: [batch, seq_len], t: [batch], guidance: [batch, seq_len]"""
        batch_size, seq_len = x.shape
        
        # 嵌入处理
        x_emb = self.item_emb(x)  # [batch, seq_len, hidden]
        
        # 处理guidance (使用相同嵌入层)
        guidance_emb = self.item_emb(guidance) * (guidance > 0).unsqueeze(-1)
        guide_vector = guidance_emb.mean(dim=1)  # [batch, hidden]
        
        # 加噪过程
        noise = torch.randn_like(x_emb)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        noisy_emb = alpha_bar.sqrt() * x_emb + (1 - alpha_bar).sqrt() * noise
        
        # 拼接guidance信息
        guide_expanded = guide_vector.unsqueeze(1).expand(-1, seq_len, -1)
        pred_input = torch.cat([noisy_emb, guide_expanded], dim=-1)
        
        # 确保输入形状正确 [batch*seq_len, hidden*2]
        pred_input = pred_input.reshape(-1, pred_input.size(-1))
        
        # 预测噪声
        pred_noise = self.noise_predictor(pred_input)
        pred_noise = pred_noise.reshape(batch_size, seq_len, -1)
        
        return F.mse_loss(noise, pred_noise)

    @torch.no_grad()
    def generate(self, user_id, user_history, all_items, steps=50, topk=5):
        """生成不包含历史交互的物品"""
        # 确保输入在正确设备上
        user_id_tensor = torch.tensor([user_id]).to(self.device)
        user_history = user_history.to(self.device)
        all_items = all_items.to(self.device)
        
        # 获取用户特征
        user_vec = self.user_emb(user_id_tensor)
        hist_emb = self.item_emb(user_history) * (user_history > 0).unsqueeze(-1)
        guide_vector = hist_emb.mean(dim=0) + user_vec.squeeze(0)  # [hidden_size]
        
        # 排除历史物品
        candidate_mask = torch.ones_like(all_items, dtype=bool)
        if len(user_history) > 0:
            candidate_mask[user_history] = False
        candidates = all_items[candidate_mask]
        
        # 从噪声开始生成
        x = torch.randn(1, topk, self.item_emb.weight.size(1)).to(self.device)
        
        for t in reversed(range(steps)):
            # 拼接guidance信息
            guide_expanded = guide_vector.unsqueeze(0).expand(topk, -1).unsqueeze(0)
            pred_input = torch.cat([x, guide_expanded], dim=-1)
            
            # 预测噪声
            pred_noise = self.noise_predictor(pred_input)
            x = self.p_sample(x, pred_noise, torch.tensor([t]).to(self.device))
        
        # 选择最匹配的候选物品
        candidate_embs = self.item_emb(candidates)
        scores = torch.matmul(x.squeeze(0), candidate_embs.T)
        selected = scores.topk(min(topk, len(candidates)), dim=-1).indices
        
        return candidates[selected]

    def p_sample(self, x, noise_pred, t):
        alpha = self.alphas[t].view(-1, 1, 1)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        noise = torch.randn_like(x) if t > 0 else 0
        
        x = (x - (1 - alpha)/ (1 - alpha_bar).sqrt() * noise_pred) / alpha.sqrt()
        return x + noise * (1 - alpha).sqrt()