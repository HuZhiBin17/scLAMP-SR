import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Poisson, Bernoulli
from torch.distributions import kl_divergence as kld
import numpy as np
import anndata
import scipy
# Only the legacy standalone CLI needs the Scanpy/Muon preprocessing stack.
# Model imports are used by held-out evaluators and should remain lightweight.
mu = None
sc = None
epi = None
import traceback
from torch.autograd import Function
from sklearn.neighbors import kneighbors_graph
from utils import *
from count_distributions import negative_binomial_from_mean_dispersion


class encoder(nn.Module):
    def __init__(self, rna_dim, atac_dim, rna_pro_dim, atac_pro_dim, hidden_dim=512, latent_dim=128, dropout_rate=0.2):
        super().__init__()
        self.rna_encoder = nn.Sequential(
            nn.Linear(rna_pro_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        self.atac_encoder = nn.Sequential(
            nn.Linear(atac_pro_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 共享层用于对齐表示
        self.shared_layer = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 分别计算均值和方差
        self.rna_mu = nn.Linear(hidden_dim // 4, latent_dim)
        self.rna_var = nn.Linear(hidden_dim // 4, latent_dim)
        self.atac_mu = nn.Linear(hidden_dim // 4, latent_dim)
        self.atac_var = nn.Linear(hidden_dim // 4, latent_dim)

        # 添加共享的潜空间参数
        self.shared_mu = nn.Linear(hidden_dim // 2, latent_dim)  # 用于融合后的共享表示
        self.shared_var = nn.Linear(hidden_dim // 2, latent_dim)

    def reparameterize(self, mu, var):
        z = Normal(mu, var.sqrt()).rsample()
        return z

    def forward(self, rna_data, atac_data):
        rna_h = self.rna_encoder(rna_data)
        atac_h = self.atac_encoder(atac_data)
        
        # 通过共享层
        rna_shared = self.shared_layer(rna_h)
        atac_shared = self.shared_layer(atac_h)
        
        # 计算各自的潜变量
        rna_mu = torch.clamp(self.rna_mu(rna_shared), min=-10, max=10)
        rna_var = torch.clamp(torch.exp(self.rna_var(rna_shared)), min=1e-6, max=1e8)
        z_rna = self.reparameterize(rna_mu, rna_var)

        atac_mu = torch.clamp(self.atac_mu(atac_shared), min=-10, max=10)
        atac_var = torch.clamp(torch.exp(self.atac_var(atac_shared)),min=1e-6, max=1e8)
        z_atac = self.reparameterize(atac_mu, atac_var)

        # 计算融合表示
        fused_h = (rna_h + atac_h) / 2
        shared_mu = torch.clamp(self.shared_mu(fused_h), min=-10, max=10)
        shared_var = torch.clamp(torch.exp(self.shared_var(fused_h)), min=1e-6, max=1e8)
        z_shared = self.reparameterize(shared_mu, shared_var)

        return z_rna, z_atac, z_shared, rna_mu, rna_var, atac_mu, atac_var, shared_mu, shared_var

class decoder(nn.Module):
    def __init__(self, rna_dim, atac_dim, hidden_dim=512, latent_dim=128, dropout_rate=0.2):
        super().__init__()
        # RNA decoder outputs mean parameter for negative binomial
        self.rna_decoder_mean = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, rna_dim),
            nn.Softplus()
        )

        self.rna_decoder_dispersion = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, rna_dim),
            nn.Softplus()
        )

        self.rna_decoder_poisson = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, rna_dim),
            nn.Softplus()
        )

        self.atac_decoder_mean = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, atac_dim),
            nn.Softplus()
        )

        self.atac_decoder_dispersion = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, atac_dim),
            nn.Softplus()
        )

        self.atac_decoder_bnl = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, atac_dim),
            # nn.Sigmoid()
            nn.Softplus()  #ps
        )

        self.rna_logits = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.atac_logits = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, z, sample=True):
        rna_mean = self.rna_decoder_mean(z) + 1e-6 
        rna_dispersion = self.rna_decoder_dispersion(z) + 1e-6 
        rna_poisson = self.rna_decoder_poisson(z) + 1e-6 

        atac_mean = self.atac_decoder_mean(z) + 1e-6 
        atac_dispersion = self.atac_decoder_dispersion(z) + 1e-6 
        atac_bnl = self.atac_decoder_bnl(z) + 1e-6 
        
        if sample:
            rna_weights = torch.softmax(self.rna_logits, dim=0)
            atac_weights = torch.softmax(self.atac_logits, dim=0)

            rna_lambda_1, rna_lambda_2 = rna_weights[0], rna_weights[1]
            atac_lambda_1, atac_lambda_2 = atac_weights[0], atac_weights[1]

            rna_NB_dist = negative_binomial_from_mean_dispersion(
                rna_mean,
                rna_dispersion,
            )
            rna_PS_dist = Poisson(rate=rna_poisson)

            atac_NB_dist = negative_binomial_from_mean_dispersion(
                atac_mean,
                atac_dispersion,
            )
            # atac_BNL_dist = Bernoulli(probs=atac_bnl)
            atac_BNL_dist = Poisson(rate=atac_bnl)

            rna_data = rna_lambda_1 * rna_NB_dist.sample() + rna_lambda_2 * rna_PS_dist.sample()
            atac_data = atac_lambda_1 * atac_NB_dist.sample() + atac_lambda_2 * atac_BNL_dist.sample()
        else:
            rna_data = rna_mean
            atac_data = atac_mean
            rna_NB_dist = None
            rna_PS_dist = None
            atac_NB_dist = None
            atac_BNL_dist = None
            rna_lambda_1, rna_lambda_2 = None, None
            atac_lambda_1, atac_lambda_2 = None, None
        
        outputs = {
            'rna_mean': rna_mean,
            'rna_dispersion': rna_dispersion,
            'rna_poisson': rna_poisson,
            'rna_nb_dist': rna_NB_dist,
            'rna_ps_dist': rna_PS_dist,
            'atac_mean': atac_mean,
            'atac_dispersion': atac_dispersion,
            'atac_bnl': atac_bnl,
            'atac_nb_dist': atac_NB_dist,
            'atac_bnl_dist': atac_BNL_dist,
            'rna_lambda_1': rna_lambda_1,
            'rna_lambda_2': rna_lambda_2,
            'atac_lambda_1': atac_lambda_1,
            'atac_lambda_2': atac_lambda_2
        }
        return rna_data, atac_data, outputs

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None



class scAPI(nn.Module):
    def __init__(self, rna_dim, atac_dim, rna_pro_dim, atac_pro_dim, hidden_dim=512, latent_dim=128, dropout_rate=0.2):
        super().__init__()
        self.encoder = encoder(rna_dim, atac_dim, rna_pro_dim, atac_pro_dim, hidden_dim, latent_dim, dropout_rate)
        self.decoder = decoder(rna_dim, atac_dim, hidden_dim, latent_dim, dropout_rate)

        self.adversarial_module = AdversarialAlignmentModule(
            latent_dim=latent_dim,
            hidden_dims=[256, 128],
            lambda_grl=1.0,
            discriminator_lr=1e-4
        )

        self.logits = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.latent_dim = latent_dim


    def forward(self, rna_data, atac_data, decoder_input_mode='shared'):
        z_rna, z_atac, z_shared, rna_mu, rna_var, atac_mu, atac_var, shared_mu, shared_var = self.encoder(rna_data, atac_data)

        # 根据输入模式选择 decoder 输入
        if decoder_input_mode == 'shared':
            decoder_input = z_shared
        elif decoder_input_mode == 'rna':
            decoder_input = z_rna
        elif decoder_input_mode == 'atac':
            decoder_input = z_atac
        else:
            raise ValueError(f"Invalid decoder_input_mode: {decoder_input_mode}")

        # 解码
        rna_data_, atac_data_, dist_params = self.decoder(decoder_input)

        adv_results = self.adversarial_module(
                z_shared, z_rna, z_atac, mode='adversarial'
        )
        
        dist_params['rna_mu'] = rna_mu
        dist_params['rna_var'] = rna_var
        dist_params['atac_mu'] = atac_mu
        dist_params['atac_var'] = atac_var
        dist_params['shared_mu'] = shared_mu
        dist_params['shared_var'] = shared_var
        dist_params['adv_loss'] = adv_results['adversarial_loss']


        return rna_data_, atac_data_, z_shared, z_rna, z_atac, dist_params

# 辅助函数
def compute_mmd(x, y, kernel='rbf', sigma=5.0):
    """计算最大均值差异（MMD）"""
    def gaussian_kernel(a, b, sigma):
        dist = ((a.unsqueeze(1) - b.unsqueeze(0)) ** 2).sum(2)
        return torch.exp(-dist / (2 * sigma ** 2))

    K_xx = gaussian_kernel(x, x, sigma)
    K_yy = gaussian_kernel(y, y, sigma)
    K_xy = gaussian_kernel(x, y, sigma)

    # mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
    mmd = (K_xx + K_yy - 2 * K_xy).sum()
    return mmd

def wasserstein_distance(x, y):
    """计算一维Wasserstein距离的近似"""
    x_sorted, _ = torch.sort(x, dim=0)
    y_sorted, _ = torch.sort(y, dim=0)
    return torch.mean(torch.abs(x_sorted - y_sorted))

def compute_alignment_loss(z_rna, z_atac, loss_type='mmd'):
    """计算对齐损失"""
    if loss_type == 'mmd':
        return compute_mmd(z_rna, z_atac)
    elif loss_type == 'wasserstein':
        # 对每个维度计算Wasserstein距离
        wd_loss = 0
        for i in range(z_rna.shape[1]):
            wd_loss += wasserstein_distance(z_rna[:, i:i+1], z_atac[:, i:i+1])
        return wd_loss / z_rna.shape[1]
    elif loss_type == 'cosine':
        # 余弦相似度损失
        cos_sim = F.cosine_similarity(z_rna.mean(0), z_atac.mean(0), dim=0)
        return 1 - cos_sim.mean()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

def compute_kld(z_mu, z_var, rna_mu, rna_var, atac_mu, atac_var):
    """
    计算KL散度
    z_mu: 潜变量均值
    z_var: 潜变量方差
    """

    kl_div_z = kld(
        Normal(z_mu, torch.sqrt(z_var)),
        Normal(0, 1)
    ).sum(dim=1)

    kl_div_rna = kld(
        Normal(rna_mu, torch.sqrt(rna_var)),
        Normal(0, 1)
    ).sum(dim=1)

    kl_div_atac = kld(
        Normal(atac_mu, torch.sqrt(atac_var)),
        Normal(0, 1)
    ).sum(dim=1)
    
    # 计算总的KL散度
    # kl_loss = kl_div_z.mean() + (kl_div_rna.mean() + kl_div_atac.mean()) / 2
    kl_loss = kl_div_z.mean()
    return kl_loss

def loss_function(rna_data, atac_data, dist_params, rna_z, atac_z, z_shared=None, 
                 gamma=0.1, alpha=1.0, beta=0.5, alignment_type='mmd'):
    """
    改进的损失函数
    gamma: KL散度权重
    alpha: 对齐损失权重
    beta: 共享表示权重
    alignment_type: 对齐损失类型 ('mmd', 'wasserstein', 'cosine')
    """
    # 原有的重建损失
    rna_data = rna_data.round().long()
    atac_data = atac_data.round().long()

    if dist_params['rna_nb_dist'] is not None:
        rna_nb_dist = dist_params['rna_nb_dist']
        rna_ps_dist = dist_params['rna_ps_dist']
        atac_nb_dist = dist_params['atac_nb_dist']
        atac_bnl_dist = dist_params['atac_bnl_dist']

        rna_lambda_1 = dist_params['rna_lambda_1']
        rna_lambda_2 = dist_params['rna_lambda_2']
        atac_lambda_1 = dist_params['atac_lambda_1']
        atac_lambda_2 = dist_params['atac_lambda_2']

        rna_nb_logprob = rna_nb_dist.log_prob(rna_data)
        rna_ps_logprob = rna_ps_dist.log_prob(rna_data)
        atac_nb_logprob = atac_nb_dist.log_prob(atac_data)
        # atac_bnl_logprob = atac_bnl_dist.log_prob((atac_data > 0).float())
        atac_bnl_logprob = atac_bnl_dist.log_prob(atac_data)

        rna_log_likelihood = rna_lambda_1 * rna_nb_logprob + rna_lambda_2 * rna_ps_logprob
        atac_log_likelihood = atac_lambda_1 * atac_nb_logprob + atac_lambda_2 * atac_bnl_logprob

        rna_recon_loss = -rna_log_likelihood.mean()
        atac_recon_loss = -atac_log_likelihood.mean()
    else:
        # 如果没有分布参数，使用MSE损失
        rna_recon_loss = F.mse_loss(dist_params['rna_mean'], rna_data.float())
        atac_recon_loss = F.mse_loss(dist_params['atac_mean'], atac_data.float())

    # KL散度损失

    kl_loss = compute_kld(dist_params['shared_mu'], dist_params['shared_var'], dist_params['rna_mu'], dist_params['rna_var'], dist_params['atac_mu'], dist_params['atac_var']) * alpha

    shared_reg_loss = 0
    if z_shared is not None:
        # 鼓励共享表示与各模态表示的相似性
        # shared_rna_loss = compute_mmd(z_shared, rna_z) * beta
        # shared_atac_loss = compute_mmd(z_shared, atac_z) * beta
        shared_rna_loss = compute_mmd(z_shared, rna_z) * beta
        shared_atac_loss = compute_mmd(z_shared, atac_z) * beta

        shared_reg_loss = shared_rna_loss + shared_atac_loss

    # 近邻
    # n_loss = neighborhood_preservation_loss(z_shared=z_shared, z_rna=rna_z, z_atac=atac_z, k=15) * gamma
    #对抗
    adv_loss = dist_params['adv_loss'] * gamma


    total_loss = rna_recon_loss + atac_recon_loss + shared_reg_loss + kl_loss + adv_loss
    return {
        'total_loss': total_loss,
        'rna_recon_loss': rna_recon_loss,
        'atac_recon_loss': atac_recon_loss,
        'kl_loss': kl_loss,
        # 'alignment_loss': alignment_loss,
        # 'adversarial_loss': adversarial_loss,
        'adv_loss': adv_loss,
        'shared_reg_loss': shared_reg_loss
    }
class MultiModalDataset(torch.utils.data.Dataset):
    """多模态数据集类"""
    def __init__(self, rna_adata, atac_adata):
        """
        初始化多模态数据集
        
        参数:
        rna_adata: AnnData对象，包含RNA-seq数据
        atac_adata: AnnData对象，包含ATAC-seq数据
        """
        self.rna_adata = rna_adata
        self.atac_adata = atac_adata
        # 提取表达矩阵
        self.rna_data = self.rna_adata.X
        self.atac_data = self.atac_adata.X

        self.rna_raw = self.rna_adata.raw.X
        self.atac_raw = self.atac_adata.raw.X

        # self.batch_key = self.rna_adata.obs['modality']
        
        # 转换为密集矩阵（如果是稀疏矩阵）
        if scipy.sparse.issparse(self.rna_data):
            self.rna_data = self.rna_data.toarray()
        if scipy.sparse.issparse(self.atac_data):
            self.atac_data = self.atac_data.toarray()
        if scipy.sparse.issparse(self.rna_raw):
            self.rna_raw = self.rna_raw.toarray()
        if scipy.sparse.issparse(self.atac_raw):
            self.atac_raw = self.atac_raw.toarray()

        # modality_map = {'GEX': 0, 'paired': 1, 'ATAC': 2}
        # self.batch_labels = self.rna_adata.obs['modality'].map(modality_map).values.astype(np.int64)
        
    def __len__(self):
        return len(self.rna_adata)
    
    def __getitem__(self, idx):
        rna_sample = self.rna_data[idx]
        atac_sample = self.atac_data[idx]
        
        rna_raw_sample = self.rna_raw[idx]
        atac_raw_sample = self.atac_raw[idx]
        # label = self.batch_labels[idx]
        return (
            torch.FloatTensor(rna_sample),
            torch.FloatTensor(atac_sample),
            torch.FloatTensor(rna_raw_sample),
            torch.FloatTensor(atac_raw_sample),
            # torch.tensor(label, dtype=torch.long)  # 返回标签
        )

def train_model(model, train_loader, optimizer, num_epochs=300, warmup_epochs=50, device='cuda'):
    """训练模型"""
    model.train()

    patience = 50
    # 前向传播
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        total_loss = 0
        total_batches = 0
        
        # 记录各个损失分量
        rna_loss_sum = 0
        atac_loss_sum = 0
        kl_loss_sum = 0
        ali_loss_sum = 0
        adv_loss_sum = 0
        nei_loss_sum = 0
        shared_reg_loss_sum = 0

        # 交替训练
        if epoch < warmup_epochs:
            decoder_input_mode = 'shared'
        else:
            # 计算在 warmup 后的 step index
            post_warmup_step = epoch - warmup_epochs

            if post_warmup_step % 2 == 0:
                decoder_input_mode = 'shared'
            else:
                # 交替 rna / atac
                if (post_warmup_step // 2) % 2 == 0:
                    decoder_input_mode = 'rna'
                else:
                    decoder_input_mode = 'atac'


        for batch_idx, (rna_batch, atac_batch, rna_raw, atac_raw) in enumerate(train_loader):
            rna_batch = rna_batch.to(device)
            atac_batch = atac_batch.to(device)
            rna_raw = rna_raw.to(device)
            atac_raw = atac_raw.to(device)

        # 每5步训练一次判别器
            if batch_idx % 5 == 0:
                # 单独训练判别器
                with torch.no_grad():
                    z_rna, z_atac, z_shared, *_ = model.encoder(rna_batch, atac_batch)
                
                disc_loss, disc_acc = model.adversarial_module.train_discriminator_step(
                    z_shared, z_rna, z_atac
                )            
                #print(f"Epoch {epoch}, Batch {batch_idx}, Discriminator Loss: {disc_loss:.4f}, Discriminator Accuracy: {disc_acc:.4f}")

            optimizer.zero_grad()
            # gamma : adv的权重
            # alpha : kl的权重
            # beta : share的权重
            rna_recon, atac_recon, z_shared, z_rna, z_atac, dist_params = model(rna_batch, atac_batch, decoder_input_mode=decoder_input_mode)
            losses = loss_function(
                rna_raw, atac_raw, dist_params, z_rna, z_atac, z_shared, gamma=0.1, alpha=1e-7, beta=1e-4, alignment_type='mmd'  # 降低KL散度的权重以增加稳定性
            )


            loss = losses['total_loss']

            # 防止梯度爆炸
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 反向传播和优化
            loss.backward()

            # total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
            # print(f"Total gradient norm: {total_norm}")

            optimizer.step()
   
            # 累计损失
            total_loss += loss.item()
            rna_loss_sum += losses['rna_recon_loss'].item()
            atac_loss_sum += losses['atac_recon_loss'].item()
            kl_loss_sum += losses['kl_loss'].item()
            adv_loss_sum += losses['adv_loss'].item()
            # ali_loss_sum += losses['alignment_loss'].item()
            # adv_loss_sum += losses['adversarial_loss'].item()
            shared_reg_loss_sum += losses['shared_reg_loss'].item()
            total_batches += 1
            
        if total_batches > 0:  # 避免除零错误
            avg_loss = total_loss / total_batches
            avg_rna_loss = rna_loss_sum / total_batches
            avg_atac_loss = atac_loss_sum / total_batches
            avg_kl_loss = kl_loss_sum / total_batches
            avg_adv_loss = adv_loss_sum / total_batches
            # avg_ali_loss = ali_loss_sum / total_batches
            # avg_adv_loss = adv_loss_sum / total_batches
            avg_shared_reg_loss = shared_reg_loss_sum / total_batches

            if avg_loss < best_loss - 1e-4:  # improvement threshold
                best_loss = avg_loss
                epochs_no_improve = 0
                # 如果需要，保存最佳模型
                # torch.save(model.state_dict(), 'best_model.pth')
            else:
                epochs_no_improve += 1

            print(f'Epoch {epoch+1}/{num_epochs}, '
                  f'mode: {decoder_input_mode}, '
                  f'TL: {avg_loss:f}, '
                  f'RNA: {avg_rna_loss:f}, '
                  f'ATAC: {avg_atac_loss:f}, '
                f'KL: {avg_kl_loss:f} '
                f'adv: {avg_adv_loss:f} '
                #   f'Ali: {avg_ali_loss:.3f} '
                #   f'Adv: {avg_adv_loss:.3f} '
                  f'Shared: {avg_shared_reg_loss:f} '
                f'rna_nb: {dist_params["rna_lambda_1"]:.3f}, '
                f'rna_ps: {dist_params["rna_lambda_2"]:.3f}, '
                f'atac_nb: {dist_params["atac_lambda_1"]:.3f}, '
                f'atac_bnl: {dist_params["atac_lambda_2"]:.3f}'

            )
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best loss: {best_loss:.4f}")
                break

def get_latent_representation(model, rna_adata, atac_adata, device='cuda', batch_size=256):
    """获取细胞的潜空间表示"""
    model.eval()
    dataset = MultiModalDataset(rna_adata, atac_adata)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    latent_reps = []
    with torch.no_grad():
        for rna_batch, atac_batch, _, _ in dataloader:
            rna_batch = rna_batch.to(device)
            atac_batch = atac_batch.to(device)
            
            _, _, fused_z, _, _, _ = model(rna_batch, atac_batch)
            latent_reps.append(fused_z.cpu().numpy())
    
    latent_rep = np.concatenate(latent_reps, axis=0)
    
    # 创建新的AnnData对象存储潜空间表示
    latent_adata = anndata.AnnData(X=latent_rep, obs=rna_adata.obs.copy())
    
    return latent_adata
import numpy as np

import numpy as np
import pandas as pd
import anndata as ad

import numpy as np

def artificial_unpair(rna_adata, atac_adata, unpair_ratio=0.75, random_seed=42):
    """
    对配对多组学数据集进行人工 unpairing 处理。

    参数:
        rna_adata (AnnData): RNA 组学数据
        atac_adata (AnnData): ATAC 组学数据
        unpair_ratio (float): 需要取消配对的比例 (0~1)
        random_seed (int): 随机种子，保证可复现

    返回:
        rna_adata_unpaired, atac_adata_unpaired: 处理后的两个 AnnData
    """
    np.random.seed(random_seed)
    
    # 获取 barcodes
    barcodes = rna_adata.obs_names
    N = len(barcodes)
    
    # 计算 unpair 数量
    num_unpair = int(N * unpair_ratio)
    
    # 随机选取 unpair 细胞
    unpair_indices = np.random.choice(N, num_unpair, replace=False)
    unpair_barcodes = barcodes[unpair_indices]
    
    # 拆分为一半 RNA-only, 一半 ATAC-only
    half = num_unpair // 2
    rna_only_barcodes = unpair_barcodes[:half]
    atac_only_barcodes = unpair_barcodes[half:]
    
    # 对 RNA-only 细胞，在 ATAC 中删除
    atac_adata_unpaired = atac_adata[~atac_adata.obs_names.isin(rna_only_barcodes)].copy()
    
    # 对 ATAC-only 细胞，在 RNA 中删除
    rna_adata_unpaired = rna_adata[~rna_adata.obs_names.isin(atac_only_barcodes)].copy()
    
    # 添加 paired_status 标签
    rna_adata_unpaired.obs['paired_status'] = 'paired'
    rna_adata_unpaired.obs.loc[rna_only_barcodes, 'paired_status'] = 'rna_only'
    
    atac_adata_unpaired.obs['paired_status'] = 'paired'
    atac_adata_unpaired.obs.loc[atac_only_barcodes, 'paired_status'] = 'atac_only'
    
    # 返回处理后的 adata
    return rna_adata_unpaired, atac_adata_unpaired


def main():
    global mu, sc, epi
    import muon as mu
    import scanpy as sc
    import episcanpy.api as epi
    import torch.optim as optim
    import pandas as pd
    import scipy.sparse as sp
    from anndata import AnnData
    # 设置随机种子以提高可重复性


    print(f"seed: {seed}")
    
    # 加载数据
    print("Loading data...")
    # adata = sc.read_h5ad("dataset/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad")

    # rna_adata = adata[:, adata.var['feature_types'] == 'GEX'].copy()
    # atac_adata = adata[:, adata.var['feature_types'] == 'ATAC'].copy()

    # mdata = mu.read("dataset/10XMultiomics_pbmc_10k_granulocyte_plus_1.h5mu")

    # rna_adata = mdata.mod['rna']
    # atac_adata = mdata.mod['atac']

    rna_adata = sc.read_h5ad("dataset/10xmultiomics_human_cortex_cortical_plate-RNA.h5ad")
    atac_adata = sc.read_h5ad("/data/hzb/dataset/10xmultiomics_human_cortex_cortical_plate-ATAC.h5ad")

    # 去掉所有细胞中都为 0 的特征（基因 / peaks）
    rna_adata = rna_adata[:, rna_adata.X.sum(axis=0).A1 > 0]
    atac_adata = atac_adata[:, atac_adata.X.sum(axis=0).A1 > 0]



    rna_adata.raw = rna_adata.copy()
    atac_adata.raw = atac_adata.copy()
    rna_dim = rna_adata.shape[1]
    atac_dim = atac_adata.shape[1]
    
    # 预处理
    print("Preprocessing RNA data...")
    sc.pp.normalize_total(rna_adata)
    sc.pp.log1p(rna_adata)
    sc.pp.highly_variable_genes(rna_adata, n_top_genes=3000)
    rna_adata = rna_adata[:, rna_adata.var.highly_variable].copy()
    
    print("Preprocessing ATAC data...")
    # epi.pp.binarize(atac_adata)
    epi.pp.filter_features(atac_adata, min_cells=np.ceil(atac_adata.shape[0] * 0.10))
    
    # TFIDF变换后，进行标准化确保数据在[0,1]范围内
    epi.pp.tfidf(atac_adata)
    
    # 确保ATAC数据在[0,1]范围内
    # atac_data = atac_adata.X
    # if scipy.sparse.issparse(atac_data):
    #     atac_data = atac_data.toarray()
    
    # max_val = np.max(atac_data)
    # print(f"ATAC data max value before normalization: {max_val}")
    
    # # 确保标准化到[0,1]
    # atac_adata.X = atac_data / max_val
    
    # 创建数据集和数据加载器
    dataset = MultiModalDataset(rna_adata, atac_adata)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 初始化模型
    rna_pro_dim = rna_adata.shape[1]
    atac_pro_dim = atac_adata.shape[1]
    print(f"RNA dimensions: {rna_dim}, ATAC dimensions: {atac_dim}")
    print(f"RNA Processing dimensions: {rna_pro_dim}, ATAC Processing dimensions: {atac_pro_dim}")
    model = scAPI(rna_dim, atac_dim, rna_pro_dim, atac_pro_dim, hidden_dim=512, latent_dim=32, dropout_rate=0.2)
    

    
 
    print(f"Using device: {device}")

    model.to(device)
    # 模型参数分组
    logit_params = [model.decoder.rna_logits, model.decoder.atac_logits]
    logit_param_ids = {id(p) for p in logit_params}
    main_params = [p for n, p in model.named_parameters() if id(p) not in logit_param_ids]
    # 定义优化器，降低学习率以提高稳定性
    optimizer = torch.optim.Adam([
        {"params": main_params},
        {"params": logit_params, "lr": 1e-4}  # logits 用更小的 lr
    ], lr=1e-3)
    
    # 添加学习率调度器

    
    # 训练模型
    try:
        train_model(model, train_loader, optimizer, num_epochs=2000, device=device)
        
        # 获取潜空间表示
        latent_adata = get_latent_representation(model, rna_adata, atac_adata, device)
        
        # 可视化潜空间
        print("Visualizing latent space...")
        sc.pp.neighbors(latent_adata)
        sc.tl.umap(latent_adata)
        if 'cell_type' in latent_adata.obs:
            sc.pl.umap(latent_adata, color='cell_type')
        
        # 保存模型和潜空间表示
        torch.save(model.state_dict(), f"scAPI/mdpt/human_cortex_cp_{seed}.pt")

        latent_adata.write(f'scAPI/mdpt/human_cortex_cp_{seed}.h5ad')
        print("Model and latent representation saved successfully!")
        
    except Exception as e:
        print("Error during training:", e)
        traceback.print_exc()  # 打印完整错误栈


if __name__ == "__main__":
    
    import opt
    seed = opt.args.seed
    setup_seed(seed)
    device = opt.args.device

    main()
