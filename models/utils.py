import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import kneighbors_graph
import numpy as np
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)   
        
def compute_knn_graph(embeddings, k):
    """计算k近邻图"""
    embeddings_np = embeddings.detach().cpu().numpy()
    knn_graph = kneighbors_graph(embeddings_np, n_neighbors=k, 
                                mode='connectivity', include_self=False)
    return torch.from_numpy(knn_graph.toarray()).float().to(embeddings.device)
    
def neighborhood_preservation_loss(z_shared, z_rna, z_atac, k=10):

    knn_shared = compute_knn_graph(z_shared, k)
    knn_rna = compute_knn_graph(z_rna, k)
    knn_atac = compute_knn_graph(z_atac, k)
    
    # 计算近邻关系的一致性
    rna_consistency = F.mse_loss(knn_shared, knn_rna)
    atac_consistency = F.mse_loss(knn_shared, knn_atac)
    
    return rna_consistency + atac_consistency

def contrastive_alignment_loss(self, z_shared, z_rna, z_atac, temperature=0.1):
    """对比学习对齐损失"""
    batch_size = z_shared.size(0)
    
    # 计算相似度矩阵
    sim_shared_rna = torch.mm(z_shared, z_rna.t()) / temperature
    sim_shared_atac = torch.mm(z_shared, z_atac.t()) / temperature
    
    # 正样本对角线，负样本非对角线
    labels = torch.arange(batch_size).to(z_shared.device)
    
    loss_rna = F.cross_entropy(sim_shared_rna, labels)
    loss_atac = F.cross_entropy(sim_shared_atac, labels)
    
    return loss_rna + loss_atac

def triplet_loss(self, z_shared, z_rna, z_atac, margin=1.0):
    """三元组损失，拉近同一样本的不同模态表示"""
    # 计算距离
    dist_shared_rna = F.pairwise_distance(z_shared, z_rna)
    dist_shared_atac = F.pairwise_distance(z_shared, z_atac)
    
    # 负样本：随机打乱
    indices = torch.randperm(z_rna.size(0))
    dist_shared_rna_neg = F.pairwise_distance(z_shared, z_rna[indices])
    dist_shared_atac_neg = F.pairwise_distance(z_shared, z_atac[indices])
    
    # 三元组损失
    loss_rna = F.relu(dist_shared_rna - dist_shared_rna_neg + margin)
    loss_atac = F.relu(dist_shared_atac - dist_shared_atac_neg + margin)
    
    return loss_rna.mean() + loss_atac.mean()

def wasserstein_distance_loss(self, z_shared, z_rna, z_atac):
    """近似Wasserstein距离损失"""
    def compute_wasserstein_approx(x, y):
        # 简化的Wasserstein距离近似
        x_sorted, _ = torch.sort(x, dim=0)
        y_sorted, _ = torch.sort(y, dim=0)
        return torch.mean((x_sorted - y_sorted) ** 2)
    
    loss_rna = compute_wasserstein_approx(z_shared, z_rna)
    loss_atac = compute_wasserstein_approx(z_shared, z_atac)
    
    return loss_rna + loss_atac

class CCAAlignment(nn.Module):
    """典型相关分析对齐"""
    
    def __init__(self, latent_dim, reg_param=1e-3):
        super().__init__()
        self.reg_param = reg_param
        self.latent_dim = latent_dim
        
    def forward(self, z_shared, z_rna, z_atac):
        """计算CCA损失"""
        batch_size = z_shared.size(0)
        
        # 中心化
        z_shared_centered = z_shared - z_shared.mean(dim=0)
        z_rna_centered = z_rna - z_rna.mean(dim=0)
        z_atac_centered = z_atac - z_atac.mean(dim=0)
        
        # 计算协方差矩阵
        cov_shared_rna = torch.mm(z_shared_centered.t(), z_rna_centered) / (batch_size - 1)
        cov_shared_atac = torch.mm(z_shared_centered.t(), z_atac_centered) / (batch_size - 1)
        
        # CCA损失：最大化相关性
        loss = -torch.trace(cov_shared_rna) - torch.trace(cov_shared_atac)
        
        return loss

class MMDAlignment(nn.Module):
    """最大均值差异对齐"""
    
    def __init__(self, kernel_type='rbf', kernel_params=None):
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_params = kernel_params or {'gamma': 1.0}
        
    def rbf_kernel(self, x, y, gamma=1.0):
        """RBF核函数"""
        x_norm = (x ** 2).sum(dim=1, keepdim=True)
        y_norm = (y ** 2).sum(dim=1, keepdim=True)
        
        dist = x_norm + y_norm.t() - 2 * torch.mm(x, y.t())
        return torch.exp(-gamma * dist)
    
    def mmd_loss(self, x, y):
        """计算MMD损失"""
        gamma = self.kernel_params['gamma']
        
        k_xx = self.rbf_kernel(x, x, gamma)
        k_yy = self.rbf_kernel(y, y, gamma)
        k_xy = self.rbf_kernel(x, y, gamma)
        
        m = k_xx.size(0)
        n = k_yy.size(0)
        
        mmd = (k_xx.sum() / (m * m) + k_yy.sum() / (n * n) - 
               2 * k_xy.sum() / (m * n))
        
        return mmd
    
    def forward(self, z_shared, z_rna, z_atac):
        """计算MMD对齐损失"""
        loss_rna = self.mmd_loss(z_shared, z_rna)
        loss_atac = self.mmd_loss(z_shared, z_atac)
        
        return loss_rna + loss_atac
    
from torch.autograd import Function
class GradientReversalFunction(Function):
    """梯度反转函数"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class GradientReversalLayer(nn.Module):
    """梯度反转层"""
    def __init__(self, lambda_=1.0):
        super().__init__()
        self.lambda_ = lambda_
    
    def set_lambda(self, lambda_):
        self.lambda_ = lambda_
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class ModalityDiscriminator(nn.Module):
    """模态判别器"""
    def __init__(self, latent_dim, hidden_dims=[256, 128], dropout_rate=0.3, use_spectral_norm=False):
        super().__init__()
        
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            
            layers.extend([
                linear,
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # 输出层：3类分类 (shared, rna, atac)
        output_layer = nn.Linear(prev_dim, 3)
        if use_spectral_norm:
            output_layer = nn.utils.spectral_norm(output_layer)
        layers.append(output_layer)
        
        self.discriminator = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.discriminator(x)

class AdversarialAlignmentModule(nn.Module):
    """对抗学习对齐模块"""
    
    def __init__(self, 
                 latent_dim, 
                 hidden_dims=[256, 128], 
                 dropout_rate=0.3, 
                 lambda_grl=1.0,
                 use_spectral_norm=True,
                 loss_type='cross_entropy',
                 discriminator_lr=1e-4):
        """
        Args:
            latent_dim: 潜空间维度
            hidden_dims: 判别器隐藏层维度列表
            dropout_rate: dropout比率
            lambda_grl: 梯度反转层的权重
            use_spectral_norm: 是否使用谱归一化（提高稳定性）
            loss_type: 损失类型 ['cross_entropy', 'wasserstein', 'least_squares']
            discriminator_lr: 判别器学习率
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        
        # 梯度反转层
        self.grl = GradientReversalLayer(lambda_grl)
        
        # 模态判别器
        self.discriminator = ModalityDiscriminator(
            latent_dim, hidden_dims, dropout_rate, use_spectral_norm
        )
        
        # 判别器优化器
        self.disc_optimizer = torch.optim.Adam(
            self.discriminator.parameters(), 
            lr=discriminator_lr, betas=(0.5, 0.999)
        )
        
        # 模态标签映射
        self.modality_labels = {
            'shared': 0,
            'rna': 1, 
            'atac': 2
        }
        
        # Lambda调度函数
        self.lambda_schedule = self._get_lambda_schedule()
        
    def _get_lambda_schedule(self):
        """梯度反转权重调度"""
        def schedule(epoch, max_epochs=1000):
            # 渐进式增加权重
            p = epoch / max_epochs
            return 2.0 / (1.0 + torch.exp(-10 * p)) - 1.0
        return schedule
        
    def set_lambda(self, lambda_):
        """动态调整梯度反转权重（用于curriculum learning）"""
        self.grl.set_lambda(lambda_)
    
    def update_lambda(self, epoch, max_epochs=1000):
        """更新梯度反转权重"""
        new_lambda = self.lambda_schedule(epoch, max_epochs)
        self.grl.set_lambda(new_lambda)
        return new_lambda
    
    def create_labels(self, batch_size, device):
        """创建模态标签"""
        labels = torch.cat([
            torch.full((batch_size,), self.modality_labels['shared'], dtype=torch.long),
            torch.full((batch_size,), self.modality_labels['rna'], dtype=torch.long),
            torch.full((batch_size,), self.modality_labels['atac'], dtype=torch.long)
        ]).to(device)
        return labels
    
    def discriminator_loss(self, z_shared, z_rna, z_atac):
        """计算判别器损失（用于单独训练判别器）"""
        batch_size = z_shared.size(0)
        device = z_shared.device
        
        # 合并所有表示
        all_representations = torch.cat([z_shared, z_rna, z_atac], dim=0)
        
        # 创建真实标签
        true_labels = self.create_labels(batch_size, device)
        
        # 判别器预测（不通过GRL）
        pred_logits = self.discriminator(all_representations)
        
        if self.loss_type == 'cross_entropy':
            disc_loss = F.cross_entropy(pred_logits, true_labels)
        elif self.loss_type == 'least_squares':
            # LSGAN损失
            one_hot_labels = F.one_hot(true_labels, num_classes=3).float()
            disc_loss = F.mse_loss(torch.sigmoid(pred_logits), one_hot_labels)
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")
        
        # 计算准确率
        pred_labels = torch.argmax(pred_logits, dim=1)
        accuracy = (pred_labels == true_labels).float().mean()
        
        return disc_loss, accuracy, pred_logits
    
    def train_discriminator_step(self, z_shared, z_rna, z_atac):
        """单独训练判别器一步 - 现在集成到模块中"""
        self.disc_optimizer.zero_grad()
        
        disc_loss, disc_acc, _ = self.discriminator_loss(z_shared, z_rna, z_atac)
        disc_loss.backward()
        self.disc_optimizer.step()
        
        return disc_loss.item(), disc_acc.item()
    
    def adversarial_loss(self, z_shared, z_rna, z_atac):
        """计算对抗损失（用于编码器训练）"""
        batch_size = z_shared.size(0)
        device = z_shared.device
        
        # 合并所有表示并通过梯度反转层
        all_representations = torch.cat([z_shared, z_rna, z_atac], dim=0)
        reversed_representations = self.grl(all_representations)
        
        # 判别器预测
        pred_logits = self.discriminator(reversed_representations)
        
        # 创建标签
        true_labels = self.create_labels(batch_size, device)
        
        if self.loss_type == 'cross_entropy':
            # 标准对抗损失（希望判别器分类错误）
            adv_loss = F.cross_entropy(pred_logits, true_labels)
            
        elif self.loss_type == 'least_squares':
            # LSGAN对抗损失
            one_hot_labels = F.one_hot(true_labels, num_classes=3).float()
            adv_loss = F.mse_loss(torch.sigmoid(pred_logits), one_hot_labels)
            
        else:
            raise NotImplementedError(f"Loss type {self.loss_type} not implemented")
        
        # 计算额外的对齐指标
        pred_probs = F.softmax(pred_logits, dim=1)
        
        # 混乱度（越高越好）：理想情况下每类概率都是1/3
        uniform_target = torch.ones_like(pred_probs) / 3.0
        confusion_score = -F.kl_div(pred_probs.log(), uniform_target, reduction='batchmean')
        
        # 最大预测概率（越低越好）：表示判别器越不确定
        max_prob = pred_probs.max(dim=1)[0].mean()
        
        return {
            'adversarial_loss': adv_loss,
            'confusion_score': confusion_score,
            'max_probability': max_prob,
            'pred_logits': pred_logits
        }
    
    def alignment_metrics(self, z_shared, z_rna, z_atac):
        """计算对齐度量指标"""
        with torch.no_grad():
            # 计算表示间的距离
            dist_shared_rna = F.pairwise_distance(z_shared, z_rna).mean()
            dist_shared_atac = F.pairwise_distance(z_shared, z_atac).mean()
            dist_rna_atac = F.pairwise_distance(z_rna, z_atac).mean()
            
            # 计算中心距离
            center_shared = z_shared.mean(dim=0)
            center_rna = z_rna.mean(dim=0)
            center_atac = z_atac.mean(dim=0)
            
            center_dist_shared_rna = F.pairwise_distance(
                center_shared.unsqueeze(0), center_rna.unsqueeze(0)
            ).item()
            center_dist_shared_atac = F.pairwise_distance(
                center_shared.unsqueeze(0), center_atac.unsqueeze(0)
            ).item()
            
            # 计算方差比较
            var_shared = z_shared.var(dim=0).mean().item()
            var_rna = z_rna.var(dim=0).mean().item()
            var_atac = z_atac.var(dim=0).mean().item()
            
            return {
                'pairwise_dist_shared_rna': dist_shared_rna.item(),
                'pairwise_dist_shared_atac': dist_shared_atac.item(),
                'pairwise_dist_rna_atac': dist_rna_atac.item(),
                'center_dist_shared_rna': center_dist_shared_rna,
                'center_dist_shared_atac': center_dist_shared_atac,
                'variance_shared': var_shared,
                'variance_rna': var_rna,
                'variance_atac': var_atac
            }
    
    def forward(self, z_shared, z_rna, z_atac, mode='adversarial'):
        """
        前向传播
        Args:
            z_shared, z_rna, z_atac: 三种模态的潜空间表示
            mode: 'adversarial' 或 'discriminator'
        """
        if mode == 'adversarial':
            return self.adversarial_loss(z_shared, z_rna, z_atac)
        elif mode == 'discriminator':
            return self.discriminator_loss(z_shared, z_rna, z_atac)
        else:
            raise ValueError(f"Unknown mode: {mode}")
