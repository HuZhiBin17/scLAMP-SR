import torch
import torch.nn as nn

import scAPI_shared_feature_lambda_continuous_prior_gentle_zi as gentle_zi


BASE_ANALYZE_RNA_FEATURE_WEIGHTS = gentle_zi.analyze_rna_feature_weights
CELL_RESIDUAL_RANK = 8
CELL_RESIDUAL_SCALE = 0.35
CELL_RESIDUAL_PENALTY = 0.50


class FeatureWiseCellResidualZIDecoder(gentle_zi.FeatureWiseMixtureDecoder):
    """ZINB/ZIP decoder with feature-level baseline plus bounded cell-level lambda residual.

    The base feature logit remains interpretable:
        a_g = continuous_prior_g + gene_residual_g

    A low-rank residual adds limited cell-state modulation:
        logit(lambda_ig) = a_g + scale * tanh(h(z_i) V_g)

    The residual is zero-initialized, so this variant starts exactly from the
    feature-level model and only uses cell-level modulation if it helps the
    likelihood enough to overcome regularization.
    """

    def __init__(
        self,
        rna_dim,
        atac_dim,
        rna_log_mean_expression,
        rna_zero_rate,
        hidden_dim=512,
        latent_dim=128,
        dropout_rate=0.2,
        residual_rank=8,
        cell_residual_scale=0.35,
        cell_residual_penalty=0.50,
    ):
        super().__init__(
            rna_dim,
            atac_dim,
            rna_log_mean_expression,
            rna_zero_rate,
            hidden_dim,
            latent_dim,
            dropout_rate,
        )
        self.residual_rank = residual_rank
        self.cell_residual_scale = cell_residual_scale
        self.cell_residual_penalty = cell_residual_penalty
        self.rna_cell_gate = nn.Sequential(
            nn.Linear(latent_dim, residual_rank),
            nn.Tanh(),
        )
        self.rna_cell_gene_factors = nn.Parameter(torch.zeros(residual_rank, rna_dim))

    def get_rna_base_logit(self):
        return self.get_rna_prior_logit() + self.rna_gene_residual

    def get_cell_residual(self, z):
        cell_scores = self.rna_cell_gate(z)
        raw_delta = torch.matmul(cell_scores, self.rna_cell_gene_factors)
        return self.cell_residual_scale * torch.tanh(raw_delta)

    def get_rna_logits(self, z=None):
        base_logit = self.get_rna_base_logit()
        if z is None:
            return torch.stack([base_logit, torch.zeros_like(base_logit)], dim=0)
        residual = self.get_cell_residual(z)
        rna_nb_logit = base_logit.unsqueeze(0) + residual
        return torch.stack([rna_nb_logit, torch.zeros_like(rna_nb_logit)], dim=0)

    def prior_regularization(self, z=None):
        base_penalty = super().prior_regularization()
        factor_penalty = self.rna_cell_gene_factors.pow(2).mean()
        gate_penalty = sum(param.pow(2).mean() for param in self.rna_cell_gate.parameters())
        penalty = base_penalty + 0.05 * factor_penalty + 0.01 * gate_penalty
        if z is not None:
            penalty = penalty + self.cell_residual_penalty * self.get_cell_residual(z).pow(2).mean()
        return penalty

    def forward(self, z, sample=True):
        rna_mean = self.rna_decoder_mean(z) + 1e-6
        rna_dispersion = self.rna_decoder_dispersion(z) + 1e-6
        rna_poisson = self.rna_decoder_poisson(z) + 1e-6
        rna_nb_zero_logits = self.rna_decoder_nb_zero_logits(z)
        rna_ps_zero_logits = self.rna_decoder_ps_zero_logits(z)

        atac_mean = self.atac_decoder_mean(z) + 1e-6
        atac_dispersion = self.atac_decoder_dispersion(z) + 1e-6
        atac_poisson = self.atac_decoder_poisson(z) + 1e-6
        atac_nb_zero_logits = self.atac_decoder_nb_zero_logits(z)
        atac_ps_zero_logits = self.atac_decoder_ps_zero_logits(z)

        rna_weights = torch.softmax(self.get_rna_logits(z), dim=0)
        atac_weights = torch.softmax(self.atac_logits, dim=0)
        rna_lambda_1, rna_lambda_2 = rna_weights[0], rna_weights[1]
        atac_lambda_1, atac_lambda_2 = atac_weights[0], atac_weights[1]

        rna_nb_dist = gentle_zi.negative_binomial_from_mean_dispersion(
            rna_mean,
            rna_dispersion,
        )
        rna_ps_dist = gentle_zi.Poisson(rate=rna_poisson)

        atac_nb_dist = gentle_zi.negative_binomial_from_mean_dispersion(
            atac_mean,
            atac_dispersion,
        )
        atac_ps_dist = gentle_zi.Poisson(rate=atac_poisson)

        if sample:
            rna_data = (
                rna_lambda_1 * gentle_zi.sample_zero_inflated(rna_nb_dist, rna_nb_zero_logits)
                + rna_lambda_2 * gentle_zi.sample_zero_inflated(rna_ps_dist, rna_ps_zero_logits)
            )
            atac_data = (
                atac_lambda_1.unsqueeze(0) * gentle_zi.sample_zero_inflated(atac_nb_dist, atac_nb_zero_logits)
                + atac_lambda_2.unsqueeze(0) * gentle_zi.sample_zero_inflated(atac_ps_dist, atac_ps_zero_logits)
            )
        else:
            rna_first_mean = (1.0 - torch.sigmoid(rna_nb_zero_logits)) * rna_mean
            rna_second_mean = (1.0 - torch.sigmoid(rna_ps_zero_logits)) * rna_poisson
            rna_data = rna_lambda_1 * rna_first_mean + rna_lambda_2 * rna_second_mean
            atac_data = atac_lambda_1.unsqueeze(0) * atac_mean + atac_lambda_2.unsqueeze(0) * atac_poisson

        outputs = {
            "rna_mean": rna_mean,
            "rna_dispersion": rna_dispersion,
            "rna_poisson": rna_poisson,
            "rna_nb_dist": rna_nb_dist,
            "rna_ps_dist": rna_ps_dist,
            "rna_nb_zero_logits": rna_nb_zero_logits,
            "rna_ps_zero_logits": rna_ps_zero_logits,
            "atac_mean": atac_mean,
            "atac_dispersion": atac_dispersion,
            "atac_bnl": atac_poisson,
            "atac_nb_dist": atac_nb_dist,
            "atac_bnl_dist": atac_ps_dist,
            "atac_nb_zero_logits": atac_nb_zero_logits,
            "atac_ps_zero_logits": atac_ps_zero_logits,
            "rna_lambda_1": rna_lambda_1,
            "rna_lambda_2": rna_lambda_2,
            "atac_lambda_1": atac_lambda_1,
            "atac_lambda_2": atac_lambda_2,
            "rna_nb_zero_rate": torch.sigmoid(rna_nb_zero_logits),
            "rna_ps_zero_rate": torch.sigmoid(rna_ps_zero_logits),
            "atac_nb_zero_rate": torch.sigmoid(atac_nb_zero_logits),
            "atac_ps_zero_rate": torch.sigmoid(atac_ps_zero_logits),
            "mixture_prior_reg": self.prior_regularization(z),
            "rna_prior_nb_weight": self.get_rna_prior_nb_weight(),
            "rna_prior_strength": self.rna_prior_strength,
            "rna_cell_residual": self.get_cell_residual(z),
        }
        return rna_data, atac_data, outputs


class scAPIFeatureLambda(gentle_zi.scAPIFeatureLambda):
    def __init__(
        self,
        rna_dim,
        atac_dim,
        rna_pro_dim,
        atac_pro_dim,
        rna_log_mean_expression,
        rna_zero_rate,
        hidden_dim=512,
        latent_dim=128,
        dropout_rate=0.2,
    ):
        super().__init__(
            rna_dim,
            atac_dim,
            rna_pro_dim,
            atac_pro_dim,
            rna_log_mean_expression,
            rna_zero_rate,
            hidden_dim,
            latent_dim,
            dropout_rate,
        )
        self.decoder = FeatureWiseCellResidualZIDecoder(
            rna_dim,
            atac_dim,
            rna_log_mean_expression,
            rna_zero_rate,
            hidden_dim,
            latent_dim,
            dropout_rate,
            residual_rank=CELL_RESIDUAL_RANK,
            cell_residual_scale=CELL_RESIDUAL_SCALE,
            cell_residual_penalty=CELL_RESIDUAL_PENALTY,
        )


def log_weight_for_likelihood(weight):
    if weight.dim() == 1:
        return torch.log(weight.unsqueeze(0) + 1e-8)
    return torch.log(weight + 1e-8)


def true_mixture_loss_function(
    rna_data,
    atac_data,
    dist_params,
    rna_z,
    atac_z,
    z_shared=None,
    gamma=0.1,
    alpha=1.0,
    beta=0.5,
    alignment_type=None,
    prior_reg_weight=2e-3,
):
    del alignment_type
    rna_data = rna_data.round().long()
    atac_data = atac_data.round().long()

    rna_nb_logprob = gentle_zi.zero_inflated_log_prob(
        dist_params["rna_nb_dist"],
        rna_data,
        dist_params["rna_nb_zero_logits"],
    )
    rna_ps_logprob = gentle_zi.zero_inflated_log_prob(
        dist_params["rna_ps_dist"],
        rna_data,
        dist_params["rna_ps_zero_logits"],
    )
    atac_nb_logprob = gentle_zi.zero_inflated_log_prob(
        dist_params["atac_nb_dist"],
        atac_data,
        dist_params["atac_nb_zero_logits"],
    )
    atac_ps_logprob = gentle_zi.zero_inflated_log_prob(
        dist_params["atac_bnl_dist"],
        atac_data,
        dist_params["atac_ps_zero_logits"],
    )

    rna_log_lambda_1 = log_weight_for_likelihood(dist_params["rna_lambda_1"])
    rna_log_lambda_2 = log_weight_for_likelihood(dist_params["rna_lambda_2"])
    atac_log_lambda_1 = log_weight_for_likelihood(dist_params["atac_lambda_1"])
    atac_log_lambda_2 = log_weight_for_likelihood(dist_params["atac_lambda_2"])

    rna_log_likelihood = torch.logsumexp(
        torch.stack([rna_log_lambda_1 + rna_nb_logprob, rna_log_lambda_2 + rna_ps_logprob], dim=0),
        dim=0,
    )
    atac_log_likelihood = torch.logsumexp(
        torch.stack([atac_log_lambda_1 + atac_nb_logprob, atac_log_lambda_2 + atac_ps_logprob], dim=0),
        dim=0,
    )

    rna_recon_loss = -rna_log_likelihood.mean()
    atac_recon_loss = -atac_log_likelihood.mean()
    kl_loss = gentle_zi.compute_kld(
        dist_params["shared_mu"],
        dist_params["shared_var"],
        dist_params["rna_mu"],
        dist_params["rna_var"],
        dist_params["atac_mu"],
        dist_params["atac_var"],
    ) * alpha

    shared_reg_loss = 0
    if z_shared is not None:
        shared_rna_loss = gentle_zi.compute_mmd(z_shared, rna_z) * beta
        shared_atac_loss = gentle_zi.compute_mmd(z_shared, atac_z) * beta
        shared_reg_loss = shared_rna_loss + shared_atac_loss

    adv_loss = dist_params["adv_loss"] * gamma
    prior_reg_loss = dist_params.get("mixture_prior_reg", torch.tensor(0.0, device=rna_recon_loss.device)) * prior_reg_weight
    total_loss = rna_recon_loss + atac_recon_loss + shared_reg_loss + kl_loss + adv_loss + prior_reg_loss
    return {
        "total_loss": total_loss,
        "rna_recon_loss": rna_recon_loss,
        "atac_recon_loss": atac_recon_loss,
        "kl_loss": kl_loss,
        "adv_loss": adv_loss,
        "shared_reg_loss": shared_reg_loss,
        "prior_reg_loss": prior_reg_loss,
    }


def analyze_rna_feature_weights(model, rna_adata, output_dir):
    BASE_ANALYZE_RNA_FEATURE_WEIGHTS(model, rna_adata, output_dir)


def install_cell_residual_variant():
    gentle_zi.scAPIFeatureLambda = scAPIFeatureLambda
    gentle_zi.true_mixture_loss_function = true_mixture_loss_function
    gentle_zi.analyze_rna_feature_weights = analyze_rna_feature_weights


def parse_args():
    args = gentle_zi.parse_args()
    setattr(args, "variant", "continuous_prior_gentle_zi_cell_residual")
    return args


def main(args):
    global CELL_RESIDUAL_RANK, CELL_RESIDUAL_SCALE, CELL_RESIDUAL_PENALTY
    CELL_RESIDUAL_RANK = args.cell_residual_rank
    CELL_RESIDUAL_SCALE = args.cell_residual_scale
    CELL_RESIDUAL_PENALTY = args.cell_residual_penalty
    install_cell_residual_variant()
    gentle_zi.main(args)


if __name__ == "__main__":
    main(parse_args())
