import stribor as st
import torch


# input_dim = (5, 3, 2, 3)
# hidden_dims = [47, 11]
# out_dim = 7
# pooling = 'mean'

# # dim = input_dim[-1]
# # model = st.net.DiffeqZeroTraceDeepSet(dim, hidden_dims, dim * out_dim, pooling, return_log_det_jac=False)
# # x = torch.randn(*input_dim).requires_grad_(True)

# # t = torch.Tensor([1])
# # f = lambda x: model(t, x)
# # trace_exact = st.util.divergence_from_jacobian(f, x)

# # assert trace_exact.sum() == 0
# # assert torch.allclose(trace_exact, torch.zeros_like(trace_exact), atol=1e-6)

# dim = input_dim[-1]

# base_dist = st.UnitNormal(dim)

# transforms = [
#     # st.Coupling(
#     #     transform=st.Affine(dim, latent_net=st.net.MLP(dim, [64], 2 * dim)),
#     #     mask='ordered_right_half',
#     # ),
#     st.ContinuousTransform(
#         dim,
#         net=st.net.DiffeqZeroTraceDeepSet(dim, hidden_dims, dim * out_dim, pooling),
#     )
# ]

# flow = st.NormalizingFlow(base_dist, transforms)


# x = torch.randn(*input_dim).requires_grad_(True)
# y = flow(x) # Forward transformation
# log_prob = flow.log_prob(y) # Log-probability p(y)
# print(y, log_prob)


# Adapted from
# https://github.com/smsharma/jet-setting/blob/1c07c72f3354936093589f66547d2face89036f3/notebooks/01_jets_set_transformer.ipynb#L129

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td


class Flow(nn.Module):
    """
    Building both normalizing flows and neural flows.

    Example:
    >>> import stribor as st
    >>> torch.manual_seed(123)
    >>> dim = 2
    >>> flow = st.Flow(st.UnitNormal(dim), [st.Affine(dim)])
    >>> x = torch.rand(1, dim)
    >>> y, ljd = flow(x)
    >>> y_inv, ljd_inv = flow.inverse(y)

    Args:
        base_dist (Type[torch.distributions]): Base distribution
        transforms (List[st.flows]): List of invertible transformations
    """

    def __init__(self, base_dist=None, transforms=[]):
        super().__init__()
        self.base_dist = base_dist
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x, latent=None, mask=None, t=None, reverse=False, **kwargs):
        """
        Args:
            x (tensor): Input sampled from base density with shape (..., dim)
            latent (tensor, optional): Conditional vector with shape (..., latent_dim)
                Default: None
            mask (tensor): Masking tensor with shape (..., 1)
                Default: None
            t (tensor, optional): Flow time end point. Default: None
            reverse (bool, optional): Whether to perform an inverse. Default: False

        Returns:
            y (tensor): Output that follows target density (..., dim)
            log_jac_diag (tensor): Log-Jacobian diagonal (..., dim)
        """
        transforms = self.transforms[::-1] if reverse else self.transforms
        _mask = 1 if mask is None else mask

        log_jac_diag = torch.zeros_like(x).to(x)
        for f in transforms:
            if reverse:
                x, ld = f.inverse_and_log_det_jacobian(
                    x * _mask, latent=latent, mask=mask, t=t, **kwargs
                )
            else:
                x, ld = f.forward_and_log_det_jacobian(
                    x * _mask, latent=latent, mask=mask, t=t, **kwargs
                )
            log_jac_diag += ld * _mask
        return x, log_jac_diag

    def inverse(self, y, latent=None, mask=None, t=None, **kwargs):
        """Inverse of forward function with the same arguments."""
        return self.forward(y, latent=latent, mask=mask, t=t, reverse=True, **kwargs)

    def log_prob(self, x, **kwargs):
        """
        Calculates log-probability of a sample.

        Args:
            x (tensor): Input with shape (..., dim)

        Returns:
            log_prob (tensor): Log-probability of the input with shape (..., 1)
        """
        if self.base_dist is None:
            raise ValueError("Please define `base_dist` if you need log-probability")
        x, log_jac_diag = self.inverse(x, **kwargs)

        log_prob = self.base_dist.log_prob(x) + log_jac_diag.sum(-1)
        return log_prob.unsqueeze(-1)

    def sample(self, num_samples, latent=None, mask=None, **kwargs):
        """
        Transforms samples from the base to the target distribution.
        Uses reparametrization trick.

        Args:
            num_samples (tuple or int): Shape of samples
            latent (tensor): Latent conditioning vector with shape (..., latent_dim)

        Returns:
            x (tensor): Samples from target distribution with shape (*num_samples, dim)
        """

        if self.base_dist is None:
            raise ValueError("Please define `base_dist` if you need sampling")
        if isinstance(num_samples, int):
            num_samples = (num_samples,)

        x = self.base_dist.rsample(num_samples)
        x, log_jac_diag = self.forward(x, latent, mask, **kwargs)
        return x


def get_exact_model(
    dim,
    hidden_dims,
    latent_dim,
    context_dim=0,
    n_transforms=4,
    n_heads=2,
    model="deepset",
    set_data=False,
    device="cpu",
    atol=1e-4,
    base_dist_mean=None,
    base_dist_cov=None,
):
    has_latent = True if context_dim > 0 else False

    transforms = []

    for _ in range(n_transforms):
        if model == "deepset":
            net = st.net.DiffeqExactTraceDeepSet(
                dim, hidden_dims, dim, d_h=latent_dim, latent_dim=context_dim
            )
        elif model == "settransformer":
            net = st.net.DiffeqExactTraceAttention(
                dim,
                hidden_dims,
                dim,
                d_h=latent_dim,
                n_heads=n_heads,
                latent_dim=context_dim,
            )
        else:
            raise NotImplementedError

        transforms.append(
            st.flows.ContinuousTransform(
                dim,
                net=net,
                divergence="exact",
                solver="dopri5",
                atol=atol,
                has_latent=has_latent,
                set_data=set_data,
            )
        )

    if base_dist_mean is None:
        base_dist_mean = torch.zeros(dim)

    if base_dist_cov is None:
        base_dist_cov = torch.ones(dim)

    model = Flow(
        st.Normal(base_dist_mean.to(device), base_dist_cov.to(device)), transforms
    ).to(device)

    return model


network = get_exact_model(
    dim=3,
    hidden_dims=[64, 64],
    latent_dim=8,
    context_dim=0,
    n_transforms=2,
    n_heads=2,
    model="deepset",
    set_data=True,
    # base_dist_mean=x_mean,
    # base_dist_cov=x_cov,
    # device=device,
    atol=1e-4,
)

x = torch.randn(10, 10, 3)
print(network.log_prob(x, mask=torch.ones_like(x)), network.sample(10))
