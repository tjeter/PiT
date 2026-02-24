import torch
import torch.nn as nn
from math import sqrt
import numpy as np


class CausalMask:
    def __init__(self, B, L, device):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool), diagonal=1
            ).to(device)

    @property
    def mask(self):
        return self._mask


class PhaseSyncAttention(nn.Module):
    def __init__(
        self,
        win_size,
        mask_flag=True,
        scale=None,
        attention_dropout=0.0,
        output_attention=False,
        gamma=2.0,
        sigma=1.0,
        lambda_smooth=0.01,
        device="cuda",
    ):
        super(PhaseSyncAttention, self).__init__()
        self.device = torch.device(device)
        self.win_size = win_size
        self.mask_flag = mask_flag
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.output_attention = output_attention
        self.gamma = gamma
        self.sigma = sigma
        self.lambda_smooth = lambda_smooth

        indices = torch.arange(win_size, device=self.device)

        ''' This value is never used ''' # Not a bug, just wasteful
        # self.distances = torch.abs(indices.unsqueeze(1) - indices.unsqueeze(0))

    def compute_fractal_abscissa(self, hurst):
        hurst = torch.clamp(hurst, min=0.01, max=1.0)
        exp_hurst = torch.exp(self.gamma * hurst)
        psi = torch.cumsum(exp_hurst, dim=1)
        return psi

    def compute_gaussian_prior(self, psi):
        B, L, H = psi.shape

        psi_i = psi.unsqueeze(2)  # [B, L, 1, H]
        psi_j = psi.unsqueeze(1)  # [B, 1, L, H]
        diff = psi_i - psi_j  # [B, L, L, H]

        prior = torch.exp(-(diff**2) / (2 * self.sigma**2))
        prior = prior.permute(0, 3, 1, 2)  # [B, H, L, L]
        prior_sum = prior.sum(dim=-1, keepdim=True)
        prior = prior / (prior_sum + 1e-6)

        return prior

    def _hilbert_transform(self, signal):
        B, C, L = signal.shape
        h = torch.zeros(L, device=signal.device, dtype=signal.dtype)
        if L % 2 == 0:
            h[0] = h[L // 2] = 1.0
            h[1 : L // 2] = 2.0
        else:
            h[0] = 1.0
            h[1 : (L + 1) // 2] = 2.0

        return torch.fft.ifft(torch.fft.fft(signal, dim=-1) * h.view(1, 1, L), dim=-1)

    def compute_phase_gate(self, x, tau):
        B, L, D = x.shape
        analytic = self._hilbert_transform(x.permute(0, 2, 1))

        real_part = torch.real(analytic)
        imag_part = torch.imag(analytic)

        phase = torch.atan2(imag_part, real_part)
        theta = torch.atan2(torch.sin(phase).mean(1), torch.cos(phase).mean(1))

        dphi = torch.sin(0.5 * (theta.unsqueeze(2) - theta.unsqueeze(1)))
        tau = tau.unsqueeze(2).expand(-1, -1, L, -1)  # [B, L, L, H]
        S = torch.exp(-(dphi.unsqueeze(-1) ** 2) / (2 * tau**2))

        return S.permute(0, 3, 1, 2)  # [B, H, L, L]

    def compute_tau_smoothness_loss(self, tau):
        diff = tau[:, 1:, :] - tau[:, :-1, :]
        loss = self.lambda_smooth * torch.mean(diff**2)

        return loss

    def compute_hurst_smoothness_loss(self, hurst):
        diff = hurst[:, 1:, :] - hurst[:, :-1, :]
        loss = self.lambda_smooth * torch.mean(diff**2)

        return loss

    def compute_beta_prior_loss(self, hurst):
        alpha, beta = 2.0, 2.0
        hurst_clamped = torch.clamp(hurst, min=1e-6, max=1.0 - 1e-6)
        log_beta = (alpha - 1) * torch.log(hurst_clamped) + (beta - 1) * torch.log(
            1 - hurst_clamped
        )
        loss = -torch.mean(log_beta)

        return loss

    def forward(self, q, k, v, sigma, hurst, tau, attn_mask):
        B, L, H, E = q.shape
        _, S, _, D = v.shape

        scale = self.scale or 1.0 / sqrt(E)

        scores = torch.einsum("blhe, bshe->bhls", q, k)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = CausalMask(B, L, device=q.device)
            scores.masked_fill(attn_mask.mask, -np.inf)
        attn = scale * scores


        hurst = torch.sigmoid(hurst) * 0.99 + 0.01
        sigma = torch.sigmoid(sigma * 5) + 1e-6
        tau = torch.sigmoid(tau) * 0.9 + 0.1

        psi = self.compute_fractal_abscissa(hurst)
        prior = self.compute_gaussian_prior(psi)

        # Hopeful bug fix
        x_flat = q.permute(0, 2, 1, 3).reshape(B, L, -1)
        gate = self.compute_phase_gate(x_flat, tau.detach())

        '''Above phase structure shapes attention behavior.
           Tau remains stable and experiences no feedback explosion.
           Encourages represation-prior alignment.

           Below the phase gate depends on a signal.
           Detaching means phase synchrony does not influence representation learning and only tau adapts.
           This is contradicting to "attention should respect phase structure".
           This version allows the model to violate phase synchrony with no penalty.
        '''
        # x_flat = q.permute(0, 2, 1, 3).reshape(B, L, -1).detach()
        # gate = self.compute_phase_gate(x_flat, tau)
        prior = prior * gate
        prior = prior / (prior.sum(dim=-1, keepdim=True) + 1e-6)


        smoothness_loss = self.compute_hurst_smoothness_loss(hurst)
        tau_smoothness_loss = self.compute_tau_smoothness_loss(tau)
        beta_prior_loss = self.compute_beta_prior_loss(hurst)

        series = self.dropout(torch.softmax(attn, dim=-1))
        # V = torch.einsum("bhls,bshd->blhd", series, v)
        ''' Above the prior never influences value aggregation.
            Prior is only used for KL loss. Attention output ignores physical structure.
            The model can learn pathological attentions and get rid of them.
            This weakens physical inductive bias and robustness to distribution shift.

            Below the prior minimally regularizes the attention behavior.
            It doesn't dominate data-driven learning.
            Stability and interpretability should be improved.
            KL-loss is still valid for anomaly scoring.
            The prior is moved from a judge view to a soft guide view.
        '''
        # Hopeful bug fix
        alpha = 0.1 # small for now, can be tuned
        attn_combined = (1 - alpha) * series + alpha * prior
        V = torch.einsum("bhls,bshd->blhd", attn_combined, v)

        if self.output_attention:
            return (
                V.contiguous(),
                series,
                prior,
                sigma,
                hurst,
                smoothness_loss,
                beta_prior_loss,
                tau_smoothness_loss,
            )
        else:
            return (V.contiguous(), None)

# From the original PyKan code: https://github.com/KindXiaoming/pykan/
def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases
    
    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde
    
    Returns:
    --------
        spline values : 3D torch.tensor
            shape (batch, in_dim, G+k). G: the number of grid intervals, k: spline order.
      
    Example
    -------
    >>> from kan.spline import B_batch
    >>> x = torch.rand(100,2)
    >>> grid = torch.linspace(-1,1,steps=11)[None, :].expand(2, 11)
    >>> B_batch(x, grid, k=3).shape
    '''
    
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:,:,0], grid=grid[0], k=k - 1)
        
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
                    grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    
    # in case grid is degenerate
    value = torch.nan_to_num(value)
    return value

def coef2curve(x_eval, grid, coef, k, device="cpu"):
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).
    
    Args:
    -----
        x_eval : 2D torch.tensor
            shape (batch, in_dim)
        grid : 2D torch.tensor
            shape (in_dim, G+2k). G: the number of grid intervals; k: spline order.
        coef : 3D torch.tensor
            shape (in_dim, out_dim, G+k)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde
        
    Returns:
    --------
        y_eval : 3D torch.tensor
            shape (batch, in_dim, out_dim)
        
    '''
    
    b_splines = B_batch(x_eval, grid, k=k)
    y_eval = torch.einsum('ijk,jlk->ijl', b_splines, coef.to(b_splines.device))
    
    return y_eval

class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_knots=16, spline_order=1, grid_range(-3.0, 3.0)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.k = spline_order

        grid = torch.linspace(grid_range[0], grid_range[1], num_knots + 1)[None, :].expand(in_dim, -1)
        self.register_buffer("grid", grid)

        num_basis = grid.shape[-1] - 3 + spline_order
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, num_basis) * 0.01)

    def forward(self, x):
        ''' x: [B, L, in_dim]
            returns: [B, L, out_dim]
        '''
        B, L, D = x.shape
        x = x.reshape(B * L, D)

        y = coef2curve(x_eval=x, grid=self.grid, coef=self.coef, k=self.k)
        y = y.sum(dim=1)
        y = y.reshape(B, L, -1)

        return y

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_k=None, d_v=None, use_kan=False):
        super(AttentionLayer, self).__init__()
        self.n_heads = n_heads
        d_k = d_k or (d_model // n_heads)
        d_v = d_v or (d_model // n_heads)

        self.LN = nn.LayerNorm(d_model)
        self.attention = attention

        self.q_projection = nn.Linear(d_model, d_k * n_heads)
        self.k_projection = nn.Linear(d_model, d_k * n_heads)
        self.v_projection = nn.Linear(d_model, d_model)

        self.sigma_projection = nn.Linear(d_model, n_heads)
        if use_kan:
            print("Using KANLayer")
            self.hurst_projection = KANLayer(d_model, n_heads)
            self.tau_projection = KANLayer(d_model, n_heads)
        else:
            print("Using Linear layer")
            self.hurst_projection = nn.Linear(d_model, n_heads)
            self.tau_projection = nn.Linear(d_model, n_heads)

        self.out_projection = nn.Linear(d_v * n_heads, d_model)

        # Initialise weights
        for m in [
            self.q_projection,
            self.k_projection,
            self.v_projection,
            self.sigma_projection,
            self.hurst_projection,
            self.tau_projection,
        ]:
            # Hopeful bug fix
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeroes_(m.bias)
            ''' Above, each linear projection is initalized independently.
                Xavier is still used as outputs feed dot products (attention) and prior parameters are lated stabilized (sigmoid).
                Unintended coupling between unrelated layers is removed.
            
                Below iterates over m but never initializes m.weight.
                Reinitializes out_projection over and over.
                Each projection layer uses the PyTorch default init, out_projection is reset repeatedly, and priors are poorly conditioned early on in training.
                This bug directly affects prior smoothness, KL stability, and training convergence.
            '''
            # nn.init.xavier_uniform_(self.out_projection.weight)
            # if self.out_projection.bias is not None:
            #     nn.init.zeros_(self.out_projection.bias)

    def forward(self, q, k, v, attn_mask):
        B, L, _ = q.shape
        _, S, _ = k.shape
        H = self.n_heads
        x = q

        q = self.q_projection(q).view(B, L, H, -1)
        k = self.k_projection(k).view(B, S, H, -1)
        v = self.v_projection(v).view(B, S, H, -1)

        sigma = self.sigma_projection(x).view(B, L, H)
        hurst = self.hurst_projection(x).view(B, L, H)
        tau = self.tau_projection(x).view(B, L, H)

        (
            out,
            series,
            prior,
            sigma,
            hurst,
            smoothness_loss,
            beta_prior_loss,
            tau_smoothness_loss,
        ) = self.attention(q, k, v, sigma, hurst, tau, attn_mask)

        out = out.reshape(B, L, -1)

        return (
            self.out_projection(out),
            series,
            prior,
            sigma,
            hurst,
            tau,
            smoothness_loss,
            beta_prior_loss,
            tau_smoothness_loss,
        )
