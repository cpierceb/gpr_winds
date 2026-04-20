# gpr_model.py
import torch
import torch.nn.functional as F
import gpytorch
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from linear_operator.operators import DiagLinearOperator

# =============================================================================
# SHARED CONSTANTS
# =============================================================================

SOURCE_MAP      = {'station': 0, 'urban_tales': 1, 'street_network': 2}
N_SOURCES       = len(SOURCE_MAP)
MATERN_NU       = 2.5
_INIT_RAW_FLOOR = -3.7

KERNEL_COLS   = ['lambda_p', 'mean_height', 'elev_diff', 'height_ag',
                 'zust', 'wd_vert', 'wd_diag_ne', 'wd_horiz', 'wd_diag_se']
MEAN_COLS     = ['zust', 'height_ag', 'z0', 'zd', 'mean_height']
MORPH_FEATS   = ['lambda_p', 'mean_height', 'height_ag']
FORCING_FEATS = ['zust', 'wd_vert', 'wd_diag_ne', 'wd_horiz', 'wd_diag_se', 'elev_diff']

n_kernel     = len(KERNEL_COLS)
morph_dims   = [KERNEL_COLS.index(c) for c in MORPH_FEATS   if c in KERNEL_COLS]
forcing_dims = [KERNEL_COLS.index(c) for c in FORCING_FEATS if c in KERNEL_COLS]

def get_mean_indices(n_kernel_input):
    """Compute ZustMean column indices given actual kernel input width."""
    return {
        'zust_idx': n_kernel_input + MEAN_COLS.index('zust'),
        'z_idx':    n_kernel_input + MEAN_COLS.index('height_ag'),
        'z0_idx':   n_kernel_input + MEAN_COLS.index('z0'),
        'zd_idx':   n_kernel_input + MEAN_COLS.index('zd'),
        'H_idx':    n_kernel_input + MEAN_COLS.index('mean_height'),
    }
# =============================================================================
# MODEL CLASSES
# =============================================================================

class PerSourceNoiseLikelihood(FixedNoiseGaussianLikelihood):
    def __init__(self, noise, source_idx, n_sources, **kwargs):
        super().__init__(noise=noise, learn_additional_noise=False, **kwargs)
        self.register_buffer('source_idx', source_idx)
        self.register_parameter(
            'raw_noise_floors',
            torch.nn.Parameter(torch.full((n_sources,), _INIT_RAW_FLOOR))
        )

    @property
    def noise_floors(self):
        return F.softplus(self.raw_noise_floors)

    def marginal(self, function_dist, *args, **kwargs):
        per_point_floor = self.noise_floors[self.source_idx]
        total_noise     = self.noise + per_point_floor
        covar = (function_dist.lazy_covariance_matrix
                 + DiagLinearOperator(total_noise))
        return function_dist.__class__(function_dist.mean, covar)


class ZustMean(gpytorch.means.Mean):
    def __init__(self, zust_idx, z_idx, z0_idx, zd_idx, H_idx):
        super().__init__()
        self.zust_idx = zust_idx; self.z_idx = z_idx
        self.z0_idx   = z0_idx;   self.zd_idx = zd_idx; self.H_idx = H_idx
        for name, val in [('kap_inv', 2.44), ('b', 0.), ('c', 0.), ('d', 1.),
                           ('e', 0.), ('f', 0.), ('alpha', 0.),
                           ('beta', 0.), ('gamma', 0.)]:
            self.register_parameter(name, torch.nn.Parameter(torch.tensor(val)))

    def forward(self, x):
        zust = x[:, self.zust_idx]
        z    = x[:, self.z_idx]
        z0   = x[:, self.z0_idx].clamp(min=1e-4)
        zd   = x[:, self.zd_idx].clamp(min=1e-2)
        H    = x[:, self.H_idx].clamp(min=1e-2)
        log_arg    = ((z - (zd + self.c)) / (z0 + self.d).clamp(min=1e-4)).clamp(min=1e-3)
        mean_above = self.kap_inv * (zust + self.b) * (torch.log(log_arg) + self.e) + self.f
        exp_arg    = (self.beta * (z - H) / H).clamp(-20., 20.)
        mean_below = self.alpha * torch.exp(exp_arg) + self.gamma
        return torch.where(z > H, mean_above, mean_below)


class WindGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood,
                 morph_dims, forcing_dims, zust_idx, z_idx, z0_idx, zd_idx, H_idx, n_pca=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZustMean(zust_idx, z_idx, z0_idx, zd_idx, H_idx)
        lsc = gpytorch.constraints.Interval(0.05, 1.5)

        if n_pca is not None:
            self.k_morph = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=MATERN_NU, ard_num_dims=n_pca,
                    active_dims=torch.arange(n_pca), 
                    lengthscale_constraint=lsc,
                )
            )
            # self.k_forcing   = gpytorch.kernels.ScaleKernel(gpytorch.kernels.ConstantKernel())  # dummy
            self.covar_module = self.k_morph
        else:
            self.k_morph = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=MATERN_NU, ard_num_dims=len(morph_dims),
                    active_dims=torch.tensor(morph_dims), lengthscale_constraint=lsc,
                )
            )
            self.k_forcing = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    nu=MATERN_NU, ard_num_dims=len(forcing_dims),
                    active_dims=torch.tensor(forcing_dims), lengthscale_constraint=lsc,
                )
            )
            self.covar_module = self.k_morph + self.k_forcing

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def build_model(train_x, train_y, noise_tr, source_idx, n_pca=None):
    """Fresh model+likelihood with randomised lengthscale init."""
    n_kernel_input = n_pca if n_pca is not None else n_kernel
    idx = get_mean_indices(n_kernel_input)
    likelihood = PerSourceNoiseLikelihood(noise=noise_tr, source_idx=source_idx, n_sources=N_SOURCES)
    model = WindGP(train_x, train_y, likelihood,
                   morph_dims, forcing_dims,
                   idx['zust_idx'], idx['z_idx'], idx['z0_idx'],
                   idx['zd_idx'],   idx['H_idx'],
                   n_pca=n_pca)
    with torch.no_grad():
        if n_pca is not None:
            model.k_morph.base_kernel.lengthscale = torch.rand(1, n_pca) * 1.4 + 0.1
        else:
            model.k_morph.base_kernel.lengthscale   = torch.rand(1, len(morph_dims))   * 1.4 + 0.1
            model.k_forcing.base_kernel.lengthscale = torch.rand(1, len(forcing_dims)) * 1.4 + 0.1
    return model, likelihood


def load_model(ckpt_path):
    """Reconstruct model+likelihood from a saved checkpoint, ready for eval."""
    import torch
    from sklearn.preprocessing import StandardScaler

    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    n_tr    = len(ckpt['noise_tr'])
    dummy_x = torch.zeros(n_tr, len(KERNEL_COLS) + len(MEAN_COLS))
    dummy_y = torch.zeros(n_tr)

    likelihood = PerSourceNoiseLikelihood(
        noise=ckpt['noise_tr'], source_idx=ckpt['source_idx'], n_sources=N_SOURCES
    )
    model = WindGP(dummy_x, dummy_y, likelihood,
                   morph_dims, forcing_dims, zust_idx, z_idx, z0_idx, zd_idx, H_idx)
    model.load_state_dict(ckpt['model_state'])
    likelihood.load_state_dict(ckpt['likelihood_state'])
    model.eval(); likelihood.eval()

    scaler = StandardScaler()
    scaler.mean_  = ckpt['scaler_mean']
    scaler.scale_ = ckpt['scaler_scale']

    return model, likelihood, scaler, ckpt

def get_pca_dims(n_pca_components):
    """After PCA, kernel input is just n_pca_components sequential dims."""
    all_dims    = list(range(n_pca_components))
    # Can no longer split morph/forcing meaningfully after PCA —
    # use a single kernel over all PCA dims instead
    return all_dims