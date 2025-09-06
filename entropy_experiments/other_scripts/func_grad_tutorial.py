# Colab starter: LoRA toy model + jvp/vjp HVP practice
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.func import functional_call, grad, jvp, vjp
torch.manual_seed(0); device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Toy data (two Gaussian blobs) ----------
def make_blobs(n=1024, d=8, sep=2.0):
    n1 = n//2; n2 = n - n1
    mu1 = torch.zeros(d); mu2 = torch.zeros(d); mu2[0] = sep
    x1 = torch.randn(n1, d) + mu1; x2 = torch.randn(n2, d) + mu2
    X = torch.cat([x1, x2], 0); y = torch.cat([torch.zeros(n1), torch.ones(n2)], 0).long()
    perm = torch.randperm(n)
    return X[perm], y[perm]

X_E, y_E = make_blobs(1024, d=16, sep=3.0)
X_U, y_U = make_blobs(1024, d=16, sep=3.0)
X_E, y_E, X_U, y_U = X_E.to(device), y_E.to(device), X_U.to(device), y_U.to(device)

# ---------- LoRA-style linear layer: W_eff = W0 + (alpha/r) * B @ A ----------
class LoRALinear(nn.Module):
    def __init__(self, in_dim, out_dim, r=4, alpha=8, bias=False):
        super().__init__()
        self.W0 = nn.Parameter(torch.empty(out_dim, in_dim))  # frozen base
        nn.init.kaiming_uniform_(self.W0, a=math.sqrt(5))
        self.W0.requires_grad_(False)

        self.A  = nn.Parameter(torch.zeros(r, in_dim))
        self.B  = nn.Parameter(torch.zeros(out_dim, r))
        self.scaling = alpha / float(r)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x):
        # W_eff = W0 + (alpha/r) * B @ A
        W_eff = self.W0 + self.scaling * (self.B @ self.A)
        y = x @ W_eff.t()
        if self.bias is not None:
            y = y + self.bias
        return y

class ToyModel(nn.Module):
    def __init__(self, d_in=16, d_out=2, r=4, alpha=8):
        super().__init__()
        # One hidden freeze + LoRA head to emphasize "some params not trainable"
        self.hidden = nn.Linear(d_in, d_in, bias=False)
        nn.init.orthogonal_(self.hidden.weight); self.hidden.weight.requires_grad_(False)
        self.head = LoRALinear(d_in, d_out, r=r, alpha=alpha, bias=False)

    def forward(self, x):
        x = self.hidden(x)  # frozen transform
        return self.head(x)

model = ToyModel(d_in=16, d_out=2, r=2, alpha=4).to(device).eval()

# ---------- Utilities: parameter dictionaries ----------
def split_params(module):
    all_params = dict(module.named_parameters())
    trainable = {n: p.detach().clone().requires_grad_(True) for n,p in all_params.items() if p.requires_grad}
    frozen    = {n: p.detach().clone().requires_grad_(False) for n,p in all_params.items() if not p.requires_grad}
    buffers   = dict(module.named_buffers())
    return trainable, frozen, buffers

trainable, frozen, buffers = split_params(model)

def merge_params(trainable_dict):
    # combine trainable overrides with frozen constants for functional_call
    merged = {**frozen, **trainable_dict}
    return merged if len(buffers)==0 else (merged, buffers)  # functional_call accepts one or two dicts

# ---------- Scalar functionals ----------
def entropy_scalar_from_logits(logits):
    # Mean categorical entropy H = E[- sum p log p]
    logp = F.log_softmax(logits, dim=-1)
    p = logp.exp()
    return (-(p * logp).sum(dim=-1)).mean()

def H_of_params(trainable_dict, X_batch):
    # H(θ) evaluated in eval mode (no dropout), with explicit parameters via functional_call
    model.eval()
    merged = merge_params(trainable_dict)
    logits = functional_call(model, merged, (X_batch,))
    return entropy_scalar_from_logits(logits)

def CE_U_of_params(trainable_dict, X_batch, y_batch):
    merged = merge_params(trainable_dict)
    logits = functional_call(model, merged, (X_batch,))
    return F.cross_entropy(logits, y_batch)

# ---------- Build an update direction v (per-unit-LR) ----------
# Option 1 (simple): v = grad_θ CE_U (identity "preconditioner")
def grad_on_U(trainable_dict, X, y, micro_bs=128):
    # Accumulate ∇ CE_U over microbatches; returns dict[name]->Tensor
    g = {k: torch.zeros_like(v) for k,v in trainable_dict.items()}
    for i in range(0, X.shape[0], micro_bs):
        xb = X[i:i+micro_bs]; yb = y[i:i+micro_bs]
        # fresh param copies per microbatch to keep autograd graphs disjoint
        local = {k: v.clone().requires_grad_(True) for k,v in trainable_dict.items()}
        loss = CE_U_of_params(local, xb, yb)
        grads = grad(lambda td: CE_U_of_params(td, xb, yb))(local)  # ∇ wrt dict
        for k in g: g[k] += grads[k].detach()
        # free graphs: no retain needed because we detach
    return g

v_named = grad_on_U(trainable, X_U, y_U, micro_bs=128)  # this is your "step direction" v

# ---------- HVP via torch.func: forward-over-reverse (jvp(grad(H))) ----------
def hvp_forward_over_reverse(trainable_dict, X, v_named, micro_bs=256):
    # microbatched H = sum H_i, so HVP sums too
    num = den = torch.tensor(0., device=device)
    for i in range(0, X.shape[0], micro_bs):
        xb = X[i:i+micro_bs]
        # define H_i(θ) for this microbatch
        def H_mb(td): return H_of_params(td, xb)
        g_fun = grad(H_mb)                           # dict -> dict : ∇H_i
        _, hvp_mb = jvp(g_fun, (trainable_dict,), (v_named,))   # (∇^2 H_i) v
        g_mb = g_fun(trainable_dict)                 # ∇H_i
        # dot products in name space
        num = num + sum((g_mb[k]*v_named[k]).sum() for k in v_named)
        den = den + sum((v_named[k]*hvp_mb[k]).sum() for k in v_named)
    return num, den

# Fallback path if any op lacks forward-mode AD coverage
def hvp_reverse_over_reverse(trainable_dict, X, v_named, micro_bs=256):
    num = den = torch.tensor(0., device=device)
    for i in range(0, X.shape[0], micro_bs):
        xb = X[i:i+micro_bs]
        def H_mb(td): return H_of_params(td, xb)
        g_fun = grad(H_mb)
        _, vjp_fn = vjp(g_fun, trainable_dict)       # returns a callable for cotangents
        hvp_mb, = vjp_fn(v_named)                    # (∇^2 H_i) v
        g_mb = g_fun(trainable_dict)
        num = num + sum((g_mb[k]*v_named[k]).sum() for k in v_named)
        den = den + sum((v_named[k]*hvp_mb[k]).sum() for k in v_named)
    return num, den

# ---------- Compute η* and a finite-difference check ----------
with torch.no_grad():
    num, den = hvp_forward_over_reverse(trainable, X_E, v_named, micro_bs=256)
    eta_star = (2.0 * num.abs()) / (den.abs() + 1e-12)
    v_norm2 = sum((t*t).sum() for t in v_named.values())
    kappa_v = den / (v_norm2 + 1e-12)
    print(f"||v||2 = {v_norm2.sqrt().item():.4e},  <∇H,v> = {num.item():.4e},  <v,Hv> = {den.item():.4e}")
    print(f"eta* ≈ {eta_star.item():.4e},  directional curvature κ_v ≈ {kappa_v.item():.4e}")

# Finite-difference along v to see linear vs quadratic terms
def apply_step(trainable_dict, v_named, eta):
    return {k: (trainable_dict[k] + eta * v_named[k]) for k in trainable_dict}

def H_on_E(td): return H_of_params(td, X_E)

with torch.no_grad():
    for eta in [1e-3, 3e-3, 1e-2]:
        H0   = H_on_E(trainable)
        Heta = H_on_E(apply_step(trainable, v_named, -eta))  # gradient descent direction
        dH1  = -eta * sum((grad(H_of_params)(trainable, X_E)[k] * v_named[k]).sum()
                          for k in v_named)                   # <∇H,v> * (-eta)
        print(f"eta={eta: .1e}  ΔH_true={float(Heta - H0):+.4e}  ΔH1={float(dH1):+.4e}  ratio≈{float((Heta-H0)/(dH1+1e-12)):.3f}")
