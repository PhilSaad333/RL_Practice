"""
Baseline strategies for entropy-gradient estimation.

This module factors out baseline logic used in DeltaEntropyApprox for both:
  - grad·dot path (RB estimator) where we need b_k per token to form (G_k - b_k) weights, and
  - JVP path where we need detached weights w_t = (G_t - b_t) at base θ across a microbatch.

Minimize coupling: this file does not import DeltaEntropyApprox. Callers pass in the model,
SequenceProcessor, and any required state (e.g., EMA buffers) explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch


@dataclass
class EmaState:
    """State for position-binned EMA residuals used in Hk_ema baseline."""
    pos_bins: int = 32
    ema_beta: float = 0.95
    ema_resid: Optional[torch.Tensor] = None  # shape [pos_bins], float32 on CPU
    ema_cnt: Optional[torch.Tensor] = None    # shape [pos_bins], int64 on CPU

    def ensure(self) -> None:
        if self.ema_resid is None:
            self.ema_resid = torch.zeros(self.pos_bins, dtype=torch.float32)
        if self.ema_cnt is None:
            self.ema_cnt = torch.zeros(self.pos_bins, dtype=torch.int64)


@dataclass
class RidgeConfig:
    """Ridge fit knobs for hk_ridge baseline (used in JVP helper)."""
    lambda_: float = 1e-3
    eps: float = 1e-8


class BaselineStrategy:
    """Abstract strategy for baselines in RB estimator path."""

    name: str = "base"

    def compute_bk_rb(
        self,
        *,
        H_rb: torch.Tensor,  # [T], graph-carrying
        G: torch.Tensor,     # [T], detached
        phi: Optional[torch.Tensor] = None,  # [T, d], detached features if available
        update_state: bool = False,
    ) -> torch.Tensor:
        """Return b_k tensor (detached, same shape as G). Override in subclasses."""
        raise NotImplementedError


class HkBaseline(BaselineStrategy):
    name = "hk"

    def compute_bk_rb(self, *, H_rb: torch.Tensor, G: torch.Tensor, phi: Optional[torch.Tensor] = None, update_state: bool = False) -> torch.Tensor:
        return H_rb.detach()


class HkEmaBaseline(BaselineStrategy):
    name = "hk_ema"

    def __init__(self, ema_state: EmaState):
        self.ema = ema_state
        self.ema.ensure()

    def compute_bk_rb(self, *, H_rb: torch.Tensor, G: torch.Tensor, phi: Optional[torch.Tensor] = None, update_state: bool = False) -> torch.Tensor:
        T = int(G.numel())
        device = H_rb.device
        pos = torch.arange(T, device=device, dtype=torch.float32) / max(T, 1)
        bins = torch.clamp((pos * self.ema.pos_bins).long(), 0, self.ema.pos_bins - 1)
        mu_hat = self.ema.ema_resid[bins.cpu()].to(device, H_rb.dtype)
        b_k = H_rb.detach() + mu_hat
        if update_state:
            resid = (G - H_rb.detach())
            with torch.no_grad():
                for j in range(T):
                    bidx = int(bins[j].item())
                    self.ema.ema_cnt[bidx] += 1
                    beta = float(self.ema.ema_beta)
                    self.ema.ema_resid[bidx] = (
                        beta * self.ema.ema_resid[bidx]
                        + (1.0 - beta) * resid[j].to(self.ema.ema_resid.dtype).cpu()
                    )
        return b_k


class RegressionBaseline(BaselineStrategy):
    name = "regression"

    def __init__(self, *, l2: float = 0.0, include_intercept: bool = True, fit_dtype: str = "float64", normalize: bool = False, clip_min: float | None = None, clip_max: float | None = None):
        self.l2 = float(l2)
        self.include_intercept = bool(include_intercept)
        self.fit_dtype = torch.float64 if str(fit_dtype).lower() == "float64" else torch.float32
        self.normalize = bool(normalize)
        self.clip_min = clip_min
        self.clip_max = clip_max

    def _fit_beta(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if X is None or X.numel() == 0 or y is None or y.numel() == 0:
            return torch.tensor([], dtype=torch.float32)
        if X.dim() == 1:
            X = X.view(-1, 1)
        Xc = X.detach().to("cpu", self.fit_dtype)
        yc = y.detach().to("cpu", self.fit_dtype).view(-1, 1)
        if self.normalize and Xc.shape[1] > 0:
            mean = Xc.mean(dim=0, keepdim=True)
            std = Xc.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
            Xn = (Xc - mean) / std
        else:
            Xn = Xc
        if self.include_intercept:
            ones = torch.ones((Xn.shape[0], 1), dtype=self.fit_dtype)
            Xn = torch.cat([Xn, ones], dim=1)
        XtX = Xn.T @ Xn
        if self.l2 > 0.0:
            I = torch.eye(XtX.shape[0], dtype=self.fit_dtype)
            XtX = XtX + self.l2 * I
        Xty = Xn.T @ yc
        try:
            beta = torch.linalg.solve(XtX, Xty)
        except RuntimeError:
            beta = torch.linalg.lstsq(Xn, yc).solution
        return beta.view(-1)

    def compute_bk_rb(self, *, H_rb: torch.Tensor, G: torch.Tensor, phi: Optional[torch.Tensor] = None, update_state: bool = False) -> torch.Tensor:
        if phi is None or phi.numel() == 0:
            return H_rb.detach()
        T = int(G.numel())
        if phi.dim() == 2 and int(phi.shape[0]) != T:
            T_eff = min(T, int(phi.shape[0]))
            phi = phi[:T_eff]
            G = G[:T_eff]
            H_rb = H_rb[:T_eff]
        beta = self._fit_beta(phi, G)
        if beta.numel() == 0:
            return H_rb.detach()
        phi_cpu = phi.detach().to("cpu", beta.dtype)
        # Apply the same preprocessing as in fitting
        if self.normalize and phi_cpu.shape[1] > 0:
            mean = phi_cpu.mean(dim=0, keepdim=True)
            std = phi_cpu.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
            phi_n = (phi_cpu - mean) / std
        else:
            phi_n = phi_cpu
        if self.include_intercept:
            ones = torch.ones((phi_n.shape[0], 1), dtype=beta.dtype)
            phi_n = torch.cat([phi_n, ones], dim=1)
        pred_cpu = phi_n @ beta
        b_k = pred_cpu.to(G.device, dtype=G.dtype).detach()
        if self.clip_min is not None or self.clip_max is not None:
            b_k = b_k.clamp(
                min=self.clip_min if self.clip_min is not None else -float("inf"),
                max=self.clip_max if self.clip_max is not None else float("inf"),
            )
        return b_k


# ---------- JVP helper (microbatch weights) ------------------------------------------------------

@torch.no_grad()
def _get_seq_entropy_base(
    model: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    T: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    out0 = model(input_ids.unsqueeze(0), attention_mask=attention_mask.unsqueeze(0), use_cache=False)
    logits0 = out0.logits
    start = max(int(prompt_len) - 1, 0)
    end = start + int(T)
    logits_slice0 = logits0[:, start:end, :]
    logp_full0 = torch.log_softmax(logits_slice0.to(torch.float32), dim=-1)
    p0 = torch.exp(logp_full0)
    H_t0 = (-(p0 * logp_full0).sum(dim=-1)).squeeze(0).detach().to(torch.float32)
    G = torch.flip(torch.cumsum(torch.flip(H_t0, dims=[0]), dim=0), dims=[0]).detach()
    return H_t0.to(device), G.to(device)


def _ridge_fit(
    feats_list: List[torch.Tensor],
    resid_list: List[torch.Tensor],
    *,
    ridge: RidgeConfig,
    to_device: torch.device,
    to_dtype: torch.dtype,
) -> torch.Tensor:
    if not feats_list:
        return torch.zeros(0, device=to_device, dtype=to_dtype)
    Phi = torch.vstack([f.detach().to("cpu", torch.float64) for f in feats_list])
    y = torch.hstack([r.detach().to("cpu", torch.float64) for r in resid_list])
    XtX = Phi.T @ Phi
    d = XtX.shape[0]
    XtX = XtX + (float(ridge.lambda_) + float(ridge.eps)) * torch.eye(d, dtype=XtX.dtype)
    Xty = Phi.T @ y
    w = torch.linalg.solve(XtX, Xty)
    return w.to(device=to_device, dtype=to_dtype)


@torch.no_grad()
def build_weights_base(
    *,
    kind: str,
    model: torch.nn.Module,
    sp,  # SequenceProcessor-like (only for hk_ridge to fetch features)
    mb_E,  # BatchedSequences
    tf_bs: int,
    ema: Optional[EmaState] = None,
    ridge: Optional[RidgeConfig] = None,
) -> List[torch.Tensor]:
    """
    Build detached per-sequence weights w_t = (G_t - b_t) at base θ.
    Supports 'hk', 'hk_ema', 'hk_ridge', and 'none'.
    """
    device = next(model.parameters()).device
    B_mb = int(mb_E.sequences.shape[0])
    weights: List[torch.Tensor] = [torch.empty(0, device=device) for _ in range(B_mb)]

    needs_feats = (str(kind).lower() == "hk_ridge")
    if needs_feats:
        # fetch features in a way consistent with SP API
        res_feats, _ = sp.teacher_force_logprobs_with_diagnostics(
            sequences=mb_E,
            tf_batch_size=tf_bs,
            compute_rb=True,
            with_grad=True,
            return_baseline_features=True,
        )
        feats_list: List[torch.Tensor] = []
        resid_list: List[torch.Tensor] = []

    for b in range(B_mb):
        input_ids = mb_E.sequences[b, 0].to(device=device)
        attention_mask = mb_E.attention_masks[b, 0].to(device=device)
        prompt_len = int(mb_E.prompt_lens[b])
        T = int(mb_E.gen_lens[b][0])
        if T <= 0:
            continue

        H_t0, G = _get_seq_entropy_base(model, input_ids=input_ids, attention_mask=attention_mask, prompt_len=prompt_len, T=T)

        if str(kind).lower() == "hk":
            b_t = H_t0
            weights[b] = (G - b_t).detach()
        elif str(kind).lower() == "hk_ema":
            assert ema is not None, "EMA state required for hk_ema baseline"
            ema.ensure()
            pos = torch.arange(T, device=H_t0.device, dtype=torch.float32) / max(T, 1)
            bins = torch.clamp((pos * ema.pos_bins).long(), 0, ema.pos_bins - 1)
            mu_hat = ema.ema_resid[bins.cpu()].to(H_t0.device, H_t0.dtype)
            b_t = H_t0 + mu_hat
            weights[b] = (G - b_t).detach()
        elif str(kind).lower() == "hk_ridge":
            if res_feats.baseline_feats_torch is None or len(res_feats.baseline_feats_torch[b]) == 0:
                raise RuntimeError("hk_ridge baseline requested but baseline_feats_torch missing.")
            phi_b = res_feats.baseline_feats_torch[b][0].to(device=device, dtype=torch.float32)
            feats_list.append(phi_b)
            resid_list.append((G - H_t0).to(torch.float32))
            weights[b] = torch.empty(0, device=device)
        else:  # none / fallback
            weights[b] = G.detach()

    if needs_feats:
        w = _ridge_fit(
            feats_list, resid_list,
            ridge=(ridge or RidgeConfig()),
            to_device=device, to_dtype=torch.float32,
        )
        for b in range(B_mb):
            T = int(mb_E.gen_lens[b][0])
            if T <= 0:
                continue
            phi_b = res_feats.baseline_feats_torch[b][0].to(device=device, dtype=torch.float32)
            mu_hat = (phi_b @ w).detach()
            H_t0, G = _get_seq_entropy_base(
                model,
                input_ids=mb_E.sequences[b, 0].to(device=device),
                attention_mask=mb_E.attention_masks[b, 0].to(device=device),
                prompt_len=int(mb_E.prompt_lens[b]),
                T=T,
            )
            weights[b] = (G - (H_t0 + mu_hat)).detach()

    return weights


def get_strategy(kind: str, *, ema: Optional[EmaState] = None, **reg_kw) -> BaselineStrategy:
    k = str(kind).lower()
    if k == "hk":
        return HkBaseline()
    if k == "hk_ema":
        if ema is None:
            ema = EmaState()
        return HkEmaBaseline(ema)
    if k in {"regression", "reg", "reg_ridge", "regression_ridge"}:
        return RegressionBaseline(**reg_kw)
    # default fallback
    return HkBaseline()


# ---------- Simple-estimator timewise baselines ---------------------------------------------------

class TimewiseBaselineStrategy:
    """Abstract strategy for simple-estimator timewise baselines.

    Given a ragged list of per-sequence log-prob tensors, returns a list of
    per-sequence baselines b_b matching shapes/devices.
    """
    def compute_timewise_baselines(self, logprob_list: List[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError


class TimeMeanBaseline(TimewiseBaselineStrategy):
    def compute_timewise_baselines(self, logprob_list: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(logprob_list) == 0:
            return []
        max_T = max((int(lp.numel()) for lp in logprob_list), default=0)
        device = logprob_list[0].device
        dtype = logprob_list[0].dtype
        sums = [torch.zeros((), device=device, dtype=dtype) for _ in range(max_T)]
        cnts = [0 for _ in range(max_T)]
        for lp in logprob_list:
            T = int(lp.numel())
            for j in range(T):
                sums[j] = sums[j] + lp[j]
                cnts[j] += 1
        baselines: List[torch.Tensor] = []
        for lp in logprob_list:
            T = int(lp.numel())
            b = torch.empty_like(lp)
            for j in range(T):
                c = max(cnts[j], 1)
                b[j] = sums[j] / c
            baselines.append(b.detach())
        return baselines


class TimeLooBaseline(TimewiseBaselineStrategy):
    def compute_timewise_baselines(self, logprob_list: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(logprob_list) == 0:
            return []
        max_T = max((int(lp.numel()) for lp in logprob_list), default=0)
        device = logprob_list[0].device
        dtype = logprob_list[0].dtype
        sums = [torch.zeros((), device=device, dtype=dtype) for _ in range(max_T)]
        cnts = [0 for _ in range(max_T)]
        for lp in logprob_list:
            T = int(lp.numel())
            for j in range(T):
                sums[j] = sums[j] + lp[j]
                cnts[j] += 1
        baselines: List[torch.Tensor] = []
        for lp in logprob_list:
            T = int(lp.numel())
            b = torch.empty_like(lp)
            for j in range(T):
                c = cnts[j]
                if c <= 1:
                    b[j] = torch.zeros((), device=device, dtype=dtype)
                else:
                    b[j] = (sums[j] - lp[j]) / (c - 1)
            baselines.append(b.detach())
        return baselines


class NoneBaseline(TimewiseBaselineStrategy):
    def compute_timewise_baselines(self, logprob_list: List[torch.Tensor]) -> List[torch.Tensor]:
        return [torch.zeros_like(lp) for lp in logprob_list]


def get_timewise_strategy(kind: str) -> TimewiseBaselineStrategy:
    k = str(kind).lower()
    if k == "time_loo":
        return TimeLooBaseline()
    if k == "time_mean":
        return TimeMeanBaseline()
    return NoneBaseline()
