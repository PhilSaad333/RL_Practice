# Just a skeleton so far




class DeltaEntropyApprox:
    def __init__(self, model, sequence_processor, config, logger):
        self.model = model
        self.sp = sequence_processor
        self.cfg = config
        self.logger = logger
        self.mb = int(config.get('approx_delta_h', {}).get('microbatch_size', 8))
        self.use_rb = not bool(config.get('estimator', {}).get('use_simple_entropy_for_x', False))
        self.baseline_kind = (config.get('approx_delta_h', {}).get('baseline', {}) or {}).get('kind', 'Hk')
        self.normalize = str(config.get('approx_delta_h', {}).get('normalize', 'per_token'))

    def _select_params_intersecting(self, v_named):
        name_to_param = OrderedDict(
            (n, p) for n, p in self.model.named_parameters() if p.requires_grad and (n in v_named)
        )
        return name_to_param

    def _scale_for_average(self, B_total, T_total, B_mb, T_mb):
        if self.normalize == 'per_sequence':
            return B_mb / max(B_total, 1)
        elif self.normalize == 'per_token':
            return T_mb / max(T_total, 1)
        else:
            return 1.0

    def compute_delta_h_approx(self, *, E_batch, v_named):
        mdl = self.model
        device = next(mdl.parameters()).device

        # Pre-count tokens for normalization
        T_total = self._count_total_gen_tokens(E_batch)

        mdl.zero_grad(set_to_none=True)
        B_total = int(E_batch['sequences'].shape[0])

        # microbatch loop
        for mb in self._iter_microbatches(E_batch, self.mb):
            res, _diag = self.sp.teacher_force_logprobs_with_diagnostics(
                sequences=mb, tf_batch_size=self.mb, compute_rb=True,
                with_grad=True, return_baseline_features=False
            )
            # tensors: logprobs[b][0] [T], rb_entropies_torch[b][0] [T] (if use_rb)
            sur = 0.0
            T_mb = 0
            for b in range(len(res.logprobs)):
                lp = res.logprobs[b][0]                       # [T]
                T = lp.numel()
                T_mb += T
                if T == 0: 
                    continue
                if self.use_rb:
                    H_rb = res.rb_entropies_torch[b][0]       # [T] (graph)
                    G = torch.flip(torch.cumsum(torch.flip(H_rb.detach(), dims=[0]), dim=0), dims=[0])
                    b_k = H_rb.detach() if self.baseline_kind == 'Hk' else torch.zeros_like(G)
                    sur = sur + ((G - b_k) * lp.detach()).sum() + H_rb.sum()
                else:
                    # Simple path (rarely used): treat −logπ as entropy proxy (no RB term)
                    G = torch.flip(torch.cumsum(torch.flip((-lp).detach(), dims=[0]), dim=0), dims=[0])
                    b_k = (-lp).detach() if self.baseline_kind == 'Hk' else torch.zeros_like(G)
                    sur = sur + ((G - b_k) * lp.detach()).sum()

            scale = self._scale_for_average(B_total, T_total, len(res.logprobs), T_mb)
            (sur * scale).backward()

        # Gather grads on the intersection (CPU fp32)
        name_to_param = self._select_params_intersecting(v_named)
        grads_named = {
            n: (p.grad.detach().to('cpu', torch.float32).clone() if p.grad is not None
                else torch.zeros_like(p.detach()).to('cpu', torch.float32))
            for n, p in name_to_param.items()
        }

        # Dot with v (uses float64 accumulation)
        delta_h_per_lr = float(dot_named(grads_named, v_named).item())  # ← your helper
        # add diagnostics...
        return {'delta_h_per_lr': delta_h_per_lr, ...}
