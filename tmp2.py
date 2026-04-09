"""
Demonstration: Analytical KL vs Jensen upper bound.

KL_exact  = -0.5 * log(1 - rho_post^2)

Jensen bound = E_p[ -V + (b-1)*logZ - b*(b-1)/2*log(2pi*tau^2) + sum_i E_{q_i}[V] ]

Slack (always >= 0) = E_p[ sum_i ( E_{q_i}[V] + log E_{q_i}[e^{-V}] ) ]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Core computation ──────────────────────────────────────────────────────────
def compute(rho, tau2, t=0.5, s1=1.0, s2=1.0, N=3000, M=200, seed=0):
    rng     = np.random.default_rng(seed)
    b       = 2
    sigma_t = np.sqrt(1 - np.exp(-2*t))
    Sigma   = np.array([[s1**2, rho*s1*s2], [rho*s1*s2, s2**2]])

    xt = np.exp(-t) * np.array([1.5, -1.0]) + sigma_t * rng.standard_normal(2)

    # posterior p(x0|xt)
    prec_like  = np.exp(-2*t) / sigma_t**2 * np.eye(2)
    Sigma_post = np.linalg.inv(np.linalg.inv(Sigma) + prec_like)
    mu_post    = Sigma_post @ prec_like @ (np.exp(t) * xt)
    X0         = rng.multivariate_normal(mu_post, Sigma_post, size=N)
    x1s, x2s   = X0[:, 0], X0[:, 1]

    # logZ — paper definition: log integral exp(-||xt-e^{-t}x0||^2/(2sigma^2) - V) dx0
    #       = log p(xt) + b/2 * log(2pi*sigma_t^2)
    Sigma_xt = np.exp(-2*t) * Sigma + sigma_t**2 * np.eye(2)
    inv_xt   = np.linalg.inv(Sigma_xt)
    log_pxt  = -0.5*(xt @ inv_xt @ xt) - 0.5*np.log((2*np.pi)**b * np.linalg.det(Sigma_xt))
    logZ     = log_pxt + b/2 * np.log(2*np.pi * sigma_t**2)

    # V(x0) = -log prior(x0)
    inv_p = np.linalg.inv(Sigma)
    def V(a, c):
        a, c = np.asarray(a), np.asarray(c)
        q = inv_p[0,0]*a**2 + 2*inv_p[0,1]*a*c + inv_p[1,1]*c**2
        return 0.5*q + 0.5*np.log((2*np.pi)**2 * np.linalg.det(Sigma))

    V_s = V(x1s, x2s)

    # E_{q_i}[V] and log E_{q_i}[exp(-V)] for each coordinate
    EqV_list, log_Eq_expnegV_list = [], []
    for ci, xi in [(0, x1s), (1, x2s)]:
        j   = 1 - ci
        x0j = np.exp(t)*xt[j] + np.sqrt(tau2)*rng.standard_normal((N, M))
        xir = xi[:, None] * np.ones((1, M))
        v   = V(xir, x0j) if ci == 0 else V(x0j, xir)     # (N, M)
        EqV_list.append(v.mean(axis=1))
        # log E[exp(-V)] via log-mean-exp
        nv   = -v
        cmax = nv.max(axis=1, keepdims=True)
        lme  = np.log(np.mean(np.exp(nv - cmax), axis=1)) + cmax[:, 0]
        log_Eq_expnegV_list.append(lme)

    # exact KL
    corr_p   = Sigma_post[0,1] / np.sqrt(Sigma_post[0,0] * Sigma_post[1,1])
    kl_exact = -0.5 * np.log(1 - corr_p**2)

    # exact formula (eq. line 4): replace -log E[e^{-V}] with its exact value
    log_Zt1 = 0.5*(b-1)*np.log(2*np.pi*tau2) + log_Eq_expnegV_list[0]
    log_Zt2 = 0.5*(b-1)*np.log(2*np.pi*tau2) + log_Eq_expnegV_list[1]
    kl_formula = np.mean(-V_s + (b-1)*logZ - log_Zt1 - log_Zt2)

    # Jensen bound: replace -log E[e^{-V}] <= E[V]
    kl_bound = np.mean(
        -V_s + (b-1)*logZ
        - b*(b-1)/2 * np.log(2*np.pi*tau2)
        + EqV_list[0] + EqV_list[1]
    )

    # per-sample Jensen slack = sum_i (E_{q_i}[V] + log E_{q_i}[e^{-V}])  >= 0
    slack = np.mean(
        (EqV_list[0] + log_Eq_expnegV_list[0]) +
        (EqV_list[1] + log_Eq_expnegV_list[1])
    )

    return kl_exact, kl_formula, kl_bound, slack

# ── Sweep 1: vary rho at fixed tau2 ──────────────────────────────────────────
rhos      = np.linspace(0.0, 0.97, 25)
tau2_fix  = 1.0
r1 = np.array([compute(r, tau2_fix) for r in rhos])

# ── Sweep 2: vary tau2 at fixed rho ──────────────────────────────────────────
tau2s    = np.logspace(-2, 1.5, 25)
rho_fix  = 0.85
r2 = np.array([compute(rho_fix, t2) for t2 in tau2s])

# ── Sweep 3: slack heatmap over (rho, tau2) ───────────────────────────────────
rho_g  = np.linspace(0.05, 0.95, 12)
tau2_g = np.logspace(-1.5, 1.5, 12)
slack_map = np.zeros((len(tau2_g), len(rho_g)))
for i, t2 in enumerate(tau2_g):
    for j, r in enumerate(rho_g):
        slack_map[i, j] = compute(r, t2, N=1000, M=100, seed=i*13+j)[3]

# ── Plot ──────────────────────────────────────────────────────────────────────
C_exact   = '#2563EB'
C_formula = '#16A34A'
C_bound   = '#DC2626'
C_fill    = '#F59E0B'

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 3, hspace=0.45, wspace=0.38)

# -- Panel 1: KL vs rho
ax1 = fig.add_subplot(gs[0, :2])
ax1.fill_between(rhos, r1[:,0], r1[:,2], alpha=0.12, color=C_fill, label='Jensen slack')
ax1.plot(rhos, r1[:,0], color=C_exact,   lw=2.5, label='Exact KL  $-\\frac{1}{2}\\log(1-\\rho_{\\mathrm{post}}^2)$')
ax1.plot(rhos, r1[:,1], color=C_formula, lw=1.8, ls='--', label='Exact formula (MC estimate, eq. 4)')
ax1.plot(rhos, r1[:,2], color=C_bound,   lw=1.8, ls=':',  label=f'Jensen bound  ($\\tau^2={tau2_fix}$)')
ax1.set_xlabel('Prior correlation $\\rho$', fontsize=12)
ax1.set_ylabel('KL divergence (nats)', fontsize=12)
ax1.set_title('KL vs correlation $\\rho$', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9.5, framealpha=0.9)
ax1.grid(alpha=0.2); ax1.set_xlim(0, 0.97)

# -- Panel 2: slack vs rho
ax2 = fig.add_subplot(gs[0, 2])
ax2.bar(rhos, r1[:,3], width=0.035, color=C_fill, alpha=0.85, edgecolor='none')
ax2.set_xlabel('Prior correlation $\\rho$', fontsize=11)
ax2.set_ylabel('Slack (nats)', fontsize=11)
ax2.set_title('Jensen slack vs $\\rho$', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.2, axis='y')

# -- Panel 3: KL vs tau2
ax3 = fig.add_subplot(gs[1, :2])
ax3.fill_between(tau2s, r2[:,0], r2[:,2], alpha=0.12, color=C_fill)
ax3.axhline(r2[0,0], color=C_exact,   lw=2.5, label=f'Exact KL  ($\\rho={rho_fix}$)')
ax3.plot(tau2s, r2[:,1], color=C_formula, lw=1.8, ls='--', label='Exact formula (MC estimate)')
ax3.plot(tau2s, r2[:,2], color=C_bound,   lw=1.8, ls=':',  label='Jensen bound')
ax3.set_xscale('log')
ax3.set_xlabel('$\\tau^2$  (variance of $q_i$)', fontsize=12)
ax3.set_ylabel('KL divergence (nats)', fontsize=12)
ax3.set_title('KL vs $\\tau^2$  —  bound loosens as $\\tau^2$ grows', fontsize=13, fontweight='bold')
ax3.legend(fontsize=9.5, framealpha=0.9)
ax3.grid(alpha=0.2)

# -- Panel 4: slack heatmap
ax4 = fig.add_subplot(gs[1, 2])
im = ax4.pcolormesh(rho_g, np.log10(tau2_g), slack_map, cmap='YlOrRd', shading='auto')
plt.colorbar(im, ax=ax4, label='Slack (nats)')
ax4.set_xlabel('Prior correlation $\\rho$', fontsize=11)
ax4.set_ylabel('$\\log_{10}(\\tau^2)$', fontsize=11)
ax4.set_title('Slack heatmap\n$(\\rho,\\, \\tau^2)$', fontsize=12, fontweight='bold')

fig.suptitle(
    'Analytical KL  vs  Jensen upper bound\n'
    r'$D_{\mathrm{KL}} \leq \mathbb{E}_p[-V+(b-1)\log Z'
    r' - \frac{b(b-1)}{2}\log 2\pi\tau^2 + \sum_i \mathbb{E}_{q_i}[V]]$',
    fontsize=12, y=1.02
)

plt.savefig('kl_demo.png', dpi=150, bbox_inches='tight')
print("done")