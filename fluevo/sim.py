"""A 市季节性流感的随机 SEIRS 仿真器。

S -> E -> I -> R -> S：康复者以速率 omega 失去免疫，重新变成易感者，所以每年都会再流行。
传播率带年周期：log beta(d) = B0 + AMP * cos(2*pi*(d - PEAK)/365)，d 是一年中的第几天。
观测只有两路：每日新增病例 y，搜索指数 s（正比于当前正在生病的人数 I）。
"""
import numpy as np

N = 5e6
SIGMA, GAMMA0, RHO = 0.5, 0.3, 0.10        # 潜伏期 2 天；病程约 3.3 天；10% 的发病被报告（均视为已知）
OMEGA_TRUE = 1 / 250                       # 免疫平均维持 250 天
B0_TRUE, AMP_TRUE, PEAK_TRUE = np.log(0.42), 0.12, 200
IMPORT = 10.0                              # 每天少量输入病例，防止淡季灭绝
PHI_TRUE = 100.0                           # 病例计数的负二项离散度（单日波动约 ±10%）
SEARCH_C, SEARCH_SD = 0.02, 0.15
YEAR = 365


def log_beta_season(t, b0=None, b1=None, b2=None):
    w = 2 * np.pi * (np.asarray(t) % YEAR) / YEAR
    if b0 is None:
        b0 = B0_TRUE
    if b1 is None:
        b1 = AMP_TRUE * np.cos(2 * np.pi * PEAK_TRUE / YEAR)
        b2 = AMP_TRUE * np.sin(2 * np.pi * PEAK_TRUE / YEAR)
    return b0 + b1 * np.cos(w) + b2 * np.sin(w)


def simulate(seed, years=6, change=None, burn_years=4):
    """change = dict(kind='beta'|'gamma'|'sus', day=记录期内的绝对天, k=幅度)。
    beta/gamma：从那天起乘以 k；sus：那天有比例 k 的康复者一次性变回易感者（免疫逃逸）。"""
    rng = np.random.default_rng(seed)
    S, E, I = 0.72 * N, 300.0, 400.0
    R = N - S - E - I
    T = years * YEAR
    y, s = np.zeros(T), np.zeros(T)
    tr = {k: np.zeros(T) for k in ("S", "E", "I", "beta", "gamma")}
    for tt in range(-burn_years * YEAR, T):
        t = max(tt, 0)
        on = change is not None and tt >= change["day"]
        beta = np.exp(log_beta_season(tt)) * (change["k"] if on and change["kind"] == "beta" else 1.0)
        gamma = GAMMA0 * (change["k"] if on and change["kind"] == "gamma" else 1.0)
        if change is not None and change["kind"] == "sus" and tt == change["day"]:
            moved = change["k"] * R
            S, R = S + moved, R - moved
        EI = rng.binomial(int(E), SIGMA)
        IR = rng.binomial(int(I), min(gamma, 1.0))
        RS = rng.binomial(int(R), OMEGA_TRUE)
        if tt >= 0:
            mu = RHO * EI
            y[t] = rng.negative_binomial(PHI_TRUE, PHI_TRUE / (PHI_TRUE + mu)) if mu > 0 else 0
            s[t] = SEARCH_C * max(I, 1.0) * np.exp(rng.normal(0, SEARCH_SD))
            tr["S"][t], tr["E"][t], tr["I"][t], tr["beta"][t], tr["gamma"][t] = S, E, I, beta, gamma
        newE = min(rng.poisson(beta * I * S / N * np.exp(rng.normal(0, 0.05)) + IMPORT), S)
        S, E, I, R = S - newE + RS, E + newE - EI, I + EI - IR, R + IR - RS
    return dict(y=y, s=s, truth=tr, change=change, seed=seed)
