"""预训练：用前五年的每日新增（和搜索指数）估计一组初始参数。

方法仍是卡尔曼滤波：把季节传播率的三个系数 (b0, b1, b2) 和免疫消退率 log(omega) 并进状态向量，
让滤波器在五年数据上边跑边估。跑若干遍，每遍以上一遍的结果为中心、把参数的不确定范围收窄。
然后固定参数再跑一遍，估计搜索指数的比例常数 c，并校准观测噪声，使标准化预测误差的方差约为 1。
"""
import numpy as np

from .filter import Filter
from .sim import N, SIGMA, GAMMA0, RHO

GUESS = dict(b0=np.log(0.50), b1=0.0, b2=0.0, lw=np.log(1 / 400))     # 故意不等于真值的初始猜测
SPREAD = dict(b0=0.25, b1=0.15, b2=0.15, lw=0.5)


def init_ensemble(y, theta, spread, n, rng):
    E = (np.mean(y[:7]) + 1) / (RHO * SIGMA) * np.exp(rng.normal(0, 0.3, n))
    I = E * SIGMA / GAMMA0 * np.exp(rng.normal(0, 0.3, n))
    S = N * rng.uniform(0.55, 0.90, n)
    ens = dict(S=S, E=E, I=I, R=N - S - E - I, lm=np.zeros(n), lg=np.zeros(n))
    for k in ("b0", "b1", "b2", "lw"):
        ens[k] = theta[k] + rng.normal(0, spread[k], n) if spread else np.full(n, theta[k])
    return ens


def pretrain(y, s, seed=0, passes=4, n=300, verbose=False):
    rng = np.random.default_rng(seed)
    theta, spread = dict(GUESS), dict(SPREAD)
    for p in range(passes):
        f = Filter(init_ensemble(y, theta, spread, n, rng), seed=seed + p, est_theta=True,
                   rw=dict(lm=0.0, lg=0.0, b0=0.001, b1=0.001, b2=0.001, lw=0.002))
        for t in range(len(y)):
            f.step(t, y[t])
        theta = {k: f.mean(k) for k in theta}
        spread = {k: max(v * 0.5, 0.02) for k, v in spread.items()}
        if verbose:
            print(f"  第{p + 1}遍", {k: round(v, 3) for k, v in theta.items()})

    # 参数固定后再跑一遍：估计 c，收集标准化预测误差
    f = Filter(init_ensemble(y, theta, None, n, rng), seed=seed + 50)
    logI = np.zeros(len(y))
    for t in range(len(y)):
        logI[t] = np.log(f.mean("I"))
        f.step(t, y[t])
    ok = np.arange(len(y)) > 365
    c = float(np.exp(np.mean(np.log(s[ok]) - logI[ok])))

    # 校准观测噪声：让无变点年份里的标准化预测误差方差约为 1
    f = Filter(init_ensemble(y, theta, None, n, rng), seed=seed + 60, c=c)
    nus = np.array([f.step(t, y[t], s[t])["nu"] for t in range(len(y))])
    v = nus[365:].var(0)
    r_case, r_search = float(v[0]), float(v[1])
    f = Filter(init_ensemble(y, theta, None, n, rng), seed=seed + 70, c=c, r_case=r_case, r_search=r_search)
    nus = np.array([f.step(t, y[t], s[t])["nu"] for t in range(len(y))])
    return dict(theta=theta, c=c, r_case=r_case, r_search=r_search, filter=f, nus=nus)
