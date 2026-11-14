"""增广状态的集合卡尔曼滤波（EnKF）。

状态向量（每个集合成员一份）：
    x = [ S, E, I, R,  lm, lg,  (b0, b1, b2, lw) ]
    lm = log(传播率的干预乘子)   lg = log(康复率乘子)
    (b0, b1, b2) 季节传播率的傅里叶系数，lw = log(免疫消退率)；只在预训练时当作待估状态
状态方程：SEIRS 机理 + 参数的随机游走
观测方程：z1 = log(rho*sigma*E + 1)  <- 每日新增病例
          z2 = log(c * I)            <- 搜索指数（可缺）
"""
import numpy as np

from .sim import N, SIGMA, GAMMA0, RHO, IMPORT, YEAR

HORIZON = 7
STATE = ("S", "E", "I", "R", "lm", "lg", "b0", "b1", "b2", "lw")


class Filter:
    def __init__(self, ens, seed=0, c=None, r_case=1.0, r_search=1.0, q=0.05, rw=None, est_theta=False):
        self.x = {k: np.array(v, float) for k, v in ens.items()}     # 每个量一个长度为 n 的数组
        self.n = len(self.x["S"])
        self.rng = np.random.default_rng(seed)
        self.c = c                                  # 搜索指数的比例常数；None 表示不用搜索指数
        self.r_case, self.r_search = r_case, r_search   # 观测噪声方差的校准系数
        self.phi, self.search_sd = 100.0, 0.15
        self.q = q                                  # 传播过程噪声
        self.rw = dict(lm=0.005, lg=0.002, b0=0.0, b1=0.0, b2=0.0, lw=0.0) | (rw or {})
        self.est_theta = est_theta

    def copy(self, seed=None):
        f = Filter(self.x, seed if seed is not None else int(self.rng.integers(1 << 30)), self.c,
                   self.r_case, self.r_search, self.q, dict(self.rw), self.est_theta)
        return f

    # ------------------------------------------------------------ 状态方程
    def _advance(self, x, t, rng, noise=True):
        w = 2 * np.pi * (t % YEAR) / YEAR
        beta = np.exp(x["b0"] + x["b1"] * np.cos(w) + x["b2"] * np.sin(w) + x["lm"])
        gamma = np.minimum(GAMMA0 * np.exp(x["lg"]), 0.95)
        omega = np.exp(x["lw"])
        S, E, I, R = x["S"], x["E"], x["I"], x["R"]
        eps = np.exp(rng.normal(0, self.q, self.n)) if noise else 1.0
        newE = np.minimum(beta * I * S / N * eps + IMPORT, S)
        x["S"] = S - newE + omega * R
        x["E"] = np.maximum(E + newE - SIGMA * E, 1.0)
        x["I"] = np.maximum(I + SIGMA * E - gamma * I, 1.0)
        x["R"] = np.maximum(R + gamma * I - omega * R, 1.0)
        if noise:
            for k, sd in self.rw.items():
                if sd > 0:
                    x[k] = x[k] + rng.normal(0, sd, self.n)

    # ------------------------------------------------------------ 观测方程
    def _obs(self, x):
        h = [np.log(RHO * SIGMA * x["E"] + 1.0)]
        if self.c is not None:
            h.append(np.log(self.c * x["I"]))
        return np.array(h)

    def _obs_var(self, x):
        mu = float(np.mean(RHO * SIGMA * x["E"]))
        r = [self.r_case * (1.0 / (mu + 1.0) + 1.0 / self.phi)]      # 负二项计数取对数后的近似方差
        if self.c is not None:
            r.append(self.r_search * self.search_sd ** 2)
        return np.array(r)

    # ------------------------------------------------------------ 一天：预测检验 -> 更新 -> 推进
    def step(self, t, y, s=None):
        """集合当前表示第 t 天开始时的状态。返回标准化预测误差和预测对数似然。"""
        x = self.x
        z = [np.log(y + 1.0)]
        use_s = self.c is not None and s is not None and not np.isnan(s)
        h, r = self._obs(x), self._obs_var(x)
        if use_s:
            z.append(np.log(s))
        else:
            h, r = h[:1], r[:1]
        z = np.array(z)
        hm = h.mean(1)
        dh = h - hm[:, None]
        Pzz = dh @ dh.T / (self.n - 1) + np.diag(r)          # 预测分布的协方差 = 集合离散 + 观测噪声
        innov = z - hm
        nu = innov / np.sqrt(np.diag(Pzz))                    # 标准化预测误差（每路观测一个）
        loglik = float(-0.5 * (innov @ np.linalg.solve(Pzz, innov) + np.log(np.linalg.det(Pzz))
                               + len(z) * np.log(2 * np.pi)))

        # 卡尔曼更新（带扰动观测的随机 EnKF），在对数空间里做，保证人数为正
        keys = ["S", "E", "I", "R", "lm", "lg"] + (["b0", "b1", "b2", "lw"] if self.est_theta else [])
        X = np.array([np.log(x[k]) if k in "SEIR" else x[k] for k in keys])
        dX = X - X.mean(1, keepdims=True)
        K = (dX @ dh.T / (self.n - 1)) @ np.linalg.inv(Pzz)   # 卡尔曼增益
        zpert = z[:, None] + self.rng.normal(0, 1, (len(z), self.n)) * np.sqrt(r)[:, None]
        X = X + K @ (zpert - h)
        for i, k in enumerate(keys):
            x[k] = np.exp(X[i]) if k in "SEIR" else X[i]
        tot = x["S"] + x["E"] + x["I"] + x["R"]              # 总人口守恒
        for k in "SEIR":
            x[k] = x[k] * N / tot
        x["lm"] = np.clip(x["lm"], -1.6, 1.0)
        x["lg"] = np.clip(x["lg"], -0.7, 1.0)

        pred = float(np.exp(hm[0]) - 1.0)                     # 今天新增的一步预测值（更新前）
        self._advance(x, t, self.rng)
        return dict(nu=nu, loglik=loglik, pred=pred)

    # ------------------------------------------------------------ 预测未来 H 天的新增
    def forecast(self, t, H=HORIZON):
        """集合表示第 t 天开始时的状态；返回第 t..t+H-1 天新增的 (5%, 50%, 95%) 分位数。"""
        x = {k: v.copy() for k, v in self.x.items()}
        rng = np.random.default_rng(int(self.rng.integers(1 << 30)))
        out = np.zeros((H, 3))
        for h in range(H):
            mu = np.maximum(RHO * SIGMA * x["E"], 1e-6)
            ys = rng.negative_binomial(self.phi, self.phi / (self.phi + mu))
            out[h] = np.quantile(ys, [0.05, 0.5, 0.95])
            self._advance(x, t + h, rng)
        return out

    def mean(self, k):
        return float(np.mean(self.x[k]))
