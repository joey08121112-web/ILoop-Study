"""流式自进化引擎：每天 预测 -> 等真值 -> 算标准化误差 -> 分级处理。

分级（门槛的依据见 cusum_threshold）：
  正常      只做卡尔曼滤波的常规微调
  疑似漂移  累计偏离超过一半门槛：当天把传播率的微调幅度放大
  变点      累计偏离超过门槛：为每条候选进化路径各开一个滤波器，逐日更新各路径的概率，
            由人确认；人不给信息时，自动取概率最高的路径
"""
import numpy as np

from .filter import Filter, HORIZON
from .sim import N

PATHS = {"beta": ("lm", 0.5), "gamma": ("lg", 0.4), "sus": ("S", 0.2)}     # 路径 -> (跳变的量, 跳变幅度的先验标准差)
PATH_CN = {"none": "没有变化（虚警）", "beta": "传播率变了", "gamma": "康复率变了", "sus": "易感人群变了"}
PRIOR = {"none": 0.1, "beta": 0.3, "gamma": 0.3, "sus": 0.3}
K_REF = 0.5


def cusum_arl(h, k=K_REF):
    """单侧 CUSUM 在“模型没问题”时平均多少天误报一次（Siegmund 近似）。"""
    b = h + 1.166
    return (np.exp(2 * k * b) - 2 * k * b - 1) / (2 * k * k)


def cusum_threshold(n_charts, years_per_false_alarm=5.0):
    """取最小的门槛 h，使 n_charts 张单侧图合起来平均 years_per_false_alarm 年才误报一次。"""
    h = 3.0
    while cusum_arl(h) / n_charts < years_per_false_alarm * 365:
        h += 0.1
    return round(h, 1)


class Engine:
    def __init__(self, pre, seed=0, use_search=True, evolve="full", human=None, prior=None,
                 min_days=3, max_days=10, p_commit=0.95, day0=0):
        """evolve: 'frozen' 参数冻结只更新状态 | 'tune' 只有常规微调 | 'full' 微调 + 变点处选路径
        human(t, probs) -> 路径名或 None：人工确认接口。返回 None 表示这天人没给信息。"""
        self.main = pre["filter"].copy(seed)
        if not use_search:
            self.main.c = None
        self.base_rw = dict(self.main.rw)
        if evolve == "frozen":                          # 参数完全冻结：不游走，集合里也不留离散度
            self.main.rw = dict(self.main.rw, lm=0.0, lg=0.0)
            for k in ("lm", "lg"):
                self.main.x[k] = np.full(self.main.n, self.main.mean(k))
        self.evolve, self.human = evolve, human
        self.min_days, self.max_days, self.p_commit = min_days, max_days, p_commit
        self.day0 = day0                                # 只影响文字里显示的天数
        self.prior = dict(prior or PRIOR)
        self.rng = np.random.default_rng(seed + 1)
        self.n_obs = 2 if use_search else 1
        self.h = cusum_threshold(2 * self.n_obs)
        self.cus = np.zeros((2, self.n_obs))            # [正向, 负向] x 观测路数
        self.zero_day = np.zeros((2, self.n_obs), int)  # 每张图上次归零是哪天
        self.cooldown = 0
        self.snap, self.obs, self.L0 = {}, {}, {}
        self.cands, self.L, self.alarm_day, self.onset = None, None, None, None
        self.log, self.events = [], []

    # ------------------------------------------------------------ 候选路径
    def _spawn(self, path, snap):
        f = snap.copy(int(self.rng.integers(1 << 30)))
        if path == "none":
            return f
        var, sd = PATHS[path]
        jump = self.rng.normal(0, sd, f.n)              # 跳变幅度的先验：方向和大小都不预设
        if var == "S":
            # 易感人群只允许突然增加（新毒株造成免疫逃逸）。突然减少在物理上不合理，
            # 而且它和“传播率下降”在方程里乘在一起、无法区分，所以不作为候选
            f.x["S"] = np.minimum(f.x["S"] * np.exp(np.abs(jump)), 0.98 * N)
            f.x["R"] = np.maximum(N - f.x["S"] - f.x["E"] - f.x["I"], 1.0)
            f.rw = dict(f.rw, lm=0.0, lg=0.0)
        else:
            f.x[var] = f.x[var] + jump
            f.rw = dict(f.rw, lm=0.0, lg=0.0) | {var: 0.01}   # 只放开这条路径对应的参数
        return f

    def _open(self, t, chart):
        """变点报警。变点到底从哪天开始并不确定，所以每条路径都从多个可能的起点各重放一遍，
        路径的证据取这些起点的平均（对起点做边际化）。"""
        first = int(max(self.zero_day[chart] - 3, t - 30, min(self.snap)))
        onsets = list(range(first, t, max(1, (t - first) // 8)))
        self.alarm_day, self.onset = t, int(self.zero_day[chart] + 1)
        self.cands, self.L = {}, {}
        for p in self.prior:
            for o in (onsets if p != "none" else onsets[:1]):
                f = self._spawn(p, self.snap[o])
                L = sum(self.L0[d] for d in range(first, o))        # 起点之前的日子按“没变”计分，保证各起点可比
                for d in range(o, t + 1):
                    L += f.step(d, *self.obs[d])["loglik"]
                self.cands[(p, o)], self.L[(p, o)] = f, L

    def _path_logev(self):
        out = {}
        for p in self.prior:
            Ls = np.array([L for (q, o), L in self.L.items() if q == p])
            out[p] = float(Ls.max() + np.log(np.mean(np.exp(Ls - Ls.max()))))
        return out

    def _probs(self):
        ev = self._path_logev()
        lw = np.array([np.log(self.prior[p]) + ev[p] for p in self.prior])
        w = np.exp(lw - lw.max())
        return dict(zip(self.prior, w / w.sum()))

    def _best(self, path):
        """某条路径下证据最强的那个起点对应的滤波器。"""
        key = max((k for k in self.L if k[0] == path), key=lambda k: self.L[k])
        return key, self.cands[key]

    def _commit(self, t, path, who):
        probs = self._probs()
        key, f = self._best(path)
        self.events.append(dict(day=t, alarm=self.alarm_day, onset=key[1], path=path, who=who, probs=probs))
        if path != "none":
            self.main = f
        self.main.rw = dict(self.base_rw)
        self.cands = None
        self.cus[:] = 0
        self.zero_day[:] = t
        self.cooldown = 14
        return f"{who}确认：{PATH_CN[path]}，始于第{key[1] - self.day0}天（当时概率 {probs[path]:.0%}）"

    # ------------------------------------------------------------ 每日一步
    def step(self, t, y, s=None):
        self.snap[t] = self.main.copy(int(self.rng.integers(1 << 30)))
        self.snap.pop(t - 45, None)
        self.obs[t] = (y, s)
        out = self.main.step(t, y, s)
        self.L0[t] = out["loglik"]
        nu = out["nu"]
        rec = dict(t=t, y=float(y), pred=out["pred"], nu=nu.tolist(), tier="正常", probs=None, action=None)

        if self.evolve == "full":
            if self.cands is not None:
                for key, f in self.cands.items():
                    self.L[key] += f.step(t, y, s)["loglik"]
            elif self.cooldown > 0:
                self.cooldown -= 1
            else:
                self.cus[0] = np.maximum(0, self.cus[0] + nu - K_REF)
                self.cus[1] = np.maximum(0, self.cus[1] - nu - K_REF)
                self.zero_day[self.cus == 0] = t
                top = float(self.cus.max())
                if top > self.h:
                    chart = np.unravel_index(self.cus.argmax(), self.cus.shape)
                    self._open(t, chart)
                    rec["action"] = f"检测到变点（估计始于第{self.onset - self.day0}天），列出候选路径"
                elif top > self.h / 2:
                    rec["tier"] = "疑似漂移"
                    self.main.x["lm"] += self.rng.normal(0, 3 * self.base_rw["lm"], self.main.n)

            if self.cands is not None:
                rec["tier"] = "变点"
                probs = self._probs()
                rec["probs"] = probs
                best = max(probs, key=probs.get)
                choice = self.human(t, probs) if self.human else None
                days = t - self.alarm_day
                if choice is not None:
                    rec["action"] = (rec["action"] + "；" if rec["action"] else "") + self._commit(t, choice, "人工")
                elif (days >= self.min_days and probs[best] >= self.p_commit) or days >= self.max_days:
                    rec["action"] = self._commit(t, best, "自动")
        rec["cusum"] = float(self.cus.max())

        # 预测：变点未确认期间，暂用当前概率最高的路径
        src = self.main
        if self.cands is not None:
            probs = self._probs()
            src = self._best(max(probs, key=probs.get))[1]
        rec["fc"] = src.forecast(t + 1, 14).tolist()        # 第 t+1 .. t+14 天（前 7 天是主要考核）
        rec["beta_mult"] = float(np.exp(src.mean("lm")))
        rec["gamma_mult"] = float(np.exp(src.mean("lg")))
        rec["S_frac"] = src.mean("S") / N
        if self.cands is not None:                          # 每条候选路径各自的预测，画图用
            rec["fc_paths"] = {p: self._best(p)[1].forecast(t + 1, 14)[:, 1].tolist() for p in self.prior}
        self.log.append(rec)
        return rec
