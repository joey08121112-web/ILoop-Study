"""出图：python3 -m fluevo.plots [种子]  ->  out_flu/*.png"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .engine import Engine, PATH_CN
from .experiment import OUT, T0, TC, LOCKDOWN, lockdown_case, issue_errors, run
from .sim import simulate, log_beta_season, YEAR

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
COL = dict(frozen="#8a897f", tune="#eda100", wrong="#eb6834", right="#2a78d6")
NAME = dict(frozen="不进化（参数冻结）", tune="只微调（无变点模块）", wrong="进化·选错路径（康复率）", right="进化·选对路径（传播率）")
PCOL = dict(none="#c3c2b7", beta="#2a78d6", gamma="#eb6834", sus="#1baf7a")
plt.rcParams.update({"font.sans-serif": ["PingFang SC", "Heiti TC", "Arial Unicode MS"], "axes.unicode_minus": False,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
                     "axes.titlelocation": "left", "axes.titlesize": 12, "legend.frameon": False})


def d(t):
    """绝对天 -> 相对封城日的天数。"""
    return np.asarray(t) - TC


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def marks(ax, alarm, confirm=None):
    ax.axvline(0, color=INK2, lw=1, ls=":")
    ax.axvline(d(alarm), color=INK2, lw=1, ls="--")
    y = ax.get_ylim()[1]
    ax.annotate("封城", (0, y), xytext=(-3, -2), textcoords="offset points", ha="right", va="top", color=INK2, fontsize=9)
    ax.annotate("报警", (d(alarm), y), xytext=(3, -2), textcoords="offset points", ha="left", va="top", color=INK2, fontsize=9)


def fig_overview(data, engines, alarm):
    y = data["y"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ts = np.arange(TC - 60, TC + 100)
    for v in ("frozen", "tune", "wrong", "right"):
        log = {r["t"]: r for r in engines[v].log}
        ax.plot(d(ts), [log[t - 7]["fc"][6][1] for t in ts], color=COL[v], lw=2, label=NAME[v])
    ax.plot(d(ts), y[ts], ".", color=INK, ms=4, label="实际每日新增")
    ax.set_yscale("log")
    ax.set_ylim(30, 6000)
    ax.set_xlabel("相对封城日的天数")
    ax.set_ylabel("每日新增（对数刻度）")
    ax.set_title("7 天前做出的预测 对比 实际值")
    ax.legend(loc="upper left", ncol=1, fontsize=9)
    marks(ax, alarm)
    save(fig, "图1_全程对比.png")


def fig_fans(data, engines, alarm):
    y = data["y"]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ts = np.arange(TC - 20, TC + 40)
    ax.plot(d(ts), y[ts], ".", color=INK, ms=6, label="实际每日新增")
    pre = {r["t"]: r for r in engines["right"].log}[TC - 1]
    ax.plot(d(np.arange(TC, TC + 14)), np.array(pre["fc"])[:, 1], color=INK2, lw=2, ls="--", label="进化前：封城前一天做出的预测")
    for v in ("frozen", "wrong", "right"):
        r = {r["t"]: r for r in engines[v].log}[alarm]
        fc = np.array(r["fc"])
        xs = d(np.arange(alarm + 1, alarm + 15))
        ax.plot(xs, fc[:, 1], color=COL[v], lw=2.2, label="报警当天做出的预测：" + NAME[v])
        if v == "right":
            ax.fill_between(xs, fc[:, 0], fc[:, 2], color="#cde2fb", lw=0, zorder=0)
    ax.set_yscale("log")
    ax.set_xlabel("相对封城日的天数")
    ax.set_ylabel("每日新增（对数刻度）")
    ax.set_title("变点附近：进化前后对未来 14 天的预测")
    ax.legend(loc="upper left", fontsize=9)
    marks(ax, alarm)
    save(fig, "图2_变点前后预测.png")


def fig_errors(data, engines, alarm):
    fig, ax = plt.subplots(figsize=(11, 4))
    for v in ("frozen", "tune", "wrong", "right"):
        e = issue_errors(engines[v], data["y"], TC - 20, TC + 45)
        ax.plot(d(list(e)), [100 * a for a, _ in e.values()], color=COL[v], lw=2, label=NAME[v])
    ax.set_yscale("log")
    ax.set_xlabel("预测发布日（相对封城日的天数）")
    ax.set_ylabel("未来 7 天预测的平均相对误差 %")
    ax.set_title("每天发布的 7 天预测，事后看错了多少")
    ax.legend(loc="upper right", fontsize=9)
    marks(ax, alarm)
    save(fig, "图3_预测误差.png")


def fig_probs(data, pre, seed):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, use_s, title in zip(axes, (True, False), ("有搜索指数", "只有每日新增")):
        e = run(data, pre, seed, use_search=use_s, min_days=12, max_days=12, p_commit=2.0)
        rows = [(r["t"], r["probs"]) for r in e.log if r["probs"] and r["t"] <= TC + 25]
        if rows:
            ts = [t for t, _ in rows]
            for p in ("beta", "gamma", "sus", "none"):
                ax.plot(d(ts), [pr[p] for _, pr in rows], color=PCOL[p], lw=2.2, marker="o", ms=4, label=PATH_CN[p])
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(title)
        ax.set_xlabel("相对封城日的天数")
    axes[0].set_ylabel("各候选路径的概率")
    axes[0].legend(loc="center right", fontsize=9)
    fig.suptitle("报警之后，各候选进化路径的概率如何随新数据变化（真实原因：传播率下降）", x=0.01, ha="left", fontsize=12, color=INK)
    save(fig, "图4_路径概率.png")


def fig_monitor(engines, pre, alarm):
    e = engines["tune"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw=dict(width_ratios=[1, 1.5, 1.5]))
    nu = pre["nus"][365:, 0]
    axes[0].hist(nu, bins=40, density=True, color="#9ec5f4")
    xs = np.linspace(-4, 4, 200)
    axes[0].plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.5)
    axes[0].set_title("预训练年份里的标准化误差\n（黑线：标准正态）")
    ts = np.arange(TC - 40, TC + 20)
    log = {r["t"]: r for r in e.log}
    axes[1].bar(d(ts), [log[t]["nu"][0] for t in ts], color="#5598e7", width=0.8)
    for v in (-2, 2):
        axes[1].axhline(v, color=INK2, lw=1, ls="--")
    axes[1].axvline(0, color=INK2, lw=1, ls=":")
    axes[1].set_title("每天的标准化预测误差（病例）\n虚线 ±2：95% 的正常波动范围")
    axes[1].set_xlabel("相对封城日的天数")
    full = Engine(pre, seed=0, min_days=99, max_days=99, p_commit=2.0)
    full.human = None
    cus = []
    for t in range(T0, TC + 20):
        r = full.step(t, DATA["y"][t], DATA["s"][t])
        cus.append(r["cusum"])
        if full.cands is not None:
            break
    xs = d(np.arange(T0, T0 + len(cus)))
    keep = xs >= -40
    axes[2].plot(xs[keep], np.array(cus)[keep], color="#2a78d6", lw=2)
    axes[2].axhline(full.h, color="#d03b3b", lw=1.5)
    axes[2].axhline(full.h / 2, color="#eda100", lw=1.5, ls="--")
    axes[2].annotate(f"变点门槛 h={full.h}", (xs[keep][0], full.h), xytext=(2, 3), textcoords="offset points", color=INK2, fontsize=9)
    axes[2].annotate("疑似漂移 h/2", (xs[keep][0], full.h / 2), xytext=(2, 3), textcoords="offset points", color=INK2, fontsize=9)
    axes[2].axvline(0, color=INK2, lw=1, ls=":")
    axes[2].set_title("累计偏离（CUSUM）\n越过红线即判定为变点")
    axes[2].set_xlabel("相对封城日的天数")
    save(fig, "图5_误差门槛.png")


def fig_pretrain(data, pre):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.6, 1]))
    y = data["y"][:T0]
    axes[0].plot(np.arange(T0) / YEAR, y, color=INK, lw=0.8)
    axes[0].set_title("预训练用的前五年每日新增")
    axes[0].set_xlabel("年")
    days = np.arange(YEAR)
    th = pre["theta"]
    axes[1].plot(days, np.exp(log_beta_season(days)), color=INK2, lw=2, ls="--", label="真实")
    axes[1].plot(days, np.exp(log_beta_season(days, th["b0"], th["b1"], th["b2"])), color="#2a78d6", lw=2, label="预训练学到的")
    axes[1].set_title("季节传播率曲线")
    axes[1].set_xlabel("一年中的第几天")
    axes[1].legend()
    save(fig, "图6_预训练.png")


def main():
    global DATA
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    OUT.mkdir(exist_ok=True)
    DATA, pre, engines = lockdown_case(seed)
    alarm = [ev["alarm"] for ev in engines["right"].events if ev["alarm"] >= TC - 2][0]
    fig_overview(DATA, engines, alarm)
    fig_fans(DATA, engines, alarm)
    fig_errors(DATA, engines, alarm)
    fig_probs(DATA, pre, seed)
    fig_monitor(engines, pre, alarm)
    fig_pretrain(DATA, pre)
    for v, e in engines.items():
        print(v, [(ev["alarm"] - TC, ev["day"] - TC, ev["path"], ev["who"]) for ev in e.events],
              "beta x%.2f gamma x%.2f" % (e.log[-1]["beta_mult"], e.log[-1]["gamma_mult"]))
    print("图已写入", OUT)


if __name__ == "__main__":
    main()
