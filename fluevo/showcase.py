"""给评委看的单例展示：python3 -m fluevo.showcase [种子 ...]   默认种子 1 2 3

每个种子是一组“今年”的 365 天数据，中间第 195 天封城（传播率 ×0.6）。每组出一张主图：
  (a) 今年的每日新增
  (b) 传播率：真值 对 三种做法的估计
  (c) 变点附近：同一天出发、用同样的数据，对未来 21 天的预测，和随后到达的真值
三种做法：
  不进化          参数冻结，只更新状态
  进化·选错路径   报警当天被告知“康复率变了”（真实原因是传播率）
  进化·选对路径   报警当天被告知“传播率变了”
主图从报警后第 3 天出发；另出一张报警当天出发的图作对照。每次运行都会重新计算并覆盖图和指标表；
只有预训练结果读缓存（out_flu/pretrain_seed*.pkl），删掉缓存文件即可连预训练一起重跑。
"""
import pickle
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .engine import Engine
from .experiment import OUT, T0, TC, LOCKDOWN, once
from .pretrain import pretrain
from .sim import simulate, log_beta_season, YEAR

H = 21
MAIN_DELAY = 3
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
KEYS = ("old", "wrong", "new")
NAME = {"old": "不进化", "wrong": "进化·选错路径", "new": "进化·选对路径"}
LONG = {"old": "不进化：沿用变点前的参数", "wrong": "进化·选错路径：当成“康复率变了”", "new": "进化·选对路径：确认“传播率变了”"}
COL = {"old": "#8a897f", "wrong": "#eb6834", "new": "#2a78d6"}
STYLE = {"old": "--", "wrong": "-.", "new": "-"}


def get_pretrain(seed, data):
    f = OUT / f"pretrain_seed{seed}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes()), True
    pre = pretrain(data["y"][:T0], data["s"][:T0], seed=seed)
    f.write_bytes(pickle.dumps(pre))
    return pre, False


def mape(pred, act):
    return float(np.mean(np.abs(pred - act) / np.maximum(act, 10)))


def run_case(seed, delays=(0, MAIN_DELAY)):
    data = simulate(seed, change=LOCKDOWN)
    pre, cached = get_pretrain(seed, data)
    y = data["y"]
    eng = {"old": Engine(pre, seed=seed, evolve="frozen", day0=T0),
           "wrong": Engine(pre, seed=seed, human=once("gamma"), day0=T0),
           "new": Engine(pre, seed=seed, human=once("beta"), day0=T0)}
    alarm, shots = None, {}
    for t in range(T0, T0 + YEAR):
        for e in eng.values():
            e.step(t, y[t], data["s"][t])
        if alarm is None and any(ev["alarm"] >= TC - 2 for ev in eng["new"].events):
            alarm = t                                                # 报警并人工确认的那天
        if alarm is not None and t - alarm in delays:                # 三种做法都从这天出发预测
            fc = {k: e.main.forecast(t + 1, H) for k, e in eng.items()}
            act = y[t + 1:t + 1 + H]
            shots[t - alarm] = dict(origin=t, fc=fc, act=act,
                                    mape={k: {h: mape(fc[k][:h, 1], act[:h]) for h in (7, 14, 21)} for k in KEYS})
    th = pre["theta"]
    season = np.exp(log_beta_season(np.arange(T0, T0 + YEAR), th["b0"], th["b1"], th["b2"]))
    ev = [e for e in eng["new"].events if e["alarm"] >= TC - 2][0]
    at = alarm + MAIN_DELAY - T0                                     # 主图出发那天各自的参数估计
    return dict(seed=seed, data=data, eng=eng, alarm=alarm, shots=shots, season=season, cached=cached, event=ev,
                beta20={k: eng[k].log[at]["beta_mult"] for k in KEYS}, gamma20={k: eng[k].log[at]["gamma_mult"] for k in KEYS},
                wrong_fix=[e for e in eng["wrong"].events if e["alarm"] > alarm])


def figure(r, delay, path, keys=KEYS, tag=""):
    """keys 决定画哪几种做法；tag 是加在标题前的说明（如“第一次运行”）。"""
    y = r["data"]["y"][T0:T0 + YEAR]
    days = np.arange(YEAR)
    sh = r["shots"][delay]
    tc, og, al = TC - T0, sh["origin"] - T0, r["alarm"] - T0
    fig, ax = plt.subplots(3, 1, figsize=(11, 11.8), height_ratios=[1, 1, 2])

    ax[0].plot(days, y, color=INK, lw=1.2)
    ax[0].set_title(f"{tag}(a) 今年的每日新增病例（365 天，数据组 {r['seed']}）")
    ax[0].set_ylabel("每日新增")

    ax[1].plot(days, r["data"]["truth"]["beta"][T0:T0 + YEAR], color=INK, lw=2.2, label="真值")
    for k in keys:
        ax[1].plot(days, r["season"] * np.array([q["beta_mult"] for q in r["eng"][k].log]), color=COL[k], lw=1.8,
                   ls=STYLE[k], label=NAME[k] + "的估计")
    ax[1].set_title("(b) 传播率 β：封城当天真值骤降")
    ax[1].set_ylabel("传播率 β")
    ax[1].set_xlabel("今年的第几天")
    ax[1].legend(loc="lower left", ncol=4, fontsize=9)
    for a in ax[:2]:
        a.axvline(tc, color=INK2, lw=1, ls=":")
        a.axvspan(tc - 30, og + H, color="#f0efec", zorder=0)
    ax[0].annotate("封城", (tc, ax[0].get_ylim()[1]), xytext=(4, -4), textcoords="offset points", va="top", color=INK2)
    ax[0].annotate("下图放大的区域", (tc - 30, ax[0].get_ylim()[1]), xytext=(-4, -4), textcoords="offset points",
                   va="top", ha="right", color=INK2, fontsize=9)

    a = ax[2]
    lo, hi = tc - 30, og + H
    f = np.arange(og + 1, og + 1 + H)
    fit = np.array([q["pred"] for q in r["eng"]["new"].log])
    a.plot(np.arange(lo, al + 1), fit[lo:al + 1], color=INK2, lw=2, label="报警之前：模型逐日的一步预测")
    if "new" in keys:
        a.fill_between(f, sh["fc"]["new"][:, 0], sh["fc"]["new"][:, 2], color="#cde2fb", lw=0, label="选对路径预测的 90% 区间")
    for k in keys:
        a.plot(f, sh["fc"][k][:, 1], color=COL[k], lw=2.6, ls=STYLE[k], label=LONG[k])
    a.plot(np.arange(lo, hi + 1), y[lo:hi + 1], "o", color=INK, ms=4.5, label="真值（每日新增）")
    a.set_yscale("log")
    a.set_ylim(top=a.get_ylim()[1] * 2.2)
    a.axvline(tc, color=INK2, lw=1, ls=":")
    a.axvline(og, color=INK2, lw=1, ls="--")
    top = a.get_ylim()[1]
    a.annotate("封城", (tc, top), xytext=(-4, -4), textcoords="offset points", va="top", ha="right", color=INK2)
    word = "报警" if keys == ("old",) else "报警并确认"
    note = f"第 {og} 天{word}\n预测从这天出发" if delay == 0 else f"第 {al} 天{word}\n第 {og} 天（{delay} 天后）预测出发"
    a.annotate(note, (og, top), xytext=(4, -4), textcoords="offset points", va="top", color=INK2)
    m = sh["mape"]
    short = {"old": "不进化", "wrong": "选错路径", "new": "选对路径"}
    txt = "未来预测的 MAPE   " + "".join(f"{short[k]:>7}" for k in keys) + "\n" + "\n".join(
        f"  未来 {h:2d} 天     " + "".join(f"{m[k][h]:>10.0%}" for k in keys) for h in (7, 14, 21))
    a.text(0.012, 0.03, txt, transform=a.transAxes, va="bottom", ha="left", fontsize=11, color=INK,
           bbox=dict(boxstyle="round,pad=0.5", fc=SURF, ec=GRID))
    a.set_title("(c) 变点附近：同一天出发、用同样的数据，对未来 21 天的预测（对数纵轴）")
    a.set_xlabel("今年的第几天")
    a.set_ylabel("每日新增")
    a.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def table(rows, delay):
    L = ["| 数据组 | 预测步长 | 不进化 | 进化·选错路径 | 进化·选对路径 |", "|---|---|---|---|---|"]
    for r in rows:
        for h in (7, 14, 21):
            m = r["shots"][delay]["mape"]
            L.append(f"| {r['seed']} | 未来 {h} 天 | {m['old'][h]:.0%} | {m['wrong'][h]:.0%} | {m['new'][h]:.0%} |")
    if len(rows) > 1:
        for h in (7, 14, 21):
            L.append(f"| **平均** | 未来 {h} 天 | " + " | ".join(
                f"**{np.mean([r['shots'][delay]['mape'][k][h] for r in rows]):.0%}**" for k in KEYS) + " |")
    return L


def build(seeds=(1, 2, 3), out=OUT, say=print):
    """跑完三种做法、出图、写指标表。out 是输出文件夹；预训练缓存始终放在项目的 out_flu/ 里。"""
    out.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({"font.sans-serif": ["PingFang SC", "Heiti TC", "Arial Unicode MS"], "axes.unicode_minus": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "font.size": 10,
                         "axes.titlelocation": "left", "axes.titlesize": 12, "legend.frameon": False})
    for old in list(out.glob("展示_数据组*.png")) + list((out / "对照_报警当天出发").glob("*.png")):
        old.unlink()                                  # 每次都重新输出，不留旧图
    alt = out / "对照_报警当天出发"
    alt.mkdir(exist_ok=True)
    rows = []
    for seed in seeds:
        r = run_case(seed)
        figure(r, MAIN_DELAY, out / f"展示_数据组{seed}.png")
        figure(r, 0, alt / f"展示_数据组{seed}_报警当天.png")
        rows.append(r)
        ev = r["event"]
        say(f"数据组 {seed}: 预训练{'读缓存' if r['cached'] else '已计算并缓存'}；第 {ev['alarm'] - T0} 天报警，"
              f"估计变点在第 {ev['onset'] - T0} 天（真值第 {TC - T0} 天）")
    L = ["# 单例展示：三组数据在变点处的进化效果", "",
         f"每组是今年的 365 天数据，第 {TC - T0} 天封城（传播率 ×0.6，康复率不变）。三种做法在同一天出发、用同样的数据"
         "（病例 + 搜索指数），对未来 21 天做预测。MAPE = |预测中位数 − 真值| / 真值 的平均，越小越好。", "",
         "- **不进化**：参数冻结，只更新状态。",
         "- **进化·选错路径**：报警当天被告知“康复率变了”。",
         "- **进化·选对路径**：报警当天被告知“传播率变了”。", "",
         "## 检测与参数估计", "",
         f"| 数据组 | 报警日 | 估计的变点日 | 报警后第 {MAIN_DELAY} 天的参数估计：选对路径 | 选错路径 | 选错后系统是否再次报警纠正 |", "|---|---|---|---|---|---|"]
    for r in rows:
        ev = r["event"]
        fix = "；".join(f"第 {e['alarm'] - T0} 天再报警，改判“{dict(beta='传播率', gamma='康复率', sus='易感人群', none='虚警')[e['path']]}”"
                       for e in r["wrong_fix"][:1]) or "否"
        L.append(f"| {r['seed']} | 第 {ev['alarm'] - T0} 天 | 第 {ev['onset'] - T0} 天 | "
                 f"传播率 ×{r['beta20']['new']:.2f}，康复率 ×{r['gamma20']['new']:.2f} | "
                 f"传播率 ×{r['beta20']['wrong']:.2f}，康复率 ×{r['gamma20']['wrong']:.2f} | {fix} |")
    L += ["", "真值：变点在第 195 天，传播率 ×0.60，康复率 ×1.00。", "",
          f"## 主结果：报警后第 {MAIN_DELAY} 天出发的预测（对应主图 `展示_数据组N.png`）", ""] + table(rows, MAIN_DELAY)
    L += ["", "## 对照：报警当天出发的预测（对应 `对照_报警当天出发/` 里的图）", "",
          "报警当天，变点之后只有 3–5 天数据，参数刚改，当前感染人数的估计还没完全跟上，所以误差比上表大。", ""] + table(rows, 0)
    L += ["", "## 读数时请注意", "",
          "- “选错路径”这条基线是在带搜索指数的条件下得到的。只用每日新增时，选错路径对短期病例预测的影响很小，"
          "主要后果是把原因解释错（见 `变点指标.md` 第三节）。",
          "- 预测区间由随机抽样得到，重跑时数字会有一两个百分点的浮动。",
          "- 全部是仿真数据。"]
    (out / "展示指标.md").write_text("\n".join(L) + "\n")
    return rows, L


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [1, 2, 3]
    _, L = build(seeds)
    print("\n" + "\n".join(L))


if __name__ == "__main__":
    main()
