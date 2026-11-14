"""变点附近的自进化效果指标：python3 -m fluevo.metrics  ->  out_flu/变点指标.md + 图7_变点MAPE.png

MAPE = 预测中位数与实际值之差的绝对值 / 实际值，对所考察的天数取平均（实际值小于 10 时按 10 计，避免淡季被零除）。
所有做法在同一天、用同样的数据发布预测，区别只在参数怎么进化。
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import OUT, T0, TC, lockdown_case, run

NAMES = {"frozen": "不进化（参数冻结）", "tune": "只微调（无变点模块）", "wrong": "进化·人工选错路径",
         "auto_cases": "进化·自动（只有每日新增）", "auto_search": "进化·自动（有搜索指数）", "right": "进化·人工选对路径"}
COL = {"frozen": "#8a897f", "tune": "#eda100", "wrong": "#eb6834", "auto_cases": "#e87ba4",
       "auto_search": "#1baf7a", "right": "#2a78d6"}
WINDOWS = ["稳定期\n(变点前 28–8 天发布)", "冲击期\n(变点前 7 天至报警前)", "进化后第 1 周\n(报警后 0–6 天发布)",
           "进化后第 2–3 周\n(报警后 7–20 天发布)"]


def mape7(e, y, days):
    """这些发布日做出的未来 7 天预测的 MAPE。"""
    log = {r["t"]: r for r in e.log}
    errs = []
    for t in days:
        med = np.array(log[t]["fc"])[:7, 1]
        act = y[t + 1:t + 8]
        errs.append(np.mean(np.abs(med - act) / np.maximum(act, 10)))
    return float(np.mean(errs)) if errs else np.nan


def main(seeds=range(1, 21)):
    OUT.mkdir(exist_ok=True)
    table = {v: [] for v in NAMES}
    before_after, used = [], 0
    for seed in seeds:
        data, pre, eng = lockdown_case(seed)
        y = data["y"]
        near = [ev["alarm"] for ev in eng["right"].events if ev["alarm"] >= TC - 2]
        if not near or not any(ev["who"] == "人工" and ev["alarm"] == near[0] for ev in eng["wrong"].events):
            continue
        alarm = near[0]
        used += 1
        eng["auto_search"] = run(data, pre, seed)
        eng["auto_cases"] = run(data, pre, seed, use_search=False)
        wins = [range(TC - 28, TC - 7), range(TC - 7, alarm), range(alarm, alarm + 7), range(alarm + 7, alarm + 21)]
        for v in NAMES:
            table[v].append([mape7(eng[v], y, w) for w in wins])
        # 同一批目标日（报警后 1–7 天）：进化前的预测（封城前一天发布）对 进化后的预测（报警当天发布）
        log = {r["t"]: r for r in eng["right"].log}
        act = y[alarm + 1:alarm + 8]
        old = np.array(log[TC - 1]["fc"])[:, 1]                 # 覆盖 TC .. TC+13
        idx = np.arange(alarm + 1, alarm + 8) - TC
        if idx.max() < 14:
            new = np.array(log[alarm]["fc"])[:7, 1]
            before_after.append((float(np.mean(np.abs(old[idx] - act) / np.maximum(act, 10))),
                                 float(np.mean(np.abs(new - act) / np.maximum(act, 10)))))
    mean = {v: np.nanmean(np.array(table[v]), axis=0) for v in NAMES}
    ba = np.array(before_after)

    lines = [f"# 变点附近的自进化效果（封城：传播率 ×0.6；{used} 条仿真数据流的平均）", "",
             "MAPE = |预测中位数 − 实际值| / 实际值 的平均，针对“未来 7 天”的预测；越小越好。", "",
             "## 一、同一批目标日，进化前后对比", "",
             f"目标日是报警后的第 1–7 天。{len(ba)} 条数据流的平均：", "",
             "| | MAPE |", "|---|---|",
             f"| 进化前：封城前一天做出的预测 | {ba[:, 0].mean():.1%} |",
             f"| 进化后：报警当天、选对路径后做出的预测 | {ba[:, 1].mean():.1%} |",
             f"| 误差缩小到原来的 | {ba[:, 1].mean() / ba[:, 0].mean():.1%} |", "",
             "## 二、按预测发布时间分段的 MAPE", "",
             "| 做法 | " + " | ".join(w.replace("\n", " ") for w in WINDOWS) + " |", "|---|---|---|---|---|"]
    for v in NAMES:
        lines.append(f"| {NAMES[v]} | " + " | ".join(f"{x:.1%}" for x in mean[v]) + " |")
    lines += ["", "说明：冲击期里所有做法都差，因为这些预测发布时变化还没发生或刚发生，没有任何方法能提前知道；"
              "自进化的效果体现在后两列恢复得多快、多彻底。"]
    (OUT / "变点指标.md").write_text("\n".join(lines))
    (OUT / "变点指标.json").write_text(json.dumps(dict(n=used, windows=[w.replace("\n", " ") for w in WINDOWS],
                                                    mape={v: mean[v].tolist() for v in NAMES},
                                                    before=float(ba[:, 0].mean()), after=float(ba[:, 1].mean()),
                                                    n_before_after=len(ba)), ensure_ascii=False, indent=2))
    print("\n".join(lines))

    plt.rcParams.update({"font.sans-serif": ["PingFang SC", "Heiti TC", "Arial Unicode MS"], "axes.unicode_minus": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#e6e5e0",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.spines.top": False,
                         "axes.spines.right": False, "font.size": 10, "legend.frameon": False})
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw=dict(width_ratios=[1, 3.2]))
    b = ax[0].bar(["进化前\n(封城前一天的预测)", "进化后\n(报警当天的预测)"], 100 * ba.mean(0), width=0.5, color=["#8a897f", "#2a78d6"])
    ax[0].bar_label(b, [f"{v:.0f}%" for v in 100 * ba.mean(0)], padding=3, color="#0b0b0b", fontsize=12)
    ax[0].set_title("同一批目标日（报警后 1–7 天）的 MAPE", loc="left", fontsize=11)
    ax[0].set_ylim(0, 100 * ba.mean(0).max() * 1.18)
    ax[0].set_yticks([])
    w = 0.14
    for i, v in enumerate(NAMES):
        xs = np.arange(4) + (i - 2.5) * w
        bars = ax[1].bar(xs, 100 * mean[v], width=w * 0.9, color=COL[v], label=NAMES[v])
        ax[1].bar_label(bars, [f"{x:.0f}" for x in 100 * mean[v]], padding=2, color="#0b0b0b", fontsize=8)
    ax[1].set_yscale("log")
    ax[1].set_ylim(3, 1500)
    ax[1].set_xticks(np.arange(4), WINDOWS)
    ax[1].set_ylabel("7 天预测的 MAPE %（对数刻度）", color="#52514e")
    ax[1].set_title(f"按预测发布时间分段的 MAPE（{used} 条仿真数据流平均，柱上数字为 %）", loc="left", fontsize=11)
    ax[1].legend(loc="upper left", ncol=2, fontsize=9)
    ax[1].grid(axis="y", color="#e6e5e0", lw=0.6)
    ax[1].set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT / "图7_变点MAPE.png", dpi=150)


if __name__ == "__main__":
    main()
