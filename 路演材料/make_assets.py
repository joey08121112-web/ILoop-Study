"""生成 PPT 用的素材：公式图、结果图、封面曲线，以及结果数字。在项目根目录运行：python3 路演材料/make_assets.py"""
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fluevo.experiment import T0, TC
from fluevo.showcase import run_case, MAIN_DELAY, H

A = Path(__file__).resolve().parent / "assets"
INK, MUTED, TEAL, ORANGE, GREY = "#16302B", "#5B6B68", "#0E7C7B", "#E4572E", "#8A8F8E"
plt.rcParams.update({"font.sans-serif": ["PingFang SC", "Heiti TC", "Arial Unicode MS", "Microsoft YaHei"],
                     "axes.unicode_minus": False, "mathtext.fontset": "cm"})


def formula(lines, name, size=22, width=6.0, gap=0.62):
    fig = plt.figure(figsize=(width, gap * len(lines) + 0.15))
    for i, s in enumerate(lines):
        fig.text(0.0, 1 - (i + 0.5) / len(lines), s, fontsize=size, color=INK, va="center", ha="left")
    fig.savefig(A / name, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


formula([r"$\dfrac{dS}{dt}=-\beta(t)\,\dfrac{SI}{N}+\omega R$",
         r"$\dfrac{dE}{dt}=\beta(t)\,\dfrac{SI}{N}-\sigma E$",
         r"$\dfrac{dI}{dt}=\sigma E-\gamma(t)\, I$",
         r"$\dfrac{dR}{dt}=\gamma(t)\, I-\omega R$"], "eq_seirs.png", width=5.2, gap=1.0)
formula([r"$\log\beta(t)=b_0+b_1\cos\dfrac{2\pi t}{365}+b_2\sin\dfrac{2\pi t}{365}+m_t$",
         r"$\gamma(t)=\gamma_0\,e^{\,g_t}$",
         r"$m_{t+1}=m_t+\varepsilon_t+J_t$"], "eq_param.png", size=19, width=6.6)
formula([r"$x_{t+1}=f(x_t)+w_t,\quad x_t=[S,E,I,R,\,m,\,g]$",
         r"$\log(y_t+1)=\log(\rho\,\sigma E_t+1)+v_t$",
         r"$\log s_t=\log(c\,I_t)+u_t$"], "eq_obs.png", size=19, width=6.6)

# 结果图：数据组 1，变点附近
rows = [run_case(s, delays=(MAIN_DELAY,)) for s in (1, 2, 3)]
r = rows[0]
sh = r["shots"][MAIN_DELAY]
y = r["data"]["y"][T0:T0 + 365]
tc, og, al = TC - T0, sh["origin"] - T0, r["alarm"] - T0
lo, hi = tc - 28, og + H
f = np.arange(og + 1, og + 1 + H)
fig, ax = plt.subplots(figsize=(8.2, 5.2))
fit = np.array([q["pred"] for q in r["eng"]["new"].log])
ax.plot(np.arange(lo, al + 1), fit[lo:al + 1], color=MUTED, lw=2)
ax.fill_between(f, sh["fc"]["new"][:, 0], sh["fc"]["new"][:, 2], color="#BFE3DF", lw=0)
ax.plot(f, sh["fc"]["old"][:, 1], color=GREY, lw=3, ls="--")
ax.plot(f, sh["fc"]["wrong"][:, 1], color=ORANGE, lw=3, ls="-.")
ax.plot(f, sh["fc"]["new"][:, 1], color=TEAL, lw=3.4)
ax.plot(np.arange(lo, hi + 1), y[lo:hi + 1], "o", color=INK, ms=4.5)
ax.set_yscale("log")
ax.set_ylim(90, 2600)
ax.axvline(tc, color=MUTED, lw=1.2, ls=":")
ax.text(tc - 0.6, 2300, "封城（变点）", ha="right", va="top", color=MUTED, fontsize=12)
ax.text(f[-1], sh["fc"]["old"][-1, 1] * 1.12, "不进化", ha="right", va="bottom", color=GREY, fontsize=13, weight="bold")
ax.text(f[0] + 0.3, sh["fc"]["wrong"][0, 1] * 1.12, "进化·选错路径", ha="left", va="bottom", color=ORANGE, fontsize=13, weight="bold")
ax.text(f[-1], sh["fc"]["new"][-1, 1] * 0.74, "进化·选对路径", ha="right", va="top", color=TEAL, fontsize=13, weight="bold")
ax.text(lo + 1, y[lo + 2] * 0.62, "真值（每日新增）", color=INK, fontsize=12)
ax.set_xlabel("今年的第几天", color=MUTED, fontsize=12)
ax.set_ylabel("每日新增病例（对数刻度）", color=MUTED, fontsize=12)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color("#C9D1CF")
ax.tick_params(colors=MUTED, labelsize=11)
ax.grid(color="#E6EBEA", lw=0.7)
fig.tight_layout()
fig.savefig(A / "result.png", dpi=220, transparent=True)
plt.close(fig)

# 封面用的装饰曲线：今年的每日新增
fig, ax = plt.subplots(figsize=(10, 2.2))
ax.plot(np.arange(365), y, color="#3DDBD9", lw=2)
ax.axvline(tc, color="#FFFFFF", lw=1, ls=":")
ax.axis("off")
fig.savefig(A / "cover_curve.png", dpi=200, transparent=True, bbox_inches="tight", pad_inches=0)
plt.close(fig)

avg = {k: {h: float(np.mean([q["shots"][MAIN_DELAY]["mape"][k][h] for q in rows])) for h in (7, 14, 21)}
       for k in ("old", "wrong", "new")}
stats = dict(avg=avg, alarm=[q["alarm"] - T0 for q in rows], onset=[q["event"]["onset"] - T0 for q in rows], tc=tc,
             group1={k: sh["mape"][k] for k in ("old", "wrong", "new")})
(A / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
print(json.dumps(stats, ensure_ascii=False))
