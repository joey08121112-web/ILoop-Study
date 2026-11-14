"""生成项目 X 型展架海报：80cm × 180cm（150 DPI -> 4724 × 10630 px，.jpg）。

用法：
    python3 路演材料/build_poster.py

输出：
    路演材料/日新_X型展架_80x180cm.jpg
"""
from __future__ import annotations

import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fluevo.engine import Engine
from fluevo.experiment import T0, TC, lockdown_case
from fluevo.showcase import H, MAIN_DELAY, run_case
from fluevo.sim import YEAR

OUT_DIR = Path(__file__).resolve().parent
OUT_JPG = OUT_DIR / "日新_X型展架_80x180cm.jpg"
CHROME_BIN = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")

# 80cm x 180cm @ 150 DPI
TARGET_DPI = 150
TARGET_W = round(80.0 / 2.54 * TARGET_DPI)   # 4724 px
TARGET_H = round(180.0 / 2.54 * TARGET_DPI)  # 10630 px
SCALE = 3
CSS_W = round(TARGET_W / SCALE)              # 1575 px
CSS_H = round(TARGET_H / SCALE)              # 3543 px

# 品牌配色（继承自 路演材料/build_ppt.py）
DARK = "#0B3A42"
TEAL = "#0E7C7B"
MINT = "#3DDBD9"
ORANGE = "#E4572E"
AMBER = "#D97706"
BLUE = "#1D4ED8"
GREY = "#8A8F8E"
INK = "#16302B"
MUTED = "#5B6B68"
GRID = "#E3ECEB"


def _setup_mpl() -> None:
    plt.rcParams.update({
        "font.sans-serif": ["Heiti SC", "STHeiti", "Arial Unicode MS"],
        "axes.unicode_minus": False,
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "axes.edgecolor": "#C6D7D5",
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.9,
        "font.size": 12.5,
        "axes.titlesize": 15.0,
        "axes.titleweight": "bold",
        "legend.frameon": False,
    })


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build_charts(tmp: Path) -> tuple[Path, Path]:
    """用 fluevo 真实仿真数据生成两张展架专属高清图表。"""
    _setup_mpl()
    results = json.loads((ROOT / "out_flu" / "results.json").read_text())

    # 1. CUSUM 检测 + 20 种子物理旁路路径辨识对比
    data2, pre2, _ = lockdown_case(2)
    full = Engine(pre2, seed=0, min_days=99, max_days=99, p_commit=2.0)
    full.human = None
    cus_t, cus_v = [], []
    for t in range(T0, TC + 18):
        res = full.step(t, data2["y"][t], data2["s"][t])
        cus_t.append(t - TC)
        cus_v.append(res["cusum"])
        if full.cands is not None:
            break
    cus_t_arr = np.asarray(cus_t)
    cus_v_arr = np.asarray(cus_v)

    sel = results["selection"]
    scenarios = [
        ("传播率骤降\n(β × 0.6)", "beta", 0.6),
        ("康复加快\n(γ × 1.6)", "gamma", 1.6),
        ("传播率上升\n(β × 1.4)", "beta", 1.4),
    ]
    acc_no = [100 * next(x["acc"] for x in sel if x["kind"] == knd and x["k"] == kv and not x["search"])
              for _, knd, kv in scenarios]
    acc_yes = [100 * next(x["acc"] for x in sel if x["kind"] == knd and x["k"] == kv and x["search"])
               for _, knd, kv in scenarios]

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 3.8), dpi=300, gridspec_kw=dict(width_ratios=[1.12, 1.0]))

    mask = cus_t_arr >= -24
    ax1.plot(cus_t_arr[mask], cus_v_arr[mask], color=TEAL, lw=3.2)
    ax1.plot(cus_t_arr[-1], cus_v_arr[-1], "o", color=ORANGE, ms=10, zorder=5)
    ax1.axhline(full.h, color=ORANGE, lw=2.2, ls="-")
    ax1.axhline(full.h / 2, color=AMBER, lw=2.0, ls="--")
    ax1.axvline(0, color=DARK, lw=1.6, ls=":")

    ax1.text(-23.5, full.h + 0.25, f"变点报警阈值 h = {full.h}（误报仅 0.05 次/年）",
             color=ORANGE, fontsize=11.5, fontweight="bold", va="bottom")
    ax1.text(-23.5, full.h / 2 + 0.22, f"疑似漂移线 h/2 = {full.h/2:.2f}（放宽过程噪声试探）",
             color=AMBER, fontsize=11.5, fontweight="bold", va="bottom")
    ax1.annotate("真实机制突变 (t = 0)", xy=(0, 5.2), xytext=(-14.5, 5.3),
                 arrowprops=dict(arrowstyle="->", color=DARK, lw=1.6),
                 fontsize=11.5, fontweight="bold", color=DARK, va="center",
                 bbox=dict(boxstyle="round,pad=0.25", fc="#FFFFFF", ec="#D5E3E1", lw=1))
    ax1.annotate(f"第 +{cus_t_arr[-1]} 天越线报警！\n(20组平均响应 3.55 天)",
                 xy=(cus_t_arr[-1], cus_v_arr[-1]), xytext=(-14.5, 9.2),
                 arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.8),
                 fontsize=12, fontweight="bold", color=ORANGE, va="center",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#FFF4F0", ec=ORANGE, lw=1.2))
    ax1.set_title("(a) 理论定界变点检测（标准化新息单侧 CUSUM）", loc="left", color=DARK, pad=10)
    ax1.set_xlabel("相对机制突变日的天数")
    ax1.set_ylabel("累计偏离统计量")
    ax1.set_xlim(-24.5, 4.6)
    ax1.set_ylim(-0.3, 10.8)

    x = np.arange(len(scenarios))
    w = 0.33
    b1 = ax2.bar(x - w / 2, acc_no, width=w, color="#F4A084", edgecolor=ORANGE, lw=1.2, label="仅靠低信息主序列（无旁路）")
    b2 = ax2.bar(x + w / 2, acc_yes, width=w, color=TEAL, edgecolor=DARK, lw=1.2, label="加入物理世界旁路监督")
    ax2.axhline(50, color=GREY, lw=1.3, ls=":")
    ax2.text(2.68, 51.5, "随机线\n50%", ha="right", va="bottom", fontsize=10.5, color=MUTED, fontweight="bold")

    for bar, v in zip(b1, acc_no):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 2.2, f"{v:.1f}%",
                 ha="center", va="bottom", fontsize=12, fontweight="bold", color=ORANGE)
    for bar, v in zip(b2, acc_yes):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 2.2, f"{v:.0f}%",
                 ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=DARK)

    ax2.set_xticks(x)
    ax2.set_xticklabels([s[0] for s in scenarios], fontsize=11.5)
    ax2.set_xlim(-0.55, 2.75)
    ax2.set_ylim(0, 128)
    ax2.set_ylabel("机制路径辨识准确率 (%)")
    ax2.set_title("(b) 物理监督彻底破除“同象异因”（20组随机种子）", loc="left", color=DARK, pad=10)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.48, 1.01), ncol=2, fontsize=10.8)

    fig1.tight_layout(w_pad=2.5)
    p1 = tmp / "chart_paradigm.png"
    fig1.savefig(p1, dpi=300)
    plt.close(fig1)

    # 2. SEIRS 实证案例轨迹对比（Seed 1, Day 195 shift, Alarm Day 199, Origin Day 202）
    r1 = run_case(1)
    y1 = r1["data"]["y"][T0:T0 + YEAR]
    sh1 = r1["shots"][MAIN_DELAY]
    tc1, al1, og1 = 195, r1["alarm"] - T0, sh1["origin"] - T0
    f1 = np.arange(og1 + 1, og1 + 1 + H)
    lo1, hi1 = tc1 - 24, og1 + H

    fig2, ax = plt.subplots(figsize=(7.6, 4.55), dpi=300)
    fit1 = np.array([q["pred"] for q in r1["eng"]["new"].log])
    ax.plot(np.arange(lo1, al1 + 1), fit1[lo1:al1 + 1], color="#64748B", lw=2.3, label="突变前：一步预测")
    ax.fill_between(f1, sh1["fc"]["new"][:, 0], sh1["fc"]["new"][:, 2], color=MINT, alpha=0.23, lw=0, label="选对路径 90% 置信区间")
    ax.plot(f1, sh1["fc"]["old"][:, 1], color=GREY, lw=2.8, ls="--")
    ax.plot(f1, sh1["fc"]["wrong"][:, 1], color=ORANGE, lw=3.0, ls="-.")
    ax.plot(f1, sh1["fc"]["new"][:, 1], color=TEAL, lw=3.8, ls="-")
    ax.plot(np.arange(lo1, hi1 + 1), y1[lo1:hi1 + 1], "o", color=DARK, ms=4.8, zorder=5, label="真实新增观测")

    ax.set_yscale("log")
    ax.set_xlim(lo1 - 1, hi1 + 13.8)
    _, ymax = ax.get_ylim()
    ax.set_ylim(95, ymax * 1.92)
    top = ax.get_ylim()[1]

    ax.axvline(tc1, color=DARK, lw=1.4, ls=":")
    ax.axvline(al1, color=TEAL, lw=1.4, ls="--")
    ax.axvline(og1, color=MUTED, lw=1.1, ls="-.")
    ax.text(tc1 - 0.8, top * 0.86, f"机制突变\n第{tc1}天", ha="right", va="top", color=DARK, fontsize=11.5, fontweight="bold")
    ax.text(al1, top * 0.88, f"报警+确认\n第{al1}天", ha="center", va="top", color=TEAL, fontsize=11.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec="#B8DBD9", lw=1))
    ax.text(og1 + 0.8, top * 0.60, f"统一出发\n第{og1}天", ha="left", va="top", color=MUTED, fontsize=10.8, fontweight="bold")

    end_x = f1[-1]
    ax.text(end_x + 0.9, sh1["fc"]["old"][-1, 1], f"不进化（沿用旧参数）\n21天 MAPE {sh1['mape']['old'][21]:.0%}",
            color=MUTED, fontsize=11.5, fontweight="bold", va="center")
    ax.text(end_x + 0.9, sh1["fc"]["wrong"][-1, 1] * 1.07, f"选错路径（无物理监督）\n21天 MAPE {sh1['mape']['wrong'][21]:.0%}",
            color=ORANGE, fontsize=11.5, fontweight="bold", va="center")
    ax.text(end_x + 0.9, sh1["fc"]["new"][-1, 1] * 1.01, f"选对路径（物理监督）\n21天 MAPE {sh1['mape']['new'][21]:.0%}",
            color=TEAL, fontsize=12, fontweight="bold", va="center")

    ax.set_title("实证验证（SEIRS 组1 · 对数纵轴）：未来 21 天轨迹预测对比", loc="left", color=DARK, pad=28)
    ax.set_xlabel("观测流第 t 天（全部模型于第 202 天统一起跑，向前预测未来 21 天）")
    ax.set_ylabel("每日新增观测（对数刻度）")
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3, fontsize=10.8)

    fig2.tight_layout()
    p2 = tmp / "chart_case.png"
    fig2.savefig(p2, dpi=300)
    plt.close(fig2)
    return p1, p2


def render_html(chart_paradigm_b64: str, chart_case_b64: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<style>
  @page {{
    size: {CSS_W}px {CSS_H}px;
    margin: 0;
  }}
  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}
  html, body {{
    width: {CSS_W}px;
    height: {CSS_H}px;
    overflow: hidden;
    font-family: "Hiragino Sans GB", "PingFang SC", "Heiti SC", "Microsoft YaHei", sans-serif;
    background: #F3F8F7;
    color: {INK};
    -webkit-font-smoothing: antialiased;
  }}
  .poster {{
    width: {CSS_W}px;
    height: {CSS_H}px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: #F3F8F7;
  }}

  /* ===== 1. HERO HEADER (flex-shrink: 0 so NEVER clipped!) ===== */
  .hero {{
    flex-shrink: 0;
    position: relative;
    background: linear-gradient(142deg, #072930 0%, {DARK} 48%, #0E585B 100%);
    color: #FFFFFF;
    padding: 68px 74px 32px 74px;
    overflow: hidden;
    border-bottom: 6px solid {MINT};
  }}
  .hero-bg-svg {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    opacity: 0.22;
  }}
  .hero-inner {{
    position: relative;
    z-index: 2;
  }}
  .hero-top-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 18px;
  }}
  .brand-pill {{
    display: inline-flex;
    align-items: center;
    gap: 11px;
    background: rgba(61, 219, 217, 0.14);
    border: 1.8px solid rgba(61, 219, 217, 0.55);
    color: #B4FAF8;
    padding: 8px 20px;
    border-radius: 999px;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 0.4px;
    white-space: nowrap;
  }}
  .brand-dot {{
    width: 11px;
    height: 11px;
    border-radius: 50%;
    background: {MINT};
    box-shadow: 0 0 10px {MINT};
  }}
  .tag-pill {{
    background: rgba(255, 255, 255, 0.10);
    border: 1.5px solid rgba(255, 255, 255, 0.28);
    color: #E3F3F2;
    padding: 8px 20px;
    border-radius: 999px;
    font-size: 19px;
    font-weight: 600;
    white-space: nowrap;
  }}
  .hero-title {{
    font-size: 64px;
    line-height: 1.16;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-bottom: 10px;
  }}
  .hero-title .hl {{
    color: {MINT};
  }}
  .hero-sub-en {{
    font-size: 20px;
    font-weight: 600;
    letter-spacing: 0.5px;
    color: #9ED3D0;
    margin-bottom: 16px;
  }}
  .hero-thesis {{
    font-size: 23.5px;
    line-height: 1.48;
    color: #ECF7F6;
    background: rgba(7, 41, 48, 0.55);
    border-left: 6px solid {MINT};
    padding: 13px 22px;
    border-radius: 0 14px 14px 0;
    margin-bottom: 20px;
  }}
  .hero-stats {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
  }}
  .hstat {{
    background: rgba(255, 255, 255, 0.09);
    border: 1.5px solid rgba(255, 255, 255, 0.18);
    border-radius: 14px;
    padding: 13px 18px;
    display: flex;
    align-items: center;
    gap: 14px;
  }}
  .hstat-num {{
    font-size: 36px;
    font-weight: 800;
    color: {MINT};
    white-space: nowrap;
  }}
  .hstat-num.warm {{
    color: #FFB088;
  }}
  .hstat-desc {{
    font-size: 17.5px;
    line-height: 1.35;
    color: #E0F0EF;
  }}
  .hstat-desc strong {{
    color: #FFFFFF;
    font-size: 19.5px;
    display: block;
    margin-bottom: 2px;
  }}

  /* ===== MAIN BODY WRAPPER ===== */
  .body-wrap {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 20px 68px 20px 68px;
    gap: 18px;
  }}

  .sec-head {{
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 11px;
    border-bottom: 2.5px solid #D3E3E1;
    padding-bottom: 7px;
  }}
  .sec-num {{
    flex-shrink: 0;
    white-space: nowrap;
    background: {DARK};
    color: {MINT};
    font-size: 18.5px;
    font-weight: 800;
    padding: 5px 14px;
    border-radius: 8px;
    letter-spacing: 0.6px;
  }}
  .sec-title {{
    font-size: 29px;
    font-weight: 800;
    color: {DARK};
    white-space: nowrap;
  }}
  .sec-sub {{
    margin-left: auto;
    font-size: 17.5px;
    color: {MUTED};
    font-weight: 600;
    white-space: nowrap;
  }}

  .card {{
    background: #FFFFFF;
    border: 1.5px solid #D5E4E2;
    border-radius: 16px;
    box-shadow: 0 5px 16px rgba(11, 58, 66, 0.045);
  }}

  /* ===== SECTION 1: TWO BLIND SPOTS ===== */
  .pain-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
  }}
  .pain-card {{
    padding: 16px 22px;
    display: flex;
    flex-direction: column;
    gap: 9px;
    border-top: 5px solid {ORANGE};
  }}
  .pain-card.b {{
    border-top-color: {TEAL};
  }}
  .pain-top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
  }}
  .pain-tag {{
    display: inline-block;
    font-size: 17px;
    font-weight: 700;
    padding: 4px 13px;
    border-radius: 999px;
    background: #FFF1EC;
    color: {ORANGE};
  }}
  .pain-card.b .pain-tag {{
    background: #E7F6F5;
    color: {TEAL};
  }}
  .pain-h {{
    font-size: 23.5px;
    font-weight: 800;
    color: {DARK};
  }}
  .pain-p {{
    font-size: 18px;
    line-height: 1.46;
    color: #2A4540;
  }}
  .pain-p strong {{
    color: {DARK};
  }}
  .pain-foot {{
    margin-top: auto;
    background: #F6FAF9;
    border-radius: 10px;
    padding: 9px 14px;
    font-size: 17.5px;
    font-weight: 700;
    color: {ORANGE};
  }}
  .pain-card.b .pain-foot {{
    color: {TEAL};
  }}

  /* ===== SECTION 2: PARADIGM ARCHITECTURE ===== */
  .paradigm-box {{
    padding: 16px 20px 14px 20px;
    display: flex;
    flex-direction: column;
    gap: 13px;
  }}
  .steps-row {{
    display: grid;
    grid-template-columns: 1fr 22px 1fr 22px 1fr 22px 1fr;
    align-items: stretch;
  }}
  .step-card {{
    background: #F7FBFA;
    border: 1.8px solid #CFE2E0;
    border-radius: 14px;
    padding: 13px 14px 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 7px;
  }}
  .step-card.highlight {{
    background: #EDF9F8;
    border: 2.2px solid {TEAL};
  }}
  .step-card.warn {{
    background: #FFF7F4;
    border: 2.2px solid {ORANGE};
  }}
  .step-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 17px;
    font-weight: 800;
    color: {TEAL};
  }}
  .step-card.warn .step-badge {{
    color: {ORANGE};
  }}
  .step-num {{
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: {DARK};
    color: {MINT};
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: 800;
  }}
  .step-card.warn .step-num {{
    background: {ORANGE};
    color: #FFFFFF;
  }}
  .step-title {{
    font-size: 20.5px;
    font-weight: 800;
    color: {DARK};
    line-height: 1.22;
  }}
  .step-eq {{
    font-family: "Menlo", "Consolas", monospace;
    font-size: 15px;
    background: #FFFFFF;
    border: 1px solid #D7E5E3;
    border-radius: 7px;
    padding: 5px 9px;
    color: {DARK};
    font-weight: 700;
  }}
  .step-desc {{
    font-size: 16.8px;
    line-height: 1.42;
    color: #2B4541;
  }}
  .arrow-col {{
    display: flex;
    align-items: center;
    justify-content: center;
  }}
  .supervision-strip {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 14px;
    background: linear-gradient(90deg, #0B3A42 0%, #104D54 100%);
    color: #FFFFFF;
    border-radius: 12px;
    padding: 12px 18px;
  }}
  .sup-item {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
  }}
  .sup-pill {{
    flex-shrink: 0;
    background: rgba(61, 219, 217, 0.18);
    border: 1.4px solid {MINT};
    color: {MINT};
    font-size: 15.5px;
    font-weight: 800;
    padding: 4px 10px;
    border-radius: 7px;
    margin-top: 2px;
    white-space: nowrap;
  }}
  .sup-txt {{
    font-size: 16.8px;
    line-height: 1.38;
    color: #E4F4F3;
  }}
  .sup-txt strong {{
    color: #FFFFFF;
  }}
  .chart-frame {{
    background: #FFFFFF;
    border: 1.5px solid #DCE9E7;
    border-radius: 12px;
    padding: 8px 12px 6px 12px;
  }}
  .chart-frame img {{
    width: 100%;
    display: block;
  }}

  /* ===== SECTION 3: EMPIRICAL CASE STUDY ===== */
  .case-grid {{
    display: grid;
    grid-template-columns: 1.24fr 1fr;
    gap: 20px;
    align-items: stretch;
  }}
  .case-left {{
    padding: 14px 16px 12px 16px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }}
  .case-left img {{
    width: 100%;
    display: block;
  }}
  .case-cap {{
    margin-top: 8px;
    font-size: 16.8px;
    color: {MUTED};
    line-height: 1.38;
    background: #F6FAF9;
    padding: 8px 13px;
    border-radius: 9px;
  }}
  .case-right {{
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    gap: 12px;
  }}
  .kpi-2x2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }}
  .kpi {{
    padding: 13px 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    border-left: 5px solid {TEAL};
  }}
  .kpi.warn {{
    border-left-color: {ORANGE};
  }}
  .kpi-label {{
    font-size: 16.8px;
    font-weight: 700;
    color: {MUTED};
  }}
  .kpi-val {{
    font-size: 27px;
    font-weight: 800;
    color: {DARK};
    line-height: 1.18;
    white-space: nowrap;
  }}
  .kpi.warn .kpi-val {{
    color: {ORANGE};
  }}
  .kpi-note {{
    font-size: 15.8px;
    line-height: 1.34;
    color: #36504C;
  }}

  .bench-card {{
    padding: 13px 16px;
  }}
  .bench-title {{
    font-size: 18px;
    font-weight: 800;
    color: {DARK};
    margin-bottom: 8px;
  }}
  table.bench {{
    width: 100%;
    border-collapse: collapse;
    font-size: 16.5px;
  }}
  table.bench th {{
    background: #EAF4F3;
    color: {DARK};
    font-weight: 800;
    padding: 7px 8px;
    text-align: left;
    border-bottom: 2px solid #BDD6D3;
    white-space: nowrap;
  }}
  table.bench td {{
    padding: 7px 8px;
    border-bottom: 1px solid #E5EFEE;
    color: #213C38;
  }}
  table.bench td.num {{
    white-space: nowrap;
  }}
  table.bench tr.best td {{
    background: #E8F7F6;
    font-weight: 800;
    color: {DARK};
  }}
  .ext-note {{
    margin-top: 8px;
    font-size: 15.8px;
    color: {MUTED};
    line-height: 1.38;
  }}
  .ext-note strong {{
    color: {DARK};
  }}

  /* ===== SECTION 4: GENERALIZATION ACROSS DOMAINS ===== */
  .domain-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
  }}
  .dom-card {{
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }}
  .dom-head {{
    padding: 12px 18px;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }}
  .dom-head.health {{ background: linear-gradient(120deg, {DARK}, {TEAL}); }}
  .dom-head.finance {{ background: linear-gradient(120deg, #1E3A8A, {BLUE}); }}
  .dom-head.industry {{ background: linear-gradient(120deg, #92400E, {AMBER}); }}
  .dom-title {{
    font-size: 21px;
    font-weight: 800;
  }}
  .dom-badge {{
    font-size: 14.5px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.22);
    white-space: nowrap;
  }}
  .dom-body {{
    padding: 12px 16px 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 7px;
    font-size: 16.8px;
    line-height: 1.38;
  }}
  .dom-row {{
    display: flex;
    gap: 8px;
    align-items: baseline;
  }}
  .dom-k {{
    flex-shrink: 0;
    font-weight: 800;
    color: {DARK};
    min-width: 88px;
  }}
  .dom-v {{
    color: #2D4742;
  }}

  /* ===== FOOTER (flex-shrink: 0, safe bottom >= 66px for X-banner grommets) ===== */
  .footer {{
    flex-shrink: 0;
    background: {DARK};
    color: #DDF1EF;
    padding: 20px 74px 66px 74px;
    border-top: 5px solid {MINT};
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 28px;
  }}
  .foot-left {{
    display: flex;
    flex-direction: column;
    gap: 5px;
    flex-shrink: 0;
  }}
  .foot-brand {{
    font-size: 26px;
    font-weight: 800;
    color: #FFFFFF;
  }}
  .foot-brand span {{
    color: {MINT};
  }}
  .foot-tag {{
    font-size: 17.5px;
    color: #B4DFDC;
    font-weight: 600;
  }}
  .foot-right {{
    max-width: 760px;
    text-align: right;
    font-size: 16.5px;
    line-height: 1.44;
    color: #B9DAD7;
  }}
</style>
</head>
<body>
<div class="poster" id="poster-root">

  <!-- ================= HERO ================= -->
  <header class="hero">
    <svg class="hero-bg-svg" viewBox="0 0 1575 560" preserveAspectRatio="none">
      <defs>
        <pattern id="g" width="60" height="60" patternUnits="userSpaceOnUse">
          <path d="M 60 0 L 0 0 0 60" fill="none" stroke="#3DDBD9" stroke-width="0.7" opacity="0.45"/>
        </pattern>
      </defs>
      <rect width="1575" height="560" fill="url(#g)"/>
      <path d="M0,420 C280,380 420,235 740,245 C980,253 1060,115 1280,145 C1410,162 1500,105 1575,90"
            fill="none" stroke="#3DDBD9" stroke-width="4" stroke-dasharray="10 8" opacity="0.65"/>
      <path d="M740,245 C940,320 1210,365 1575,405"
            fill="none" stroke="#E4572E" stroke-width="3.5" opacity="0.75"/>
      <circle cx="740" cy="245" r="10" fill="#E4572E"/>
    </svg>
    <div class="hero-inner">
      <div class="hero-top-row">
        <div class="brand-pill"><span class="brand-dot"></span>日新 · 低信息时序自进化方法论</div>
        <div class="tag-pill">低信息观测 · 变点快速响应 · 物理世界监督</div>
      </div>
      <h1 class="hero-title">面向<span class="hl">低信息时序</span>的<br/>物理监督流式自进化范式</h1>
      <div class="hero-sub-en">Physically-Supervised Parameter Evolution for Low-Information Time Series under Regime Shifts</div>
      <div class="hero-thesis">
        当系统长期只能观测到<strong>一维/低维结果序列</strong>（如每日新增数、股票日频价量、单传感器读数），且现实机制发生<strong>结构变点</strong>时：平时靠机理滤波稳健吸收噪声，突变来临<strong>数天内快速检出</strong>，并引入<strong>物理世界多模态监督</strong>破除“同象异因”歧义、选对演化方向。
      </div>
      <div class="hero-stats">
        <div class="hstat">
          <div class="hstat-num">3.55天</div>
          <div class="hstat-desc"><strong>极速变点响应</strong>20组随机仿真平均检出<br/>稳态误报仅 <b>0.05 次/年</b></div>
        </div>
        <div class="hstat">
          <div class="hstat-num warm">42%→100%</div>
          <div class="hstat-desc"><strong>物理监督破除歧义</strong>仅单序列识别 <b>42%–68%</b><br/>接入互补旁路达 <b>100%</b></div>
        </div>
        <div class="hstat">
          <div class="hstat-num">175%→13%</div>
          <div class="hstat-desc"><strong>突变后误差骤降</strong>3组7天平均预测MAPE<br/>20组严格验证 <b>204.8%→11.6%</b></div>
        </div>
      </div>
    </div>
  </header>

  <!-- ================= BODY ================= -->
  <main class="body-wrap">

    <!-- 01. PROBLEM -->
    <section>
      <div class="sec-head">
        <span class="sec-num">01 / 核心科学问题</span>
        <h2 class="sec-title">低信息时序在机制突变面前的“双盲困境”</h2>
        <span class="sec-sub">只有一条结果曲线时，为什么传统模型既迟钝又迷失？</span>
      </div>
      <div class="pain-grid">
        <div class="card pain-card">
          <div class="pain-top">
            <span class="pain-tag">困境一 · 时间维度盲区</span>
            <svg width="46" height="30" viewBox="0 0 46 32"><path d="M2,26 Q14,22 22,12 T44,4" fill="none" stroke="#8A8F8E" stroke-width="2.5" stroke-dasharray="4 3"/><path d="M2,26 Q14,22 22,12 L32,22 L44,27" fill="none" stroke="#E4572E" stroke-width="3"/><circle cx="22" cy="12" r="4" fill="#E4572E"/></svg>
          </div>
          <div class="pain-h">变点缺乏训练样本，响应滞后致误差爆炸</div>
          <p class="pain-p">
            真实场景常只有单一低频流，且<strong>政策突变、市场切换、设备故障在历史上几乎无重复样本</strong>。黑盒模型依赖同类样本而失效；常规卡尔曼滤波用固定小随机游走缓慢追赶，面对阶跃跳变剧烈滞后。
          </p>
          <div class="pain-foot">▶ 后果：变点后未来 7 天预测误差飙升至 175%–205%，21 天达 474%</div>
        </div>

        <div class="card pain-card b">
          <div class="pain-top">
            <span class="pain-tag">困境二 · 机理维度盲区（同象异因）</span>
            <svg width="46" height="30" viewBox="0 0 46 32"><path d="M4,8 C18,8 22,22 42,24" fill="none" stroke="#0E7C7B" stroke-width="3"/><path d="M4,8 C18,12 22,21 42,24" fill="none" stroke="#E4572E" stroke-width="2.5" stroke-dasharray="4 3"/><circle cx="4" cy="8" r="3.5" fill="#0B3A42"/></svg>
          </div>
          <div class="pain-h">同一条结果曲线，对应完全相反的底层归因</div>
          <p class="pain-p">
            低信息观测天然存在<strong>结构不可辨识性</strong>：不同机理变化短期内会画出几乎一致的曲线（病例下降 = 传播降或康复升；波动飙升 = 基本面切换或流动性冲击），只看主序列本身无法唯一反演原因。
          </p>
          <div class="pain-foot">▶ 后果：单序列自动选路仅 42.1%–68.4%，选错则因果倒置且外推发散</div>
        </div>
      </div>
    </section>

    <!-- 02. PARADIGM -->
    <section>
      <div class="sec-head">
        <span class="sec-num">02 / 通用方法论范式</span>
        <h2 class="sec-title">四阶段流式闭环：机理定骨架，物理督方向</h2>
        <span class="sec-sub">不依赖黑盒海量样本 · 通用抽象状态与观测流</span>
      </div>
      <div class="card paradigm-box">
        <div class="steps-row">
          <div class="step-card">
            <div class="step-badge"><span class="step-num">1</span>稳态运行</div>
            <div class="step-title">机理骨架 + 增广滤波</div>
            <div class="step-eq">x_t+1 = f(x_t, θ_t) + ε_t</div>
            <div class="step-desc">领域常识编码状态转移，参数 θ_t 增广随机游走。日常用集合滤波在线微调状态与平滑波动，<strong>不把偶然噪点误当突变</strong>。</div>
          </div>

          <div class="arrow-col">
            <svg width="20" height="20" viewBox="0 0 24 24"><path d="M4 12h14m-5-6 6 6-6 6" fill="none" stroke="#0E7C7B" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </div>

          <div class="step-card warn">
            <div class="step-badge"><span class="step-num">2</span>快速察觉</div>
            <div class="step-title">理论定界 CUSUM 报警</div>
            <div class="step-eq">S_t = max(0, S_t-1 + |ν_t| - k)</div>
            <div class="step-desc">对标准化新息 ν_t 做单侧累计检验，按预期误报率 ARL₀ 标定阈值 h：过 h/2 放宽噪声试探，过 h <strong>约 3–4 天锁定变点</strong>。</div>
          </div>

          <div class="arrow-col">
            <svg width="20" height="20" viewBox="0 0 24 24"><path d="M4 12h14m-5-6 6 6-6 6" fill="none" stroke="#0E7C7B" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </div>

          <div class="step-card">
            <div class="step-badge"><span class="step-num">3</span>并行演化</div>
            <div class="step-title">起点回溯 + 多路径分叉</div>
            <div class="step-eq">P(H_j | D_t) ∝ P(H_j) Σ P(y,z | H_j, τ)</div>
            <div class="step-desc">回溯历史快照并边际化未知突变起点 τ，并发运行候选机理分支 {{H₁…H_K}} 与虚警分支 H₀，逐日更新后验概率。</div>
          </div>

          <div class="arrow-col">
            <svg width="20" height="20" viewBox="0 0 24 24"><path d="M4 12h14m-5-6 6 6-6 6" fill="none" stroke="#0E7C7B" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"/></svg>
          </div>

          <div class="step-card highlight">
            <div class="step-badge"><span class="step-num">4</span>选对且可纠</div>
            <div class="step-title">物理世界监督与自纠</div>
            <div class="step-eq">L(y_t, z_t) × 先验 P(H_j | u_t)</div>
            <div class="step-desc">联合旁路物理观测 z_t 与外部事件先验 u_t 定性拍板；保留回退快照与冷却期，若选错与物理矛盾则<strong>再报警自纠</strong>。</div>
          </div>
        </div>

        <div class="supervision-strip">
          <div class="sup-item">
            <span class="sup-pill">物理监督 A</span>
            <div class="sup-txt"><strong>互补连续旁路 z_t：</strong>如搜索指数、气温、资金流或工况表，单看粗糙却与机理正交，直接打破多路径似然简并。</div>
          </div>
          <div class="sup-item">
            <span class="sup-pill">物理监督 B</span>
            <div class="sup-txt"><strong>外部语义事件 u_t：</strong>政策公告、财报、设备维保或专家“只点一次”确认，可作为条件先验立即锁定正确分支。</div>
          </div>
          <div class="sup-item">
            <span class="sup-pill">防错与自纠</span>
            <div class="sup-txt"><strong>虚警兜底 + 二次报警：</strong>H₀ 分支吸收孤立脉冲；哪怕暂选错分支，后续物理矛盾仍会以平均 <strong>1.05 次</strong>再报警纠正。</div>
          </div>
        </div>

        <div class="chart-frame">
          <img src="data:image/png;base64,{chart_paradigm_b64}" alt="CUSUM检测与物理监督路径识别对比" />
        </div>
      </div>
    </section>

    <!-- 03. CASE STUDY -->
    <section>
      <div class="sec-head">
        <span class="sec-num">03 / 典型实证案例</span>
        <h2 class="sec-title">传染病 SEIRS 验证：从低信息病例到物理监督自进化</h2>
        <span class="sec-sub">案例作范式验证载体 · 全程零神经网络 · 可复现</span>
      </div>
      <div class="case-grid">
        <div class="card case-left">
          <img src="data:image/png;base64,{chart_case_b64}" alt="SEIRS 21天预测对比曲线" />
          <div class="case-cap">
            <strong>实验设计：</strong>第 195 天传播率骤降（β×0.6），所有模型仅读低信息每日新增流与物理搜索旁路；第 199 天触发 CUSUM 报警，第 202 天统一起跑预测未来 21 天。<strong>选对路径紧密贴合真实曲线与 90% 置信区间</strong>。
          </div>
        </div>

        <div class="case-right">
          <div class="kpi-2x2">
            <div class="card kpi">
              <div class="kpi-label">变点检出速度 / 误报率</div>
              <div class="kpi-val">3.55 天</div>
              <div class="kpi-note">20 组均值；理论门槛 h=7.1 控制稳态误报仅 <strong>0.05 次/年</strong></div>
            </div>
            <div class="card kpi warn">
              <div class="kpi-label">3 组演示 7 天 MAPE</div>
              <div class="kpi-val">175% → 13%</div>
              <div class="kpi-note">选错路径仍达 <strong>40%</strong>；未来 21 天 <strong>474% → 14%</strong></div>
            </div>
            <div class="card kpi">
              <div class="kpi-label">同窗口演化前后对比</div>
              <div class="kpi-val">155.8% → 15.6%</div>
              <div class="kpi-note">19 组对齐 alarm 后 1–7 天误差直降至原 <strong>1/10</strong></div>
            </div>
            <div class="card kpi">
              <div class="kpi-label">错选后自我纠正能力</div>
              <div class="kpi-val">1.05 次再报警</div>
              <div class="kpi-note">错误分支迫使参数扭曲与物理信号冲突，触发二次自纠</div>
            </div>
          </div>

          <div class="card bench-card">
            <div class="bench-title">20 组随机种子严格评测（19 组有效对比 · 平均相对误差 MAPE）</div>
            <table class="bench">
              <thead>
                <tr><th>策略对照（仿真验证）</th><th>未来 1–7 天</th><th>未来 8–14 天</th><th>机制辨识</th></tr>
              </thead>
              <tbody>
                <tr><td>不进化（参数冻结）</td><td class="num">204.8%</td><td class="num">473.6%</td><td class="num">—</td></tr>
                <tr><td>常规微调（无变点模块）</td><td class="num">70.0%</td><td class="num">139.0%</td><td class="num">—</td></tr>
                <tr><td>进化 · 选错路径</td><td class="num">30.0%</td><td class="num">44.8%</td><td class="num">冲突错判</td></tr>
                <tr class="best"><td>进化 · 选对路径</td><td class="num">11.6%</td><td class="num">14.3%</td><td class="num">100%（带旁路）</td></tr>
              </tbody>
            </table>
            <div class="ext-note">
              <strong>多信号扩展原型（epievo v0→v1，35 条未知序列）：</strong>引入物理语义与联合似然元学习后，机制辨识率 <strong>46% → 94%</strong>，概率区间得分 WIS <strong>0.182 → 0.116</strong>。
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- 04. GENERALIZATION -->
    <section>
      <div class="sec-head">
        <span class="sec-num">04 / 跨领域范式迁移</span>
        <h2 class="sec-title">抽象“三元组”迁移至金融、工业等低信息变点系统</h2>
        <span class="sec-sub">低信息主观测 y_t + 领域机理骨架 f + 物理监督 (z_t, u_t)</span>
      </div>
      <div class="domain-grid">
        <div class="card dom-card">
          <div class="dom-head health">
            <span class="dom-title">公共卫生 · 疫情监测</span>
            <span class="dom-badge">已仿真验证</span>
          </div>
          <div class="dom-body">
            <div class="dom-row"><span class="dom-k">低信息序列</span><span class="dom-v">每日新增确诊数（滞后、残缺、高噪）</span></div>
            <div class="dom-row"><span class="dom-k">机理骨架</span><span class="dom-v">SEIRS 动力学（传播率 β、康复率 γ、易感池 S）</span></div>
            <div class="dom-row"><span class="dom-k">物理监督</span><span class="dom-v">互联网搜索指数、气温季节、封控公告与专家确认</span></div>
            <div class="dom-row"><span class="dom-k">破局核心</span><span class="dom-v">精准区分“传播率骤降”与“检测/康复加快”</span></div>
          </div>
        </div>

        <div class="card dom-card">
          <div class="dom-head finance">
            <span class="dom-title">金融市场 · 股价与波动</span>
            <span class="dom-badge">范式泛化展望</span>
          </div>
          <div class="dom-body">
            <div class="dom-row"><span class="dom-k">低信息序列</span><span class="dom-v">单标的日频/分钟级收益率与实现波动率</span></div>
            <div class="dom-row"><span class="dom-k">机理骨架</span><span class="dom-v">状态空间随机波动率、均值回归速率与跳跃冲击</span></div>
            <div class="dom-row"><span class="dom-k">物理监督</span><span class="dom-v">宏观利率、盘口资金流、财报公告与央行政策舆情</span></div>
            <div class="dom-row"><span class="dom-k">破局核心</span><span class="dom-v">快速区分“基本面估值范式切换”与“短期流动性踩踏”</span></div>
          </div>
        </div>

        <div class="card dom-card">
          <div class="dom-head industry">
            <span class="dom-title">工业运维 · 设备与路网</span>
            <span class="dom-badge">范式泛化展望</span>
          </div>
          <div class="dom-body">
            <div class="dom-row"><span class="dom-k">低信息序列</span><span class="dom-v">关键部件单维温振读数或城市断面车流量</span></div>
            <div class="dom-row"><span class="dom-k">机理骨架</span><span class="dom-v">部件疲劳退化动力学 / 热传导与守恒排队方程</span></div>
            <div class="dom-row"><span class="dom-k">物理监督</span><span class="dom-v">电耗负载工况表、停机维保日志、天气与交通调度</span></div>
            <div class="dom-row"><span class="dom-k">破局核心</span><span class="dom-v">精准分离“传感器零点漂移 / 工况切换”与“真实结构磨损”</span></div>
          </div>
        </div>
      </div>
    </section>

  </main>

  <!-- ================= FOOTER ================= -->
  <footer class="footer">
    <div class="foot-left">
      <div class="foot-brand">日新 · <span>Adaptive Stream Paradigm</span></div>
      <div class="foot-tag">小数据 · 机理可解释 · 变点快响应 · 物理监督 · 错可回退自纠</div>
    </div>
    <div class="foot-right">
      诚实边界与数据说明：展架中所有定量指标（3.55天检出、0.05次/年误报、MAPE对比、100%辨识与1.05次自纠）均来自项目 SEIRS 及多信号仿真实验真实运行结果（未使用神经网络）；金融与工业迁移板块为同一方法论框架之应用场景映射展望。
    </div>
  </footer>

</div>
</body>
</html>
"""


def main() -> None:
    if not CHROME_BIN.exists():
        raise FileNotFoundError(f"未找到 Chrome: {CHROME_BIN}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        p1, p2 = build_charts(tmp)
        html = render_html(_b64(p1), _b64(p2))
        html_path = tmp / "poster.html"
        raw_png = tmp / "raw_poster.png"
        html_path.write_text(html, encoding="utf-8")

        cmd = [
            str(CHROME_BIN),
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--hide-scrollbars",
            f"--window-size={CSS_W},{CSS_H}",
            f"--force-device-scale-factor={SCALE}",
            f"--screenshot={raw_png}",
            f"file://{html_path}",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if not raw_png.exists():
            raise RuntimeError(f"Chrome 未生成截图: {res.stderr}")

        img = Image.open(raw_png).convert("RGB")
        if img.size != (TARGET_W, TARGET_H):
            img = img.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
        img.save(OUT_JPG, "JPEG", quality=95, subsampling=0, dpi=(TARGET_DPI, TARGET_DPI))
        print(f"Saved: {OUT_JPG}")
        print(f"Pixels: {img.size}, DPI: ({TARGET_DPI}, {TARGET_DPI}), Physical: {img.size[0]/TARGET_DPI*2.54:.2f}cm x {img.size[1]/TARGET_DPI*2.54:.2f}cm")
        print(f"File size: {OUT_JPG.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
