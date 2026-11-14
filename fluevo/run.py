"""交互式流式运行：python3 -m fluevo.run [--seed 2] [--scenario lockdown|treatment|reopen|none] [--no-search] [--auto]

逐日运行第六年。出现变点时列出候选进化路径和概率，等你确认：
  输入 1/2/3/4 选路径；直接回车 = 今天不给信息（系统继续观察，到期自动取概率最高的）。
加 --auto 则全程不询问。
"""
import argparse

import numpy as np

from .engine import Engine, PATH_CN
from .experiment import T0, TC
from .pretrain import pretrain
from .sim import simulate, YEAR, B0_TRUE, AMP_TRUE, OMEGA_TRUE

SCEN = {"lockdown": dict(kind="beta", day=TC, k=0.6), "treatment": dict(kind="gamma", day=TC, k=1.6),
        "reopen": dict(kind="beta", day=TC, k=1.4), "none": None}


def ask(t, probs):
    keys = list(probs)
    print(f"\n  第 {t - T0} 天，候选进化路径：")
    for i, p in enumerate(keys, 1):
        print(f"    {i}. {PATH_CN[p]:<12} 概率 {probs[p]:6.1%}")
    ans = input("  你的确认（输入编号；回车=暂不确认）: ").strip()
    return keys[int(ans) - 1] if ans.isdigit() and 1 <= int(ans) <= len(keys) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--scenario", default="lockdown", choices=list(SCEN))
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--auto", action="store_true")
    a = ap.parse_args()
    data = simulate(a.seed, change=SCEN[a.scenario])
    print("预训练：用前五年数据估计初始参数 ...")
    pre = pretrain(data["y"][:T0], data["s"][:T0], seed=a.seed)
    th = pre["theta"]
    print(f"  季节幅度 {np.hypot(th['b1'], th['b2']):.3f}（真值 {AMP_TRUE}） 基础传播率 {np.exp(th['b0']):.3f}（真值 {np.exp(B0_TRUE):.3f}）"
          f" 免疫期 {1 / np.exp(th['lw']):.0f} 天（真值 {1 / OMEGA_TRUE:.0f}）")
    e = Engine(pre, seed=a.seed, use_search=not a.no_search, human=None if a.auto else ask, day0=T0)
    print(f"变点门槛 h = {e.h}（依据：模型无误时平均 5 年才误报一次）。开始第六年的逐日运行。\n")
    print("  天   实际  一步预测  标准化误差  累计偏离  分级      传播率×  康复率×")
    for t in range(T0, T0 + YEAR):
        r = e.step(t, data["y"][t], data["s"][t])
        day = t - T0
        if r["tier"] != "正常" or r["action"] or day % 20 == 0:
            print(f"{day:4d} {int(r['y']):6d} {r['pred']:9.0f} {r['nu'][0]:+10.2f} {r['cusum']:9.1f}  {r['tier']:<6} "
                  f"{r['beta_mult']:8.2f} {r['gamma_mult']:8.2f}" + (f"   <- {r['action']}" if r["action"] else ""))
    print("\n本年的进化记录：")
    for ev in e.events:
        print(f"  第 {ev['alarm'] - T0} 天报警 -> 第 {ev['day'] - T0} 天{ev['who']}确认：{PATH_CN[ev['path']]}（始于第 {ev['onset'] - T0} 天）")
    if SCEN[a.scenario]:
        c = SCEN[a.scenario]
        print(f"真实情况：第 {TC - T0} 天起 {'传播率' if c['kind'] == 'beta' else '康复率'} ×{c['k']}")


if __name__ == "__main__":
    main()
