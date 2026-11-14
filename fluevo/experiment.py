"""封城实验：python3 -m fluevo.experiment  ->  out_flu/results.json + 图

同一条数据流、同一天做出的预测，比较四种做法：
  不进化        参数冻结，只更新状态
  只微调        常规卡尔曼微调，没有变点模块
  进化·选错路径  变点处被告知“康复率变了”
  进化·选对路径  变点处被告知“传播率变了”
另外统计：不给人工信息时，有/无搜索指数的自动选择正确率。
"""
import json
from pathlib import Path

import numpy as np

from .engine import Engine
from .pretrain import pretrain
from .sim import simulate, YEAR

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out_flu"
T0 = 5 * YEAR                 # 前五年预训练，第六年在线运行
TC = T0 + 195                 # 第六年第 195 天封城（流行上升期）
LOCKDOWN = dict(kind="beta", day=TC, k=0.6)


def once(choice):
    """模拟人工确认：第一次报警时给出 choice，之后不再给信息。"""
    state = {"used": False}

    def human(t, probs):
        if state["used"]:
            return None
        state["used"] = True
        return choice
    return human


def run(data, pre, seed, **kw):
    e = Engine(pre, seed=seed, **kw)
    for t in range(T0, T0 + YEAR):
        e.step(t, data["y"][t], data["s"][t])
    return e


def issue_errors(e, y, lo, hi):
    """对每个发布日 t：未来 1-7 天、8-14 天预测中位数的平均相对误差。"""
    out = {}
    for r in e.log:
        t = r["t"]
        if lo <= t <= hi and t + 14 < len(y):
            med = np.array(r["fc"])[:, 1]
            act = y[t + 1:t + 15]
            err = np.abs(med - act) / np.maximum(act, 10)
            out[t] = (float(err[:7].mean()), float(err[7:].mean()))
    return out


def variants():
    """每条数据流都要新建人工回调，否则“只确认一次”的状态会串到下一条数据流。"""
    return {"frozen": dict(evolve="frozen"), "tune": dict(evolve="tune"),
            "wrong": dict(human=once("gamma")), "right": dict(human=once("beta"))}


VARIANTS = tuple(variants())


def lockdown_case(seed):
    data = simulate(seed, change=LOCKDOWN)
    pre = pretrain(data["y"][:T0], data["s"][:T0], seed=seed)
    engines = {name: run(data, pre, seed, **kw) for name, kw in variants().items()}
    return data, pre, engines


def summarize(seeds):
    """多个随机种子上的平均：变点后各做法的预测误差，以及自动选路径的正确率。"""
    err = {v: [] for v in VARIANTS}
    sel = {}
    alarms, false_alarms, n_pairs, fixes = [], 0, 0, []
    params = {v: [] for v in VARIANTS}
    for seed in seeds:
        data, pre, engines = lockdown_case(seed)
        near = [ev["alarm"] for ev in engines["right"].events if ev["alarm"] >= TC - 2]
        if not near:
            continue
        alarm = near[0]
        alarms.append(alarm - TC)
        if not any(ev["who"] == "人工" and ev["alarm"] == alarm for ev in engines["wrong"].events):
            continue                    # 人工确认被更早的一次虚警用掉了，这条数据流不计入路径对比
        alarms[-1:] = [alarm - TC]
        n_pairs += 1
        fixes.append(sum(1 for ev in engines["wrong"].events if ev["alarm"] > alarm and ev["alarm"] <= alarm + 60))
        for v, e in engines.items():
            errs = issue_errors(e, data["y"], alarm, alarm + 21)
            err[v].append(np.mean(list(errs.values()), axis=0))
            params[v].append((e.log[alarm + 20 - T0]["beta_mult"], e.log[alarm + 20 - T0]["gamma_mult"]))
        for kind, k in (("beta", 0.6), ("gamma", 1.6), ("beta", 1.4)):
            d2 = data if (kind, k) == ("beta", 0.6) else simulate(seed, change=dict(kind=kind, day=TC, k=k))
            p2 = pre if d2 is data else pretrain(d2["y"][:T0], d2["s"][:T0], seed=seed)
            for use_s in (True, False):
                e = run(d2, p2, seed, use_search=use_s)
                near = [ev for ev in e.events if TC - 2 <= ev["alarm"] <= TC + 20]
                first = near[0] if near else None
                ok = bool(first and first["path"] == kind)
                sel.setdefault((kind, k, use_s), []).append((ok, float(first["probs"][kind]) if first else 0.0))
        d0 = simulate(seed)
        false_alarms += len(run(d0, pretrain(d0["y"][:T0], d0["s"][:T0], seed=seed), seed).events)
    return dict(
        seeds=len(alarms), pairs=n_pairs, alarm_delay=float(np.mean(alarms)),
        wrong_path_later_corrections=float(np.mean(fixes)),
        params_20d_after_alarm={v: dict(beta=float(np.mean([p[0] for p in params[v]])),
                                        gamma=float(np.mean([p[1] for p in params[v]]))) for v in params},
        err7={v: float(np.mean([e[0] for e in err[v]])) for v in err},
        err14={v: float(np.mean([e[1] for e in err[v]])) for v in err},
        selection=[dict(kind=k[0], k=k[1], search=k[2], acc=float(np.mean([a for a, _ in v])),
                        prob=float(np.mean([p for _, p in v]))) for k, v in sel.items()],
        false_alarms_per_year=false_alarms / len(seeds))


def main():
    OUT.mkdir(exist_ok=True)
    summ = summarize(range(1, 21))
    (OUT / "results.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    print(json.dumps(summ, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
