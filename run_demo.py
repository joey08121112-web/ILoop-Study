"""ILoop Study · 流感版 —— 两次运行演示

怎么运行（任选一种）：
  1. 在 VS Code 里打开本文件，点右上角的 ▶（Run Python File）；
  2. 在 VS Code 里按 F5，选“运行演示”；
  3. 在终端里：python3 run_demo.py

第一次运行：只输出“不进化”的结果 —— 系统发现了变点，但参数保持变点前的样子去预测未来 21 天。
第二次运行：输出“进化”的结果 —— 分别走“选错路径”（当成康复率变了）和“选对路径”（确认传播率变了），
            并与第一次运行留下的记录放在一起对比。
再运行：重新生成第二次运行的结果。

清除记录（清完之后再运行，就从“第一次运行”重新开始）：
  python3 run_demo.py --reset        或者直接运行 reset_demo.py

三组“今年”的仿真数据，每组 365 天，第 195 天封城（传播率 ×0.6）。
输出在桌面的文件夹「日新_演示结果」。全部数据是仿真生成的。
"""
import csv
import io
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # 不打印找不到字体粗细的提示
sys.path.insert(0, str(ROOT))                     # 保证从任何目录、用“点击运行”都能找到 fluevo 包

try:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scipy  # noqa: F401
except ImportError as e:
    sys.exit(f"缺少依赖：{e.name}。请先在终端运行：python3 -m pip install numpy matplotlib scipy")

from fluevo.showcase import run_case, figure, NAME, MAIN_DELAY   # noqa: E402
from fluevo.experiment import OUT as CACHE, T0                    # noqa: E402

SEEDS = (1, 2, 3)
OUT = Path.home() / "Desktop" / "日新_演示结果"
STATE = OUT / "运行记录.json"
HORIZONS = (7, 14, 21)
FONTS = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "Microsoft YaHei", "SimHei"]


def reset():
    """清除所有运行记录和输出；下次运行从“第一次运行”开始，并连预训练一起重算。"""
    if OUT.exists():
        shutil.rmtree(OUT)
    n = 0
    for f in CACHE.glob("pretrain_seed*.pkl"):
        f.unlink()
        n += 1
    print(f"已清除：桌面的「{OUT.name}」文件夹，以及 {n} 个预训练缓存。下次运行将从“第一次运行”开始。")


def setup_style():
    plt.rcParams.update({"font.sans-serif": FONTS, "axes.unicode_minus": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#e6e5e0",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                         "grid.linewidth": 0.6, "font.size": 10, "axes.titlelocation": "left", "axes.titlesize": 12,
                         "legend.frameon": False})


def mape_of(rows, keys):
    """{做法: {数据组或'平均': {步长: MAPE}}}"""
    out = {}
    for k in keys:
        per = {str(r["seed"]): {str(h): r["shots"][MAIN_DELAY]["mape"][k][h] for h in HORIZONS} for r in rows}
        per["平均"] = {str(h): float(np.mean([per[str(r["seed"])][str(h)] for r in rows])) for h in HORIZONS}
        out[k] = per
    return out


def table_rows(columns):
    """columns: [(列名, {数据组: {步长: MAPE}})] -> 表头和各行。"""
    head = ["数据组", "预测步长"] + [c for c, _ in columns]
    body = []
    for g in [str(s) for s in SEEDS] + ["平均"]:
        for h in HORIZONS:
            body.append([("三组平均" if g == "平均" else f"数据组 {g}"), f"未来 {h} 天"] + [m[g][str(h)] for _, m in columns])
    return head, body


def write_table(head, body, stem, title, highlight=None):
    # 输出文件名只从固定映射取得（stem 仅作查表键），不做路径拼接，杜绝路径穿越
    names = {"第一次运行_MAPE": ("第一次运行_MAPE.csv", "第一次运行_MAPE.png"),
             "第二次运行_MAPE对比": ("第二次运行_MAPE对比.csv", "第二次运行_MAPE对比.png")}
    try:
        csv_name, png_name = names[stem]
    except KeyError:
        raise ValueError(f"未知表名：{stem!r}") from None
    buf = io.StringIO(newline="")                        # CSV 先写进内存缓冲，不手工 open 文件
    w = csv.writer(buf)
    w.writerow(head[:2] + [h + " MAPE" for h in head[2:]])
    for row in body:
        w.writerow(row[:2] + [f"{v:.1%}" for v in row[2:]])
    (OUT / csv_name).write_bytes(buf.getvalue().encode("utf-8-sig"))   # 带 BOM，Excel 打开不乱码
    fig, ax = plt.subplots(figsize=(2.2 * len(head) + 1.5, 5.4))
    ax.axis("off")
    t = ax.table(cellText=[r[:2] + [f"{v:.0%}" for v in r[2:]] for r in body], colLabels=head, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(12)
    t.scale(1, 1.75)
    for (i, j), c in t.get_celld().items():
        c.set_edgecolor("#e3e2dc")
        c.set_facecolor("#fcfcfb")
        if i == 0 or i > len(body) - len(HORIZONS):
            c.set_text_props(weight="bold")
            c.set_facecolor("#f0efec")
        if highlight is not None and j == highlight and i > 0:
            c.set_facecolor("#cde2fb")
    ax.set_title(title, fontsize=12, loc="left", color="#0b0b0b", pad=6)
    fig.text(0.125, 0.04, "MAPE = |预测中位数 − 真值| / 真值 的平均，越小越好。全部为仿真数据。", fontsize=9, color="#52514e")
    fig.savefig(OUT / png_name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return head, body


def show(head, body):
    print("  " + "".join(f"{h:<12}" for h in head))
    for row in body:
        print("  " + f"{row[0]:<10}{row[1]:<12}" + "".join(f"{v:>10.0%}    " for v in row[2:]))


def compute():
    rows = []
    for seed in SEEDS:
        r = run_case(seed, delays=(MAIN_DELAY,))
        ev = r["event"]
        print(f"  数据组 {seed}：预训练{'读缓存' if r['cached'] else '已完成'}；封城在第 195 天，系统在第 {ev['alarm'] - T0} 天报警")
        rows.append(r)
    return rows


def first_run():
    print("【第一次运行】不进化：发现变点后，参数仍保持变点前的样子\n")
    rows = compute()
    for r in rows:
        figure(r, MAIN_DELAY, OUT / f"第一次运行_数据组{r['seed']}.png", keys=("old",), tag="第一次运行（不进化）  ")
    m = mape_of(rows, ("old",))
    head, body = write_table(*table_rows([("不进化", m["old"])]), "第一次运行_MAPE",
                             f"第一次运行：变点附近的预测误差 MAPE（不进化）\n报警后第 {MAIN_DELAY} 天出发，预测未来 7 / 14 / 21 天")
    STATE.write_text(json.dumps(dict(runs=1, first=dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), mape=m["old"])),
                                ensure_ascii=False, indent=2))
    print("\n变点附近的 MAPE（越小越好）")
    show(head, body)
    print("\n下一步：再运行一次本文件，看进化之后的结果。")


def second_run(state):
    again = state["runs"] >= 2
    print("【第二次运行】进化：报警后改参数，分别走“选错路径”和“选对路径”" + ("（重新生成）" if again else "") + "\n")
    rows = compute()
    for r in rows:
        figure(r, MAIN_DELAY, OUT / f"第二次运行_数据组{r['seed']}.png", tag="第二次运行（进化）  ")
    m = mape_of(rows, ("wrong", "new"))
    first = state["first"]
    head, body = write_table(*table_rows([("第一次：不进化", first["mape"]), ("第二次：" + NAME["wrong"], m["wrong"]),
                                          ("第二次：" + NAME["new"], m["new"])]), "第二次运行_MAPE对比",
                             f"两次运行对比：变点附近的预测误差 MAPE\n第一次的数字取自 {first['time']} 的运行记录；"
                             f"三种做法都在报警后第 {MAIN_DELAY} 天、用同样的数据出发预测", highlight=4)
    state.update(runs=2, second=dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), mape=m))
    STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2))
    print("\n两次运行对比：变点附近的 MAPE（越小越好）")
    show(head, body)
    a, b, c = (np.mean([row[i] for row in body[-3:]]) for i in (2, 3, 4))
    print(f"\n三组、三个步长的总平均：不进化 {a:.0%} → 选错路径 {b:.0%} → 选对路径 {c:.0%}")
    print("说明：“选错路径”是把封城当成“康复率变了”。如需从头再演示一遍：python3 run_demo.py --reset")


def main():
    if "--reset" in sys.argv[1:]:
        return reset()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style()
    state = json.loads(STATE.read_text()) if STATE.exists() else None
    print("ILoop Study · 流感版 两次运行演示")
    print(f"输出文件夹：{OUT}\n")
    first_run() if state is None else second_run(state)
    print(f"\n用时 {time.time() - t0:.0f} 秒。文件夹里现在有：")
    for f in sorted(OUT.iterdir()):
        print("  " + f.name)
    if sys.platform == "darwin":
        subprocess.run(["open", str(OUT)], check=False)
    elif sys.platform.startswith("win"):
        subprocess.run(["explorer", str(OUT)], check=False)


if __name__ == "__main__":
    main()
