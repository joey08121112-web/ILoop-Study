"""X 展架海报（80cm × 180cm）：python3 build_poster.py  → poster.html → X展架_80x180cm.jpg
页面按 1px = 1mm 排版（800 × 1800），用 Chrome 无头模式 6 倍输出（4800 × 10800，约 152 dpi）。
data.json 是 fluevo.showcase.run_case(1) 的结果（数据组 1，报警后第 3 天出发的预测）。"""
import json, math, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
d = json.loads((HERE / "data.json").read_text())
INK, PAPER, RED, BLUE, GREY, NAVY = "#141a24", "#f6f3ec", "#e2512a", "#2468d8", "#8d8b82", "#0e1726"

# ---------- 主图：变点附近的预测（对数纵轴） ----------
def chart(W=676, Hh=216):
    L, R, T, B = 44, 78, 30, 34
    x0, x1, y0, y1 = 166, 223, 100, 2200
    X = lambda t: L + (t - x0) / (x1 - x0) * (W - L - R)
    Y = lambda v: T + (math.log10(y1) - math.log10(max(v, 1))) / (math.log10(y1) - math.log10(y0)) * (Hh - T - B)
    path = lambda pts: "M" + " L".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in pts)
    tc, al, og = d["tc"], d["alarm"], d["origin"]
    s = [f'<svg viewBox="0 0 {W} {Hh}" width="{W}" height="{Hh}" xmlns="http://www.w3.org/2000/svg" font-family="PingFang SC, Hiragino Sans GB, sans-serif">']
    for v in (100, 200, 500, 1000, 2000):
        s.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#dedbd2" stroke-width=".6"/>'
                 f'<text x="{L-5}" y="{Y(v)+2.6:.1f}" font-size="7.5" fill="{GREY}" text-anchor="end">{v}</text>')
    for t in range(170, 221, 10):
        s.append(f'<text x="{X(t):.1f}" y="{Hh-B+12}" font-size="7.5" fill="{GREY}" text-anchor="middle">第 {t} 天</text>')
    s.append(f'<rect x="{X(tc):.1f}" y="{T}" width="{X(al)-X(tc):.1f}" height="{Hh-T-B}" fill="{RED}" opacity=".10"/>')
    s.append(f'<line x1="{X(tc):.1f}" x2="{X(tc):.1f}" y1="{T}" y2="{Hh-B}" stroke="{INK}" stroke-width=".9" stroke-dasharray="2 2"/>')
    s.append(f'<line x1="{X(al):.1f}" x2="{X(al):.1f}" y1="{T}" y2="{Hh-B}" stroke="{RED}" stroke-width="1.1"/>')
    s.append(f'<text x="{X(tc)-3:.1f}" y="{T+9}" font-size="8.5" fill="{INK}" text-anchor="end" font-weight="600">变点：世界变了</text>')
    s.append(f'<text x="{X(al)+3:.1f}" y="{T+9}" font-size="8.5" fill="{RED}" font-weight="600">{al-tc} 天后报警 · 选路 · 进化</text>')
    fc = d["fc"]; days = [og + 1 + i for i in range(len(d["act"]))]
    new = fc["new"]
    band = "M" + " L".join(f"{X(t):.1f},{Y(q[2]):.1f}" for t, q in zip(days, new)) + " L" + \
           " L".join(f"{X(t):.1f},{Y(q[0]):.1f}" for t, q in reversed(list(zip(days, new)))) + "Z"
    s.append(f'<path d="{band}" fill="{BLUE}" opacity=".16"/>')
    s.append(f'<path d="{path([(t, d["pred"][t]) for t in range(x0, al + 1)])}" fill="none" stroke="#5b5a55" stroke-width="1.3"/>')
    for k, col, w, dash in (("old", GREY, 2.2, "5 3"), ("wrong", RED, 2.2, "6 2 1.5 2"), ("new", BLUE, 2.8, "")):
        s.append(f'<path d="{path([(t, q[1]) for t, q in zip(days, fc[k])])}" fill="none" stroke="{col}" stroke-width="{w}" stroke-dasharray="{dash}" stroke-linecap="round"/>')
    for t in range(x0, x1 + 1):
        s.append(f'<circle cx="{X(t):.1f}" cy="{Y(d["y"][t]):.1f}" r="1.9" fill="{INK}"/>')
    ex = W - R + 5
    for k, col, a, b in (("old", GREY, "不进化", "越跑越偏"), ("wrong", RED, "选错路径", "有改善"), ("new", BLUE, "选对路径", "贴住现实")):
        yy = Y(fc[k][-1][1])
        s.append(f'<text x="{ex}" y="{yy:.1f}" font-size="10" font-weight="700" fill="{col}">{a}</text>'
                 f'<text x="{ex}" y="{yy+10:.1f}" font-size="7.5" fill="{col}">{b}</text>')
    s.append(f'<circle cx="{L+6}" cy="{T+8}" r="1.9" fill="{INK}"/><text x="{L+12}" y="{T+10.7}" font-size="8" fill="{INK}">实际每日新增</text>')
    s.append(f'<line x1="{L+2}" x2="{L+10}" y1="{T+20}" y2="{T+20}" stroke="#5b5a55" stroke-width="1.3"/><text x="{L+12}" y="{T+22.7}" font-size="8" fill="{INK}">报警前的逐日预测</text>')
    s.append(f'<rect x="{L+2}" y="{T+29.5}" width="8" height="5" fill="{BLUE}" opacity=".25"/><text x="{L+12}" y="{T+34.7}" font-size="8" fill="{INK}">选对路径的 90% 区间</text>')
    s.append('</svg>')
    return "".join(s)

# ---------- 题图：全年曲线（真实数据组 1） ----------
def hero_curve(W=800, Hh=104):
    y = d["y"]; m = max(y)
    X = lambda t: t / 364 * W
    Y = lambda v: Hh - 8 - (v / m) * (Hh - 50)
    pts = " L".join(f"{X(t):.1f},{Y(v):.1f}" for t, v in enumerate(y))
    tc = d["tc"]
    return (f'<svg class="curve" viewBox="0 0 {W} {Hh}" width="{W}" height="{Hh}" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="{RED}" stop-opacity=".38"/><stop offset="1" stop-color="{RED}" stop-opacity="0"/></linearGradient></defs>'
            f'<path d="M0,{Hh} L{pts} L{W},{Hh}Z" fill="url(#g)"/><path d="M{pts}" fill="none" stroke="{RED}" stroke-width="1.6"/>'
            f'<line x1="{X(tc):.1f}" x2="{X(tc):.1f}" y1="6" y2="{Hh}" stroke="#fff" stroke-width="1" stroke-dasharray="3 3" opacity=".8"/>'
            f'<circle cx="{X(tc):.1f}" cy="{Y(y[tc]):.1f}" r="5" fill="#fff"/>'
            f'<text x="{X(tc)+16:.1f}" y="26" fill="#fff" font-size="15" font-weight="600" font-family="PingFang SC, sans-serif">变点</text>'
            f'<text x="{X(tc)+16:.1f}" y="42" fill="#fff" opacity=".7" font-size="10.5" font-family="PingFang SC, sans-serif">规律在这一天改写</text></svg>')

# ---------- 自进化示意：一条轨迹在变点处分叉，物理监督选定其一 ----------
def fork(W=688, Hh=196):
    c = W / 4; my = 104
    import random; random.seed(3)
    pts = [(x, my + random.uniform(-3, 3)) for x in range(0, int(c * 1.5) + 1, 8)]
    base = "M" + " L".join(f"{x},{y:.1f}" for x, y in pts)
    x0 = c * 1.5; xe = c * 2.55
    s = [f'<svg viewBox="0 0 {W} {Hh}" width="{W}" height="{Hh}" xmlns="http://www.w3.org/2000/svg" font-family="PingFang SC, Hiragino Sans GB, sans-serif">']
    for k in (1, 2, 3):
        s.append(f'<line x1="{c*k}" x2="{c*k}" y1="0" y2="{Hh}" stroke="#fff" stroke-opacity=".12"/>')
    s.append(f'<path d="{base}" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round"/>')
    s.append(f'<text x="14" y="{my-16}" fill="#fff" fill-opacity=".75" font-size="11.5">逐日微调，平稳运行</text>')
    # 报警
    s.append(f'<circle cx="{x0}" cy="{my}" r="17" fill="{RED}" fill-opacity=".25"/><circle cx="{x0}" cy="{my}" r="7" fill="{RED}"/>')
    s.append(f'<text x="{x0}" y="{my-28}" fill="{RED}" font-size="14" font-weight="700" text-anchor="middle">报警！</text>')
    # 分叉
    for lab, ye, sel in (("参数 A 变了", 30, True), ("参数 B 变了", 82, False), ("参数 C 变了", 128, False), ("没有变（虚警）", 172, False)):
        d_ = f"M{x0},{my} C{x0+60},{my} {xe-70},{ye} {xe},{ye}"
        if sel:
            s.append(f'<path d="{d_} L{W-70},{ye}" fill="none" stroke="{RED}" stroke-width="4.2" stroke-linecap="round"/>')
            s.append(f'<path d="M{W-70},{ye-8} L{W-54},{ye} L{W-70},{ye+8}Z" fill="{RED}"/>')
            s.append(f'<text x="{c*3-22}" y="{ye-11}" fill="#fff" font-size="13" font-weight="700" text-anchor="end">{lab}</text>')
            s.append(f'<text x="{c*3+16}" y="{ye+24}" fill="#fff" font-size="12.5" font-weight="600">沿这条路径继续进化</text>')
            s.append(f'<circle cx="{c*3}" cy="{ye}" r="13" fill="{INK}" stroke="#fff" stroke-width="2"/><path d="M{c*3-6},{ye} l4.5,5 l8,-10" fill="none" stroke="#fff" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>')
        else:
            s.append(f'<path d="{d_}" fill="none" stroke="#fff" stroke-opacity=".42" stroke-width="1.8" stroke-dasharray="5 4" stroke-linecap="round"/>')
            s.append(f'<text x="{xe+8}" y="{ye+4}" fill="#fff" fill-opacity=".62" font-size="11.5">{lab}</text>')
    s.append(f'<text x="{c*3+16}" y="{my+30}" fill="#fff" fill-opacity=".6" font-size="11">其余路径不采纳</text>')
    s.append('</svg>')
    return "".join(s)

HTML = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>ILoop Study · X展架</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{width:800px;height:1800px;overflow:hidden}}
body{{background:{PAPER};color:{INK};font-family:"PingFang SC","Hiragino Sans GB",sans-serif;font-size:12px;line-height:1.5;-webkit-font-smoothing:antialiased}}
.num{{font-family:"Avenir Next","Helvetica Neue",sans-serif;font-weight:700}}
.serif{{font-family:"Songti SC","STSong",serif;font-weight:900}}
.hero{{position:relative;height:456px;background:{NAVY};color:#fff;padding:50px 56px 0;overflow:hidden}}
.hero .top{{display:flex;justify-content:space-between;align-items:baseline;font-size:12px;letter-spacing:.12em;opacity:.85}}
.hero .top b{{font-size:18px;letter-spacing:.04em}}
.hero h1{{font-size:82px;line-height:1.14;margin-top:24px}}
.hero h1 em{{font-style:normal;color:{RED}}}
.hero .sub{{font-size:23px;margin-top:16px;font-weight:500;letter-spacing:.04em}}
.chips{{display:flex;gap:10px;margin-top:14px;position:relative;z-index:2}}
.chips span{{border:1px solid rgba(255,255,255,.5);border-radius:99px;padding:5px 14px;font-size:15px;letter-spacing:.06em;background:rgba(14,23,38,.6)}}
.curve{{position:absolute;left:0;bottom:0}}
section{{padding:0 56px}}
.h{{display:flex;align-items:baseline;gap:12px;margin-bottom:14px}}
.h .n{{white-space:nowrap;font-size:13px;color:{RED};letter-spacing:.14em;font-weight:700}}
.h h2{{font-size:30px;line-height:1.2;white-space:nowrap}}
.h p{{font-size:12px;color:#5e5c55;margin-left:auto}}
.pain{{padding-top:28px}}
.g3{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}
.card{{background:#fff;border-radius:8px;padding:14px 16px 15px;border:1px solid #e4e0d6}}
.card h3{{font-size:20px;margin-bottom:3px}}
.card p{{font-size:12.5px;color:#3d3c38;line-height:1.5}}
.para{{margin-top:28px;background:{INK};color:#fff;padding:28px 56px 26px}}
.para .h{{display:block;margin-bottom:6px}}
.para .h h2{{color:#fff;font-size:68px;margin-top:2px;line-height:1.15}}
.para .h h2 em{{font-style:normal;color:{RED}}}
.para .lead{{font-size:19px;color:#fff;margin-bottom:14px;letter-spacing:.02em}} .para .lead small{{display:block;font-size:13.5px;color:#b9b6ab;margin-top:3px}}
.steps{{display:grid;grid-template-columns:repeat(4,1fr);margin-top:4px}}
.step{{padding:12px 14px 0 0;border-top:2px solid rgba(255,255,255,.3)}}
.step:last-child{{border-color:{RED}}}
.step .k{{font-size:13px;color:{RED};letter-spacing:.1em}}
.step h3{{font-size:22px;margin:1px 0 4px}}
.step p{{font-size:12.5px;line-height:1.5;opacity:.85}}
.sup{{margin-top:20px;padding-top:14px;border-top:1px solid rgba(255,255,255,.2);display:flex;align-items:center;gap:12px;font-size:14px}}
.sup b{{font-size:15px;margin-right:6px;color:#fff}}
.sup span{{border:1px solid {RED};color:#fff;border-radius:99px;padding:4px 14px}}
.case{{padding-top:26px}}
.fig{{background:#fff;border:1px solid #e4e0d6;border-radius:8px;padding:8px 4px 2px 4px}}
.note{{font-size:12px;color:#5e5c55;margin-top:9px}}
.gen{{padding-top:22px}}
.g4{{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}}
.dom{{border-top:2px solid {INK};padding-top:8px}}
.dom h3{{font-size:18px}}
.dom p{{font-size:12px;color:#3d3c38;line-height:1.5;margin-top:2px}}
.dom i{{font-style:normal;font-size:10px;color:#5e5c55;border:1px solid {GREY};border-radius:3px;padding:0 5px;margin-left:6px;vertical-align:3px;font-weight:400}}
.dom.done{{border-color:{RED}}} .dom.done i{{background:{RED};border-color:{RED};color:#fff}}
footer{{position:absolute;left:0;right:0;bottom:0;height:78px;padding:14px 56px 0;border-top:1px solid #d6d2c6;display:flex;justify-content:space-between;align-items:baseline;font-size:10.5px;color:#5e5c55}}
footer b{{color:{INK};font-size:16px;letter-spacing:.04em}}
</style>
<body>
<div class="hero">
  <div class="top"><b>ILoop Study</b><span>BoHack 2026 天津青年黑客松 ·「日新」赛道</span></div>
  <h1 class="serif">世界突然变了，<br>模型<em>自己进化</em></h1>
  <div class="sub">物理世界监督下的自进化范式</div>
  <div class="chips"><span>低信息数据</span><span>变点快速响应</span><span>物理世界监督</span></div>
  {hero_curve()}
</div>

<section class="pain">
  <div class="h"><span class="n">01 难题</span><h2 class="serif">每天只有一个数，世界却会突变</h2></div>
  <div class="g3">
    <div class="card"><h3>信息少</h3><p>一天一个数，喂不饱大模型</p></div>
    <div class="card"><h3>会突变</h3><p>变点一到，旧规律立刻失效</p></div>
    <div class="card"><h3>原因多</h3><p>表象相似，分不清是谁变了</p></div>
  </div>
</section>

<div class="para">
  <div class="h"><span class="n">02 范式</span><h2 class="serif"><em>自进化</em></h2></div>
  <p class="lead">变点处，模型自己改写参数<small>不重新训练，不等数据堆够——报警、分叉、选路，然后继续跑</small></p>
  {fork()}
  <div class="steps">
    <div class="step"><div class="k num">STEP 1</div><h3>机理打底</h3><p>方程描述世界，数据只校准几个参数</p></div>
    <div class="step"><div class="k num">STEP 2</div><h3>察觉变点</h3><p>预测与现实的偏差累积过线就报警</p></div>
    <div class="step"><div class="k num">STEP 3</div><h3>枚举路径</h3><p>每种“可能变了的参数”各走一条路</p></div>
    <div class="step"><div class="k num">STEP 4</div><h3>物理监督</h3><p>由真实世界选路；选错会再报警纠正</p></div>
  </div>
  <div class="sup"><b>物理世界的监督来自</b><span>机理方程</span><span>旁路观测</span><span>人的判断</span></div>
</div>

<section class="case">
  <div class="h"><span class="n">03 实例</span><h2 class="serif">流感：封城后的每日新增</h2><p>仿真数据</p></div>
  <div class="fig">{chart()}</div>
  <p class="note">变点后平均 3.6 天报警；选对路径后，未来 21 天预测误差由 474% 降到 14%。</p>
</section>

<section class="gen">
  <div class="h"><span class="n">04 泛化</span><h2 class="serif">换一套机理方程，就是另一个领域</h2></div>
  <div class="g4">
    <div class="dom done"><h3>传染病<i>已验证</i></h3><p>每日新增病例</p></div>
    <div class="dom"><h3>金融市场<i>设想</i></h3><p>每日收盘价</p></div>
    <div class="dom"><h3>设备运维<i>设想</i></h3><p>传感器日读数</p></div>
    <div class="dom"><h3>能源负荷<i>设想</i></h3><p>每日用电量</p></div>
  </div>
</section>

<footer><b>ILoop Study</b><span>全部结果来自仿真数据；其他领域为迁移设想，尚未验证。</span></footer>
</body></html>"""
(HERE / "poster.html").write_text(HTML)

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
scale = int(sys.argv[1]) if len(sys.argv) > 1 else 6
png = HERE / "poster.png"
subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--force-device-scale-factor={scale}",
                "--window-size=800,1800", f"--screenshot={png}", (HERE / "poster.html").as_uri()], check=True, capture_output=True)
from PIL import Image
im = Image.open(png).convert("RGB")
out = HERE / ("X展架_80x180cm.jpg" if scale == 6 else f"preview_{scale}x.jpg")
im.save(out, quality=93, dpi=(im.width / 80 * 2.54, im.width / 80 * 2.54), subsampling=0)
png.unlink()
print(out, im.size)
