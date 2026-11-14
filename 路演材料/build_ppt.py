"""生成 4 页路演 PPT：python3 路演材料/build_ppt.py   （先运行 make_assets.py 生成素材）"""
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
A = HERE / "assets"
S = json.loads((A / "stats.json").read_text())
FONT = "Hiragino Sans GB"   # 冬青黑体，macOS 自带；在 Windows 上打开时会自动换成系统中文字体
DARK, TEAL, MINT, ORANGE, GREY = "0B3A42", "0E7C7B", "3DDBD9", "E4572E", "8A8F8E"
INK, MUTED, CARD, WHITE = "16302B", "5B6B68", "EEF5F4", "FFFFFF"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def rgb(h):
    return RGBColor.from_string(h)


def bg(slide, color):
    f = slide.background.fill
    f.solid()
    f.fore_color.rgb = rgb(color)


def text(slide, s, x, y, w, h, size, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, spacing=None):
    """s 可以是字符串，或 [(文字, {size, color, bold})] 组成的多段。"""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    paras = s if isinstance(s, list) else [(s, {})]
    for i, (t, o) in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if spacing:
            p.space_after = Pt(spacing)
        r = p.add_run()
        r.text = t
        f = r.font
        f.size = Pt(o.get("size", size))
        f.bold = o.get("bold", bold)
        f.color.rgb = rgb(o.get("color", color))
        f.name = FONT
        rpr = r._r.get_or_add_rPr()
        for tag in ("a:ea", "a:cs"):                       # 中文也用同一字体
            el = rpr.find(qn(tag))
            if el is None:
                el = rpr.makeelement(qn(tag), {})
                rpr.append(el)
            el.set("typeface", FONT)
    return tb


def card(slide, x, y, w, h, color=CARD):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.06
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(color)
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def chip(slide, x, y, color, d=0.22):
    sh = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(color)
    sh.line.fill.background()
    sh.shadow.inherit = False


def picture(slide, name, x, y, w, h):
    """等比缩放放进 (x, y, w, h) 的框里，靠左、垂直居中。"""
    iw, ih = Image.open(A / name).size
    k = min(w / iw, h / ih)
    pw, ph = iw * k, ih * k
    slide.shapes.add_picture(str(A / name), Inches(x), Inches(y + (h - ph) / 2), Inches(pw), Inches(ph))


pct = lambda v: f"{v:.0%}"

# ---------------------------------------------------------------- 第 1 页：项目名
s = prs.slides.add_slide(BLANK)
bg(s, DARK)
text(s, "BoHack 2026 · 「日新」主题赛道", 0.9, 0.8, 11, 0.4, 16, MINT)
text(s, "日新", 0.9, 1.6, 11.5, 1.5, 80, WHITE, bold=True)
text(s, "物理信息引导参数进化的传染病流式预测", 0.9, 3.25, 11.5, 0.8, 36, WHITE, bold=True)
text(s, "变点来临时，让模型的参数朝正确的方向进化", 0.9, 4.2, 11.5, 0.5, 20, MINT)
picture(s, "cover_curve.png", 0.9, 5.05, 11.5, 1.75)
s.notes_slide.notes_text_frame.text = "开场：报出项目名称，一句话点出痛点——形势突变时，预测模型最容易失准。"

# ---------------------------------------------------------------- 第 2 页：方程
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
text(s, "机理模型：状态方程、参数方程与观测方程", 0.6, 0.45, 12.1, 0.8, 36, INK, bold=True)

card(s, 0.6, 1.5, 4.35, 5.0)
text(s, "SEIRS 微分方程组", 0.9, 1.72, 3.8, 0.4, 20, TEAL, bold=True)
text(s, "易感 → 潜伏 → 感染 → 康复 → 易感", 0.9, 2.17, 3.8, 0.35, 14, MUTED)
picture(s, "eq_seirs.png", 0.95, 2.65, 3.7, 3.65)

card(s, 5.2, 1.5, 7.55, 2.38)
text(s, "参数方程", 5.5, 1.72, 2.4, 0.4, 20, TEAL, bold=True)
text(s, [("季节项 + 干预项", {}), ("平时小步微调 ε", {}), ("变点处发生跳变 J", {})], 5.5, 2.2, 2.3, 1.5, 14, MUTED, spacing=6)
picture(s, "eq_param.png", 7.85, 1.68, 4.7, 2.02)

card(s, 5.2, 4.12, 7.55, 2.38)
text(s, "状态方程与观测方程", 5.5, 4.34, 2.4, 0.75, 20, TEAL, bold=True)
text(s, [("参数并入状态一起估计", {}), ("y：每日新增病例", {}), ("s：搜索指数", {})], 5.5, 5.12, 2.3, 1.3, 14, MUTED, spacing=6)
picture(s, "eq_obs.png", 7.85, 4.3, 4.7, 2.02)

text(s, "求解框架：增广状态的集合卡尔曼滤波，每天“预测 → 用真值检验 → 更新”", 0.6, 6.72, 12.1, 0.4, 16, INK)
s.notes_slide.notes_text_frame.text = ("左侧是 SEIRS 机理；右上是传播率和康复率的参数方程，平时微调、变点处跳变；"
                                       "右下是状态方程和两路观测。整体用集合卡尔曼滤波逐日求解。")

# ---------------------------------------------------------------- 第 3 页：结果
s = prs.slides.add_slide(BLANK)
bg(s, WHITE)
text(s, "结果：变点之后，选对进化路径预测才准", 0.6, 0.45, 12.1, 0.8, 36, INK, bold=True)
picture(s, "result.png", 0.45, 1.4, 7.75, 4.95)
al, tc = S["alarm"][0], S["tc"]
text(s, f"数据组 1：第 {tc} 天封城，第 {al} 天报警；三种做法在报警后第 3 天用同样的数据出发，预测未来 21 天",
     0.6, 6.42, 7.6, 0.55, 12, MUTED)

card(s, 8.55, 1.5, 4.2, 3.72)
text(s, "未来 7 天预测误差 MAPE", 8.85, 1.7, 3.7, 0.4, 18, TEAL, bold=True)
text(s, "三组仿真数据平均，越小越好", 8.85, 2.1, 3.7, 0.3, 12, MUTED)
for i, (k, name, col) in enumerate((("old", "不进化", GREY), ("wrong", "进化 · 选错路径", ORANGE), ("new", "进化 · 选对路径", TEAL))):
    yy = 2.55 + i * 0.86
    chip(s, 8.87, yy + 0.27, col)
    text(s, pct(S["avg"][k]["7"]), 9.25, yy, 1.75, 0.75, 40, INK, bold=True, anchor=MSO_ANCHOR.MIDDLE)
    text(s, name, 11.0, yy, 1.65, 0.75, 14, INK, anchor=MSO_ANCHOR.MIDDLE)

a = S["avg"]
text(s, [(f"未来 21 天：{pct(a['old']['21'])} → {pct(a['wrong']['21'])} → {pct(a['new']['21'])}", {"bold": True}),
         (f"封城后 {min(S['alarm']) - tc}–{max(S['alarm']) - tc} 天报警，变点日估计准确", {}),
         ("自动选对：有搜索 100%，仅病例 42–68%", {})],
     8.6, 5.42, 4.2, 1.2, 14, INK, spacing=5)
text(s, "全部为仿真数据；“选错路径”为带搜索指数条件下的结果", 8.6, 6.68, 4.2, 0.3, 11, MUTED)
s.notes_slide.notes_text_frame.text = ("灰线不进化，误差上百；橙线选错路径；青线选对路径，误差回到约一成。"
                                       "只看病例时路径难分，加入搜索指数或人工确认后才能选对。")

# ---------------------------------------------------------------- 第 4 页：致谢
s = prs.slides.add_slide(BLANK)
bg(s, DARK)
text(s, "谢谢大家", 0.9, 2.35, 11.5, 1.3, 72, WHITE, bold=True, align=PP_ALIGN.CENTER)
text(s, "敬请各位评委老师批评指正", 0.9, 3.85, 11.5, 0.5, 22, MINT, align=PP_ALIGN.CENTER)
picture(s, "cover_curve.png", 0.9, 5.05, 11.5, 1.75)
s.notes_slide.notes_text_frame.text = "致谢，进入提问环节。"

out = HERE / "日新_路演.pptx"
prs.save(out)
print("已生成", out)
