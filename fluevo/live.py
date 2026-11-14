"""现场实时看板（流感版）：python3 -m fluevo.live  ->  打开 http://127.0.0.1:8767

页面每前进一天，这里就真的让三个引擎各算一步：
  a 不进化（参数冻结）  b 只微调  c 完整系统（变点处的路径选择由页面上的人工点击决定）
未来的真值不会提前发给页面。只监听本机。
"""
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

import numpy as np

from .engine import Engine, PATH_CN
from .experiment import T0, TC
from .pretrain import pretrain
from .run import SCEN
from .sim import simulate, YEAR, B0_TRUE, AMP_TRUE, PEAK_TRUE, OMEGA_TRUE, SEARCH_C

PORT = 8767
LOCK = threading.Lock()
SESSIONS = {}
SCEN_CN = {"lockdown": "封城（传播率 ×0.6）", "treatment": "治疗提速（康复率 ×1.6）",
           "reopen": "解封（传播率 ×1.4）", "none": "无干预"}


def r2(x, n=2):
    return None if x is None else round(float(x), n)


class Session:
    def __init__(self, scenario, use_search, seed):
        self.scenario, self.use_search, self.seed = scenario, use_search, seed
        self.data = simulate(seed, change=SCEN[scenario])
        self.pre = pretrain(self.data["y"][:T0], self.data["s"][:T0], seed=seed)
        self.choice = None
        self.d = 0                                    # 下一个要前进到的天（第六年内）
        self.eng = {
            "a": Engine(self.pre, seed, use_search, evolve="frozen", day0=T0),
            "b": Engine(self.pre, seed, use_search, evolve="tune", day0=T0),
            "c": Engine(self.pre, seed, use_search, evolve="full", human=self._human, day0=T0),
        }
        self.med = {k: {} for k in self.eng}          # 每个引擎每天做出的第 7 天预测：目标日 -> 值
        self.err = {k: [] for k in self.eng}
        self.n_ev = 0

    def _human(self, t, probs):                       # 只交付一次，交付后清空
        c, self.choice = self.choice, None
        return c if c in probs else None

    def pretrain_info(self):
        th = self.pre["theta"]
        ang = float(np.arctan2(th["b2"], th["b1"]))
        rows = [
            ("季节幅度", float(np.hypot(th["b1"], th["b2"])), AMP_TRUE, ""),
            ("流行高峰日（一年中第几天）", float((ang / (2 * np.pi) * YEAR) % YEAR), float(PEAK_TRUE), "天"),
            ("基础传播率", float(np.exp(th["b0"])), float(np.exp(B0_TRUE)), ""),
            ("免疫维持时间", float(1 / np.exp(th["lw"])), float(1 / OMEGA_TRUE), "天"),
        ]
        if self.use_search:
            rows.append(("搜索指数比例常数", float(self.pre["c"]), SEARCH_C, ""))
        return [dict(name=n, learned=r2(a, 3), truth=r2(b, 3), unit=u) for n, a, b, u in rows]

    def step(self, choice=None):
        d = self.d
        t = T0 + d
        y, s = self.data["y"][t], self.data["s"][t]
        self.choice = choice
        out = dict(d=d, y=int(y), s=r2(s) if self.use_search else None, eng={})
        for k, e in self.eng.items():
            rec = e.step(t, y, s if self.use_search else None)
            fc = np.array(rec["fc"])
            self.med[k][d + 7] = float(fc[6, 1])
            p7 = self.med[k].get(d)
            if p7 is not None:
                self.err[k].append(abs(p7 - y) / max(y, 10))
            o = dict(p7=r2(p7), beta=r2(rec["beta_mult"], 3), gamma=r2(rec["gamma_mult"], 3),
                     rel7=r2(np.mean(self.err[k]), 4) if self.err[k] else None, n7=len(self.err[k]))
            if k == "c":
                o.update(pred=r2(rec["pred"]), nu=[r2(v) for v in rec["nu"]], cusum=r2(rec["cusum"]),
                         tier=rec["tier"], action=rec["action"],
                         fc=[[r2(v) for v in row] for row in fc.tolist()],
                         probs={p: r2(v, 4) for p, v in rec["probs"].items()} if rec["probs"] else None,
                         fc_paths={p: [r2(v) for v in vs] for p, vs in rec["fc_paths"].items()} if rec.get("fc_paths") else None,
                         alarm=(e.alarm_day - T0) if e.cands is not None else None,
                         onset=(e.onset - T0) if e.cands is not None else None)
            out["eng"][k] = o
        out["events"] = [dict(day=ev["day"] - T0, alarm=ev["alarm"] - T0, onset=ev["onset"] - T0, path=ev["path"],
                              path_cn=PATH_CN[ev["path"]], who=ev["who"],
                              probs={p: r2(v, 4) for p, v in ev["probs"].items()}) for ev in self.eng["c"].events]
        self.d += 1
        out["done"] = self.d >= YEAR
        return out


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/":
                tpl = (Path(__file__).parent / "live_template.html").read_text(encoding="utf-8")
                cfg = dict(scen=[dict(key=k, name=v) for k, v in SCEN_CN.items()], path_cn=PATH_CN, year=YEAR)
                return self.send(tpl.replace("__CFG__", json.dumps(cfg, ensure_ascii=False)).encode(), "text/html")
            if u.path == "/api/start":
                sc = q.get("scenario", "lockdown")
                if sc not in SCEN:
                    return self.send(dict(error="unknown scenario"), code=400)
                seed = random.randrange(100, 99999) if q.get("seed") == "new" else int(q.get("seed", 2))
                use_search = q.get("search", "1") != "0"
                with LOCK:
                    SESSIONS.clear()
                    t0 = time.perf_counter()
                    s = Session(sc, use_search, seed)
                    sid = str(random.randrange(10 ** 9))
                    SESSIONS[sid] = s
                    h = s.eng["c"].h
                    return self.send(dict(sid=sid, seed=seed, scenario=sc, search=use_search, h=h, year=YEAR,
                                          pretrain=s.pretrain_info(), secs=round(time.perf_counter() - t0, 2)))
            if u.path == "/api/step":
                with LOCK:
                    s = SESSIONS.get(q.get("sid"))
                    if not s:
                        return self.send(dict(error="session expired"), code=410)
                    if s.d >= YEAR:
                        return self.send(dict(error="finished"), code=409)
                    return self.send(s.step(q.get("choice") or None))
            if u.path == "/api/truth":                 # 只有勾选“揭晓真实变化”才会来取
                with LOCK:
                    s = SESSIONS.get(q.get("sid"))
                    if not s:
                        return self.send(dict(error="session expired"), code=410)
                    ch = SCEN[s.scenario]
                    if not ch:
                        return self.send(dict(change=None, text="这条数据流没有真实变化"))
                    what = "传播率" if ch["kind"] == "beta" else "康复率"
                    return self.send(dict(change=dict(day=ch["day"] - T0, kind=ch["kind"], k=ch["k"]),
                                          text=f"第 {ch['day'] - T0} 天起，真实的{what} ×{ch['k']}"))
        except Exception as ex:                        # 让页面看到错误而不是挂起
            import traceback
            traceback.print_exc()
            return self.send(dict(error=repr(ex)), code=500)
        self.send(dict(error="not found"), code=404)


def main():
    print(f"流感版实时看板: http://127.0.0.1:{PORT}   (Ctrl+C 结束)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
