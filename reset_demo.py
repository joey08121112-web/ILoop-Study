"""清除演示的运行记录：在 VS Code 里点 ▶ 运行本文件，或在终端里 python3 reset_demo.py
清除之后再运行 run_demo.py，就从“第一次运行”重新开始。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_demo  # noqa: E402

if __name__ == "__main__":
    run_demo.reset()
