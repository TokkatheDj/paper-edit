"""Spike 2b -- is the export duration drift per-cut, or a constant?

+174 ms over 30 cuts is harmless if it is a fixed encoder tail. If it scales with
cut count it is ~6 ms per cut, which on a heavily filler-cut two-hour podcast
(hundreds of cuts) becomes seconds of drift -- and that WOULD break sync.
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))

from paperedit.edl import Cut, EditPlan
from paperedit.render import media_info, render

SRC = HERE / "media" / "test-300s.mp4"


def plan_with(n: int) -> EditPlan:
    """n evenly spaced 4s cuts -- same total output length regardless of n."""
    total = 120.0
    seg = total / n
    return EditPlan([Cut(round(i * 8.0, 3), round(i * 8.0 + seg, 3)) for i in range(n)])


def main():
    print(f"{'cuts':>5} {'expected':>10} {'actual':>10} {'drift ms':>9} {'ms/cut':>8}  codec")
    print("-" * 60)
    for n in (1, 2, 5, 15, 30, 60):
        for codec, extra in (("aac", []), ("pcm", ["-c:a", "pcm_s16le"])):
            plan = plan_with(n)
            dst = HERE / "out" / f"drift_{n}_{codec}.{'mkv' if codec=='pcm' else 'mp4'}"
            render(SRC, plan, dst, gpu=True, extra=extra)
            actual = media_info(dst)["duration"]
            drift = (actual - plan.duration) * 1000
            print(f"{n:>5} {plan.duration:>9.3f}s {actual:>9.3f}s "
                  f"{drift:>+9.0f} {drift/n:>8.1f}  {codec}")
            dst.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
