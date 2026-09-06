#!/usr/bin/env python3
"""
verify_env.py  --  Check your environment for the Delivery Robot, and fix it.

By the Electronics & Robotics Club, BITS Goa.

Run this from inside the activated course environment:

    conda activate rl
    python env/verify_env.py

It checks that everything the assignment needs is installed. If something is
missing, it tries to install it for you (with conda, then pip). Run it again
afterwards to confirm everything is green.
"""

import importlib
import platform
import subprocess
import sys

PASS = "[ OK ]"
FAIL = "[FAIL]"
WARN = "[WARN]"

# (import name, install name)
REQUIRED = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("PIL", "pillow"),
    ("tqdm", "tqdm"),
]


def have(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def install(packages):
    """Install packages into the active environment: try conda, then pip."""
    attempts = [
        ["conda", "install", "-y"] + packages,
        [sys.executable, "-m", "pip", "install"] + packages,
    ]
    for cmd in attempts:
        try:
            print("   running: " + " ".join(cmd))
            subprocess.check_call(cmd)
            return True
        except Exception as exc:
            print("   (that did not work: {})".format(exc))
    return False


def ffmpeg_available():
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.animation import FFMpegWriter
        return FFMpegWriter.isAvailable()
    except Exception:
        return False


def main():
    print("Delivery Robot: environment check and auto-fix")
    print("By the Electronics & Robotics Club, BITS Goa.\n")
    print("Python {} on {}".format(platform.python_version(), platform.system()))
    ok = True

    # 1) Python packages -----------------------------------------------------
    missing = [(m, p) for (m, p) in REQUIRED if not have(m)]
    if missing:
        names = ", ".join(p for _, p in missing)
        print("\nMissing packages: {}. Installing them now...".format(names))
        install([p for _, p in missing])
        importlib.invalidate_caches()

    print("\nPackages")
    print("--------")
    for module_name, pkg in REQUIRED:
        if have(module_name):
            print("{} {}".format(PASS, pkg))
        else:
            print("{} {} (could not install automatically; try: conda install {})"
                  .format(FAIL, pkg, pkg))
            ok = False

    # 2) ffmpeg for the rollout video ----------------------------------------
    print("\nVideo (rollout.mp4)")
    print("-------------------")
    if not ffmpeg_available():
        print("ffmpeg not found; installing (needed to save the rollout video)...")
        install(["ffmpeg"])
    if ffmpeg_available():
        print("{} ffmpeg is available; rollout videos will save as MP4.".format(PASS))
    else:
        print("{} ffmpeg still not available. The video will fall back to a GIF."
              .format(WARN))
        print("       You can install it with:  conda install ffmpeg")

    # 3) Headless plotting smoke test ----------------------------------------
    print("\nPlotting (no screen needed)")
    print("---------------------------")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import tempfile
        fig, ax = plt.subplots()
        ax.plot([0, 1, 2], [0, 1, 4])
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            fig.savefig(tmp.name)
        plt.close(fig)
        print("{} Matplotlib can draw and save a figure headless (Agg backend)."
              .format(PASS))
    except Exception as exc:
        print("{} Matplotlib could not save a figure: {}".format(FAIL, exc))
        ok = False

    print()
    if ok:
        print("All set. Your environment is ready. Now open the assignment and start on dp.py.")
        return 0
    print("Some checks failed. Re-run this file, or ask a mentor if it keeps failing.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
