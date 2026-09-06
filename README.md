# Week 2 Assignment: The Delivery Robot

**Electronics & Robotics Club, BITS Goa · Introduction to Reinforcement Learning**

This week you turn the four Dynamic Programming algorithms from the notes into working
code and use them to drive a delivery robot through progressively larger warehouses. You
write one file, `dp.py`. Everything else is provided. There is no notebook: you run Python
scripts from the terminal in the course `rl` environment.

## What's in this folder

```
dp.py               <- THE ONE FILE YOU EDIT (your six DP functions)
assignment.pdf         the assignment: read this first
README.md              this file
figures/               your plots and the rollout video appear here after you run
env/                   the environment: environment.yml and verify_env.py
utils/                 provided code you do not touch (env, maps, checker, plots)
```

You only ever edit **`dp.py`**. The `utils/` folder holds the warehouse, the grader, and
the plotting code; do not change it.

## 1. Set up the environment (once)

If you have not built the course environment yet (see the top-level `setup/` folder):

```bash
conda env create -f env/environment.yml
conda activate rl
```

Then check everything is ready. This also installs anything that is missing:

```bash
python env/verify_env.py
```

If a package (or `ffmpeg`, used for the video) is missing, `verify_env.py` installs it and
tells you. Run it again until every line is green.

## 2. Write your solver and check it

Open `dp.py` and fill in the six functions. After each one, grade your work:

```bash
python utils/check.py
```

Each line tells you exactly which function passed or failed. A function you have not
written yet is reported politely, not as a crash, so you can fill them in one at a time.
The rhythm for every function: read its docstring in `dp.py` and the chapter of the notes
it points to, work out the update yourself, write the loop over states, then run
`python utils/check.py`.

## 3. Make your figures and the rollout video

Once every check passes:

```bash
python utils/run.py
```

This solves all four levels with **your** `dp.py` and writes everything into the
**`figures/`** folder, including `rollout.mp4`, a video of your robot solving the huge
Level 3 maze.

## 4. Watch and debug your robot

You do not need to open the `utils/` folder for any of this. Everything you look at is in
`figures/`, next to `dp.py`:

- **Ubuntu:** `xdg-open figures/rollout.mp4` (or just double-click it in the file manager).
- **WSL:** run `explorer.exe figures` and double-click `rollout.mp4` in the window that
  opens.

The video is a debugging tool, not just a demo. On the left, your robot moves through the
maze along the route your policy chose. On the right, a panel shows, for every step:

- the value `V` of the cell the robot is on,
- the four action values `q(s, a)`, with the move your policy took and the move with the
  highest `q` both marked,
- a warning if your policy did **not** pick the highest-`q` move (a sign your
  `greedy_policy` has a bug),
- the reward, the running return, and a note whenever the slippery floor made it slide.

So if your value map looks wrong or the robot walks into a wall or a hazard, scrub through
the video, find the step where it goes wrong, and read the panel to see what your code
computed there. The static value maps (`level1_value_policy.png`, `level3_value_policy.png`)
help the same way.

## What to hand in (on Google Classroom)

1. A single **PDF** of your Part A answers (H1 to H4).
2. Your completed **`dp.py`** with a passing `python utils/check.py`.
3. Your Level 3 **`figures/rollout.mp4`** (you must upload this video when you hand in).

## Rebuilding (for mentors)

- Reference answers: `python make_reference.py` (uses `solutions/dp.py`).
- Maps: `python make_maps.py` prints the ASCII to paste into `maps.py`.
- The world figures for the PDF come from `plotting.py`.
- PDFs use the Aptos body font, so compile with **LuaLaTeX**, twice:
  ```bash
  lualatex assignment.tex && lualatex assignment.tex
  cd solutions && lualatex solutions.tex && lualatex solutions.tex
  ```
- Sanity check: copy `solutions/dp.py` over `dp.py`, run `python utils/check.py` (all pass),
  then `python utils/run.py` (all figures written, including `rollout.mp4`).
