"""Plot training curves from the tensorboard event files written by training.

usage:
    python plot_training.py                          #the two newest runs in logs/
    python plot_training.py out.png "label=path" ... #explicit runs, oldest first
"""
import sys
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_GLOB = "logs/DDQN_Minesweeper/events.out.tfevents.*"

#Categorical slots 1 and 2 of the reference palette, validated for colour vision
#deficiency against this surface. Colour follows the run, the same in every panel.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e5e4e0"
AXIS = "#c9c8c3"
SERIES = ["#2a78d6", "#eb6834"]

def load(path):
    ea = EventAccumulator(path, size_guidance={'scalars': 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()['scalars']:
        ev = ea.Scalars(tag)
        out[tag] = (np.array([e.step for e in ev], dtype=float),
                    np.array([e.value for e in ev], dtype=float))
    return out

def smooth(y, frac=0.02):
    n = max(3, int(len(y) * frac))
    pad = np.r_[np.full(n // 2, y[0]), y, np.full(n - n // 2 - 1, y[-1])]
    return np.convolve(pad, np.ones(n) / n, mode='valid')

def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11.5, color=INK, pad=9, loc='left', fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=9.5, color=INK_SOFT)
    ax.set_ylabel(ylabel, fontsize=9.5, color=INK_SOFT)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3, width=0.8)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v/1000:.0f}k")

#A line plus a dot at its end. The label text stays in ink, the dot carries identity.
def series(ax, steps, values, color, label=None, frac=0.02, raw=True):
    if raw:
        ax.plot(steps, values, color=color, linewidth=0.8, alpha=0.18, zorder=2)
    ax.plot(steps, smooth(values, frac), color=color, linewidth=2.0, label=label, zorder=3)
    ax.plot(steps[-1], smooth(values, frac)[-1], 'o', color=color, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)

def end_label(ax, steps, values, text, frac=0.02, dy=0):
    y = smooth(values, frac)[-1]
    ax.annotate(text, xy=(steps[-1], y), xytext=(-6, 11 + dy), textcoords='offset points',
                fontsize=9.5, color=INK, ha='right', fontweight='bold')

def main():
    args = list(sys.argv[1:])
    out_path = "training_curves.png"
    if args and not ("=" in args[0]):
        out_path = args.pop(0)

    if args:
        runs = [(a.split("=", 1)[0], a.split("=", 1)[1]) for a in args]
    else:
        paths = sorted(glob.glob(LOG_GLOB), key=os.path.getmtime)[-2:]
        names = ["previous run", "current run"][-len(paths):]
        runs = list(zip(names, paths))
    if not runs:
        raise SystemExit(f"no event files found in {LOG_GLOB}")

    data = [(name, load(path)) for name, path in runs]
    current_name, current = data[-1]

    fig = plt.figure(figsize=(12.5, 8.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1], hspace=0.42, wspace=0.26,
                          left=0.07, right=0.97, top=0.815, bottom=0.09)
    ax_len = fig.add_subplot(gs[0, :])
    axes = [ax_len] + [fig.add_subplot(gs[1, i]) for i in range(3)]
    for ax in axes:
        ax.set_facecolor(SURFACE)

    #Episode length is in moves for every run, so it is the one metric that can be
    #compared across runs whose reward functions differ
    for i, (name, d) in enumerate(data):
        if 'rollout/ep_len_mean' not in d:
            continue
        st, va = d['rollout/ep_len_mean']
        series(ax_len, st, va, SERIES[i % len(SERIES)], label=name)
        end_label(ax_len, st, va, f"{smooth(va)[-1]:.1f} moves")
    style(ax_len, "Moves survived per episode", "environment steps", "moves")
    leg = ax_len.legend(frameon=False, fontsize=10, loc='upper left', labelcolor=INK)
    for t in leg.get_texts():
        t.set_color(INK)

    ax_rew, ax_loss, ax_eps = axes[1], axes[2], axes[3]
    colour = SERIES[(len(data) - 1) % len(SERIES)]

    st, va = current['rollout/ep_rew_mean']
    series(ax_rew, st, va, colour)
    style(ax_rew, "Mean episode reward", "environment steps", "reward")
    #Is it still climbing? Compare the last fifth of training with the point before it
    sm = smooth(va)
    cut = int(len(sm) * 0.8)
    gain = sm[-1] - sm[cut]
    ax_rew.annotate(f"last 20% of training: {gain:+.2f}",
                    xy=(st[cut], sm[cut]), xytext=(0.30, 0.12), textcoords='axes fraction',
                    fontsize=9, color=INK,
                    arrowprops=dict(arrowstyle='-|>', color=INK_SOFT, linewidth=1,
                                    shrinkA=0, shrinkB=4))

    if 'train/loss' in current:
        st, va = current['train/loss']
        series(ax_loss, st, va, colour)
        ax_loss.set_yscale('log')
    style(ax_loss, "Training loss", "environment steps", "huber loss (log)")

    if 'rollout/exploration_rate' in current:
        st, va = current['rollout/exploration_rate']
        series(ax_eps, st, va, colour, frac=0.005)
        floor = va.min()
        reached = st[np.argmax(va <= floor + 1e-9)]
        ax_eps.axvline(reached, color=AXIS, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax_eps.annotate(f"floor at {reached/1000:.0f}k", xy=(reached, 0.55),
                        xytext=(6, 0), textcoords='offset points',
                        fontsize=9, color=INK_SOFT, va='center')
    style(ax_eps, "Exploration rate", "environment steps", "epsilon")

    fig.text(0.07, 0.955, "DDQN on Minesweeper (beginner, 8x8, 10 mines)",
             fontsize=15, color=INK, fontweight='bold')
    #Kept short and free of run names, which are long enough to overflow the figure
    fig.text(0.07, 0.925,
             "Moves per episode is the one metric both runs share. Reward and loss below are "
             "the newer run only —\nthe reward function changed between runs, so neither is "
             "comparable across them.",
             fontsize=10, color=INK_SOFT, linespacing=1.5, va='top')

    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    print(f"wrote {out_path}")

if __name__ == "__main__":
    main()
