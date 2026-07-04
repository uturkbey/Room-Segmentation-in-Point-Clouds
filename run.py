#!/usr/bin/env python
"""Segment a point cloud into rooms; write labeled PLY + renders.

Usage: python run.py examples/multiRoom_input1.ply
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from roomseg import segment
from roomseg.io import load_ply, save_labeled_ply


def render_pf(pf2d, path):
    fig, ax = plt.subplots(figsize=(8, 8 * pf2d.shape[1] / pf2d.shape[0]))
    im = ax.imshow(np.where(pf2d.T > 0, pf2d.T, np.nan), origin="lower", cmap="jet")
    fig.colorbar(im, ax=ax, label="PF [m]", shrink=0.8)
    ax.set(title="2D anisotropic potential field", xticks=[], yticks=[])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def render_cloud(xyz, colors, path, max_pts=60_000, frames=72, fps=10):
    """Render a spinning point cloud; .gif animates, other suffixes are static."""
    keep = np.random.default_rng(0).permutation(len(xyz))[:max_pts]
    p = xyz[keep]
    fig = plt.figure(figsize=(8, 5.6))
    ax = fig.add_subplot(projection="3d")
    ax.scatter(*p.T, s=0.5, c=colors[keep])
    ax.set_box_aspect(p.max(0) - p.min(0))
    ax.view_init(elev=40, azim=-60)
    ax.set_axis_off()
    ax.set_position((0, 0, 1, 1))
    if str(path).endswith(".gif"):
        spin = FuncAnimation(fig, lambda i: ax.view_init(40, -60 + i * 360 / frames),
                             frames=frames)
        spin.save(path, writer=PillowWriter(fps=fps), dpi=64)
    else:
        fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def render_stages(xyz, labels, art, path, max_pts=60_000):
    """Six-panel overview mirroring Fig. 2 of the paper."""
    interior, pf, pf2d = art["interior"], art["pf"], art["pf2d"]
    keep = np.random.default_rng(0).permutation(len(xyz))[:max_pts]
    p, lab = xyz[keep], labels[keep]
    z_colors = plt.cm.viridis((p[:, 2] - p[:, 2].min()) / np.ptp(p[:, 2]))
    room_colors = plt.cm.tab20(lab % 20)
    room_colors[lab < 0] = (0.8, 0.8, 0.8, 1)
    stacks = np.argwhere(pf2d > 0)
    maxima = np.column_stack([stacks, pf.argmax(2)[tuple(stacks.T)]])
    pf_colors = plt.cm.jet(pf2d[tuple(stacks.T)] / pf2d.max())
    lab2d = np.full(pf2d.shape, -1)
    lab2d[tuple(art["cells"].T)] = art["cell_labels"]
    lab_rgb = plt.cm.tab20(lab2d % 20)
    lab_rgb[lab2d < 0] = 1  # white background

    fig = plt.figure(figsize=(26, 4.5))
    captions = []
    for pos, title, pts, colors in [
            (1, "(a) input point cloud", p, z_colors),
            (2, "(b) interior free space", np.argwhere(interior), "firebrick"),
            (3, "(c) maxima of 3D PF", maxima, pf_colors),
            (6, "(f) labeled point cloud", p, room_colors)]:
        ax = fig.add_subplot(1, 6, pos, projection="3d")
        ax.scatter(*pts.T, s=0.5 if len(pts) > 30_000 else 2, c=colors)
        ax.set_box_aspect(np.ptp(pts, axis=0), zoom=1.4)
        ax.view_init(elev=40, azim=-60)
        captions.append((ax, title))
    for pos, title, img in [
            (4, "(d) 2D PF", np.where(pf2d > 0, pf2d, np.nan).T),
            (5, "(e) labeled image", lab_rgb.transpose(1, 0, 2))]:
        ax = fig.add_subplot(1, 6, pos)
        ax.imshow(img, origin="lower", cmap="jet")
        captions.append((ax, title))
    fig.subplots_adjust(left=0.002, right=0.998, top=0.93, bottom=0.02, wspace=0.04)
    for ax, title in captions:
        ax.set_axis_off()
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.97, title, ha="center", va="center", fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="input .ply point cloud")
    ap.add_argument("-o", "--out", default="out", help="output directory")
    ap.add_argument("--voxel", type=float, default=0.18, help="voxel size [m]")
    ap.add_argument("--up", choices="yz", default="y", help="vertical axis of the data")
    ap.add_argument("--vis-targets", type=int, help="subsample visibility targets")
    args = ap.parse_args()

    xyz, normals = load_ply(args.input, args.up)
    labels, art = segment(xyz, normals, voxel=args.voxel, vis_targets=args.vis_targets)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = Path(args.input).stem
    save_labeled_ply(out / f"{stem}_labeled.ply", xyz, normals, labels, args.up)
    render_pf(art["pf2d"], out / f"{stem}_pf.png")
    render_stages(xyz, labels, art, out / f"{stem}_stages.png")
    z = xyz[:, 2]
    render_cloud(xyz, plt.cm.viridis((z - z.min()) / np.ptp(z)), out / f"{stem}_input.gif")
    room_colors = plt.cm.tab20(labels % 20)
    room_colors[labels < 0] = (0.8, 0.8, 0.8, 1)
    render_cloud(xyz, room_colors, out / f"{stem}_rooms.gif")

    print(f"{'stage':<18}{'time [s]':>10}{'peak RSS [MB]':>16}")
    for name, sec, rss in art["bench"]:
        print(f"{name:<18}{sec:>10.2f}{rss:>16.0f}")
    print(f"{'total':<18}{sum(b[1] for b in art['bench']):>10.2f}")
    print(f"\n{len(xyz):,} points -> {art['n_rooms']} rooms, "
          f"{(labels >= 0).mean():.1%} of points labeled -> {out}/{stem}_*")


if __name__ == "__main__":
    main()
