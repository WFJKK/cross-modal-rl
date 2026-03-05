"""Image renderer for mechanical plate compliance dataset.

Takes a PlateConfig and renders a technical drawing PNG using matplotlib.

Annotation levels:
    full: all diameters, edge distances, spacing labeled
    partial: some annotations randomly hidden
    minimal: hole IDs only, scale bar, no dimension labels

Elements:
    Plate outline, holes (circles + crosshairs + labels), bolts (filled
    vs empty), zone dividers (dashed line), scale bar, dimension lines
    (edge distances, diameters, spacing).
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib
import matplotlib.patches as patches
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sampler import Hole, PlateConfig, min_edge_distance  # noqa: E402

MARGIN_LEFT: int = 25
MARGIN_RIGHT: int = 15
MARGIN_BOTTOM: int = 25
MARGIN_TOP: int = 15

DIM_OFFSET: int = 8
DIM_TEXT_SIZE: int = 7
HOLE_LABEL_SIZE: int = 8
ZONE_LABEL_SIZE: int = 9
TITLE_SIZE: int = 10

PLATE_COLOR: str = "#F5F5F0"
PLATE_EDGE_COLOR: str = "black"
HOLE_EDGE_COLOR: str = "black"
BOLT_COLOR: str = "#555555"
EMPTY_HOLE_COLOR: str = "white"
DIM_COLOR: str = "#333333"
ZONE_LINE_COLOR: str = "#888888"
CROSSHAIR_COLOR: str = "#999999"


def plate_to_fig(
    cx: float, cy: float, origin_x: float, origin_y: float, scale: float
) -> tuple[float, float]:
    """Convert plate coordinates (mm) to figure coordinates.

    Args:
        cx: X position in plate space (mm).
        cy: Y position in plate space (mm).
        origin_x: Figure-space X origin of the plate.
        origin_y: Figure-space Y origin of the plate.
        scale: Inches per mm.

    Returns:
        (fig_x, fig_y) tuple in figure coordinates.
    """
    return origin_x + cx * scale, origin_y + cy * scale


def draw_plate(
    ax: Any, pw: float, ph: float, ox: float, oy: float, scale: float
) -> None:
    """Draw the plate rectangle.

    Args:
        ax: Matplotlib axes.
        pw: Plate width in mm.
        ph: Plate height in mm.
        ox: Figure-space X origin.
        oy: Figure-space Y origin.
        scale: Inches per mm.
    """
    rect = patches.Rectangle(
        (ox, oy),
        pw * scale,
        ph * scale,
        linewidth=1.5,
        edgecolor=PLATE_EDGE_COLOR,
        facecolor=PLATE_COLOR,
        zorder=1,
    )
    ax.add_patch(rect)


def draw_hole(ax: Any, hole: Hole, ox: float, oy: float, scale: float) -> None:
    """Draw a single hole with crosshairs and bolt fill.

    Bolted holes are rendered as filled circles with an inner bolt-head
    circle. Empty holes are white circles.

    Args:
        ax: Matplotlib axes.
        hole: The Hole to draw.
        ox: Figure-space X origin.
        oy: Figure-space Y origin.
        scale: Inches per mm.
    """
    fx, fy = plate_to_fig(hole.cx, hole.cy, ox, oy, scale)
    r = (hole.diameter / 2) * scale

    if hole.has_bolt:
        circle = patches.Circle(
            (fx, fy),
            r,
            facecolor=BOLT_COLOR,
            edgecolor=HOLE_EDGE_COLOR,
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(circle)
        inner = patches.Circle(
            (fx, fy),
            r * 0.5,
            facecolor="#777777",
            edgecolor=HOLE_EDGE_COLOR,
            linewidth=0.5,
            zorder=4,
        )
        ax.add_patch(inner)
    else:
        circle = patches.Circle(
            (fx, fy),
            r,
            facecolor=EMPTY_HOLE_COLOR,
            edgecolor=HOLE_EDGE_COLOR,
            linewidth=1.0,
            zorder=3,
        )
        ax.add_patch(circle)

    ch_len = r * 1.5
    ax.plot(
        [fx - ch_len, fx + ch_len],
        [fy, fy],
        color=CROSSHAIR_COLOR,
        linewidth=0.5,
        zorder=2,
    )
    ax.plot(
        [fx, fx],
        [fy - ch_len, fy + ch_len],
        color=CROSSHAIR_COLOR,
        linewidth=0.5,
        zorder=2,
    )


def draw_hole_label(ax: Any, hole: Hole, ox: float, oy: float, scale: float) -> None:
    """Draw hole ID label above the hole.

    Args:
        ax: Matplotlib axes.
        hole: The Hole to label.
        ox: Figure-space X origin.
        oy: Figure-space Y origin.
        scale: Inches per mm.
    """
    fx, fy = plate_to_fig(hole.cx, hole.cy, ox, oy, scale)
    r = (hole.diameter / 2) * scale
    ax.text(
        fx,
        fy + r + 2 * scale,
        hole.id,
        ha="center",
        va="bottom",
        fontsize=HOLE_LABEL_SIZE,
        fontweight="bold",
        zorder=5,
    )


def draw_zone_divider(
    ax: Any,
    zones: dict,
    pw: float,
    ox: float,
    oy: float,
    scale: float,
) -> None:
    """Draw dashed zone boundary lines with labels.

    Args:
        ax: Matplotlib axes.
        zones: Zone definitions dict with y_min/y_max per zone.
        pw: Plate width in mm.
        ox: Figure-space X origin.
        oy: Figure-space Y origin.
        scale: Inches per mm.
    """
    zone_names = sorted(zones.keys())
    if len(zone_names) < 2:
        return

    boundaries: set[float] = set()
    for z in zone_names:
        boundaries.add(zones[z]["y_min"])
        boundaries.add(zones[z]["y_max"])

    for z in zone_names:
        y_min = zones[z]["y_min"]
        y_max = zones[z]["y_max"]
        mid_y = (y_min + y_max) / 2
        fx_label = ox - 5 * scale
        fy_label = oy + mid_y * scale
        ax.text(
            fx_label,
            fy_label,
            f"Zone {z}",
            ha="right",
            va="center",
            fontsize=ZONE_LABEL_SIZE,
            fontstyle="italic",
            color=ZONE_LINE_COLOR,
            zorder=5,
        )

    y_vals = sorted(boundaries)
    for y in y_vals:
        if y == 0 or y == max(y_vals):
            continue
        fy = oy + y * scale
        ax.plot(
            [ox, ox + pw * scale],
            [fy, fy],
            color=ZONE_LINE_COLOR,
            linewidth=1.0,
            linestyle="--",
            zorder=2,
        )


def draw_scale_bar(ax: Any, ox: float, oy: float, scale: float, pw: float) -> None:
    """Draw a scale bar below the plate.

    Chooses the smallest round length that exceeds 15 % of the plate width.

    Args:
        ax: Matplotlib axes.
        ox: Figure-space X origin.
        oy: Figure-space Y origin.
        scale: Inches per mm.
        pw: Plate width in mm.
    """
    for bar_mm in [10, 20, 25, 50]:
        if bar_mm * scale > 0.15 * pw * scale:
            break

    bar_x = ox
    bar_y = oy - 12 * scale
    bar_len = bar_mm * scale

    ax.plot(
        [bar_x, bar_x + bar_len],
        [bar_y, bar_y],
        color="black",
        linewidth=2,
        zorder=5,
    )
    ax.plot(
        [bar_x, bar_x],
        [bar_y - 1 * scale, bar_y + 1 * scale],
        color="black",
        linewidth=1.5,
        zorder=5,
    )
    ax.plot(
        [bar_x + bar_len, bar_x + bar_len],
        [bar_y - 1 * scale, bar_y + 1 * scale],
        color="black",
        linewidth=1.5,
        zorder=5,
    )
    ax.text(
        bar_x + bar_len / 2,
        bar_y - 2.5 * scale,
        f"{bar_mm} mm",
        ha="center",
        va="top",
        fontsize=DIM_TEXT_SIZE,
        zorder=5,
    )


def draw_dim_horizontal(
    ax: Any,
    x1: float,
    x2: float,
    y: float,
    label: str,
    scale: float,
    side: str = "below",
    offset: float | None = None,
) -> None:
    """Draw a horizontal dimension line with arrows and label.

    Args:
        ax: Matplotlib axes.
        x1: Left figure-space X coordinate.
        x2: Right figure-space X coordinate.
        y: Figure-space Y of the feature being dimensioned.
        label: Dimension text (e.g. '120.0').
        scale: Inches per mm.
        side: 'below' or 'above' the feature.
        offset: Override for dimension line offset (figure units).
    """
    if offset is None:
        offset = DIM_OFFSET * scale
    y_dim = y - offset if side == "below" else y + offset

    y_ext_lo = min(y, y_dim) - 1 * scale
    y_ext_hi = max(y, y_dim) + 1 * scale
    ax.plot([x1, x1], [y_ext_lo, y_ext_hi], color=DIM_COLOR, linewidth=0.5, zorder=4)
    ax.plot([x2, x2], [y_ext_lo, y_ext_hi], color=DIM_COLOR, linewidth=0.5, zorder=4)

    ax.annotate(
        "",
        xy=(x2, y_dim),
        xytext=(x1, y_dim),
        arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, linewidth=0.8),
        zorder=4,
    )

    text_y = y_dim - 2 * scale if side == "below" else y_dim + 2 * scale
    va = "top" if side == "below" else "bottom"
    ax.text(
        (x1 + x2) / 2,
        text_y,
        label,
        ha="center",
        va=va,
        fontsize=DIM_TEXT_SIZE,
        color=DIM_COLOR,
        zorder=5,
    )


def draw_dim_vertical(
    ax: Any,
    y1: float,
    y2: float,
    x: float,
    label: str,
    scale: float,
    side: str = "left",
    offset: float | None = None,
) -> None:
    """Draw a vertical dimension line with arrows and label.

    Args:
        ax: Matplotlib axes.
        y1: Lower figure-space Y coordinate.
        y2: Upper figure-space Y coordinate.
        x: Figure-space X of the feature being dimensioned.
        label: Dimension text.
        scale: Inches per mm.
        side: 'left' or 'right' of the feature.
        offset: Override for dimension line offset (figure units).
    """
    if offset is None:
        offset = DIM_OFFSET * scale
    x_dim = x - offset if side == "left" else x + offset

    x_ext_lo = min(x, x_dim) - 1 * scale
    x_ext_hi = max(x, x_dim) + 1 * scale
    ax.plot([x_ext_lo, x_ext_hi], [y1, y1], color=DIM_COLOR, linewidth=0.5, zorder=4)
    ax.plot([x_ext_lo, x_ext_hi], [y2, y2], color=DIM_COLOR, linewidth=0.5, zorder=4)

    ax.annotate(
        "",
        xy=(x_dim, y2),
        xytext=(x_dim, y1),
        arrowprops=dict(arrowstyle="<->", color=DIM_COLOR, linewidth=0.8),
        zorder=4,
    )

    text_x = x_dim - 2 * scale if side == "left" else x_dim + 2 * scale
    ha = "right" if side == "left" else "left"
    ax.text(
        text_x,
        (y1 + y2) / 2,
        label,
        ha=ha,
        va="center",
        fontsize=DIM_TEXT_SIZE,
        color=DIM_COLOR,
        rotation=90,
        zorder=5,
    )


def draw_diameter_label(
    ax: Any, hole: Hole, ox: float, oy: float, scale: float
) -> None:
    """Draw diameter annotation next to a hole.

    Args:
        ax: Matplotlib axes.
        hole: The Hole to annotate.
        ox: Figure-space X origin.
        oy: Figure-space Y origin.
        scale: Inches per mm.
    """
    fx, fy = plate_to_fig(hole.cx, hole.cy, ox, oy, scale)
    r = (hole.diameter / 2) * scale
    ax.text(
        fx + r + 2 * scale,
        fy - 2 * scale,
        f"⌀{hole.diameter}",
        ha="left",
        va="top",
        fontsize=DIM_TEXT_SIZE,
        color=DIM_COLOR,
        zorder=5,
    )


def decide_annotations(config: PlateConfig, rng: np.random.Generator) -> dict[str, Any]:
    """Decide which annotations to show based on annotation_level.

    For 'partial', randomly hides some annotations while ensuring at
    least one violating hole has a hidden annotation (forcing the model
    to reason from the scale bar).

    Args:
        config: PlateConfig with annotation_level set.
        rng: NumPy random generator.

    Returns:
        Dict with keys: show_diameters (set of hole IDs),
        show_edge_distances (set of hole IDs),
        show_spacing (set of (holeA_id, holeB_id) tuples),
        show_plate_dims (bool).
    """
    hole_ids = [h.id for h in config.holes]

    if config.annotation_level == "full":
        return {
            "show_diameters": set(hole_ids),
            "show_edge_distances": set(hole_ids),
            "show_spacing": _all_adjacent_pairs(config),
            "show_plate_dims": True,
        }

    elif config.annotation_level == "partial":
        n_holes = len(hole_ids)

        n_show_dia = max(1, int(n_holes * 0.6))
        show_dia_ids = set(rng.choice(hole_ids, size=n_show_dia, replace=False))

        n_show_edge = max(1, int(n_holes * 0.5))
        show_edge_ids = set(rng.choice(hole_ids, size=n_show_edge, replace=False))

        for hole in config.holes:
            if hole.intended_violations:
                for vid in hole.intended_violations:
                    rule_type = next(r.rule_type for r in config.rules if r.id == vid)
                    if rule_type == "tolerance":
                        show_dia_ids.discard(hole.id)
                        break
                    elif rule_type == "edge_distance":
                        show_edge_ids.discard(hole.id)
                        break
                break

        all_pairs = _all_adjacent_pairs(config)
        n_show_sp = max(0, int(len(all_pairs) * 0.4))
        if all_pairs:
            indices = rng.choice(
                len(all_pairs),
                size=min(n_show_sp, len(all_pairs)),
                replace=False,
            )
            show_spacing = set(all_pairs[i] for i in indices)
        else:
            show_spacing = set()

        return {
            "show_diameters": show_dia_ids,
            "show_edge_distances": show_edge_ids,
            "show_spacing": show_spacing,
            "show_plate_dims": True,
        }

    elif config.annotation_level == "minimal":
        return {
            "show_diameters": set(),
            "show_edge_distances": set(),
            "show_spacing": set(),
            "show_plate_dims": False,
        }

    return {}


def _all_adjacent_pairs(
    config: PlateConfig,
) -> list[tuple[str, str]]:
    """Get pairs of holes in the same zone (candidates for spacing annotations).

    Args:
        config: PlateConfig with holes list.

    Returns:
        List of (hole_id_a, hole_id_b) tuples for same-zone pairs.
    """
    pairs: list[tuple[str, str]] = []
    holes = config.holes
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            if holes[i].zone == holes[j].zone:
                pairs.append((holes[i].id, holes[j].id))
    return pairs


def render_plate(
    config: PlateConfig,
    output_path: str,
    seed: int | None = None,
    dpi: int = 150,
    annotations: dict | None = None,
) -> None:
    """Render a PlateConfig as a technical drawing PNG.

    Args:
        config: PlateConfig from the sampler.
        output_path: Where to save the PNG file.
        seed: Random seed for annotation decisions (partial level).
        dpi: Output resolution.
        annotations: Pre-computed annotation visibility dict. If None,
            computed internally via decide_annotations.
    """
    rng = np.random.default_rng(seed)

    pw = config.plate_width
    ph = config.plate_height

    scale = 6.0 / pw
    fig_w = (MARGIN_LEFT + pw + MARGIN_RIGHT) * scale
    fig_h = (MARGIN_BOTTOM + ph + MARGIN_TOP) * scale

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    ox = MARGIN_LEFT * scale
    oy = MARGIN_BOTTOM * scale

    if annotations is None:
        annotations = decide_annotations(config, rng)

    draw_plate(ax, pw, ph, ox, oy, scale)
    draw_zone_divider(ax, config.zones, pw, ox, oy, scale)

    for hole in config.holes:
        draw_hole(ax, hole, ox, oy, scale)
        draw_hole_label(ax, hole, ox, oy, scale)

    if annotations["show_plate_dims"]:
        draw_dim_horizontal(
            ax,
            ox,
            ox + pw * scale,
            oy,
            f"{pw:.0f}",
            scale,
            side="below",
            offset=6 * scale,
        )
        draw_dim_vertical(
            ax,
            oy,
            oy + ph * scale,
            ox,
            f"{ph:.0f}",
            scale,
            side="left",
            offset=6 * scale,
        )

    for hole in config.holes:
        if hole.id in annotations["show_diameters"]:
            draw_diameter_label(ax, hole, ox, oy, scale)

    for hole in config.holes:
        if hole.id in annotations["show_edge_distances"]:
            _draw_edge_distance(ax, hole, pw, ph, ox, oy, scale)

    for pair in annotations["show_spacing"]:
        h1 = next(h for h in config.holes if h.id == pair[0])
        h2 = next(h for h in config.holes if h.id == pair[1])
        _draw_spacing_line(ax, h1, h2, ox, oy, scale)

    draw_scale_bar(ax, ox, oy, scale, pw)

    ax.text(
        ox + pw * scale / 2,
        oy + ph * scale + 5 * scale,
        f"Plate: {pw:.0f} × {ph:.0f} mm",
        ha="center",
        va="bottom",
        fontsize=TITLE_SIZE,
        fontweight="bold",
        zorder=5,
    )

    plt.tight_layout(pad=0.5)
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    plt.close(fig)


def _draw_edge_distance(
    ax: Any,
    hole: Hole,
    pw: float,
    ph: float,
    ox: float,
    oy: float,
    scale: float,
) -> None:
    """Draw edge distance dimension for the closest edge.

    Args:
        ax: Matplotlib axes.
        hole: The Hole to annotate.
        pw: Plate width in mm.
        ph: Plate height in mm.
        ox: Figure-space X origin.
        oy: Figure-space Y origin.
        scale: Inches per mm.
    """
    edges: dict[str, tuple[float, str]] = {
        "left": (hole.cx, "h"),
        "right": (pw - hole.cx, "h"),
        "bottom": (hole.cy, "v"),
        "top": (ph - hole.cy, "v"),
    }
    closest = min(edges, key=lambda k: edges[k][0])
    dist = edges[closest][0]
    direction = edges[closest][1]

    fx, fy = plate_to_fig(hole.cx, hole.cy, ox, oy, scale)

    if closest == "left" and direction == "h":
        draw_dim_horizontal(
            ax,
            ox,
            fx,
            fy,
            f"{dist:.1f}",
            scale,
            side="below",
            offset=4 * scale,
        )
    elif closest == "right" and direction == "h":
        draw_dim_horizontal(
            ax,
            fx,
            ox + pw * scale,
            fy,
            f"{dist:.1f}",
            scale,
            side="below",
            offset=4 * scale,
        )
    elif closest == "bottom" and direction == "v":
        draw_dim_vertical(
            ax,
            oy,
            fy,
            fx,
            f"{dist:.1f}",
            scale,
            side="left",
            offset=4 * scale,
        )
    elif closest == "top" and direction == "v":
        draw_dim_vertical(
            ax,
            fy,
            oy + ph * scale,
            fx,
            f"{dist:.1f}",
            scale,
            side="right",
            offset=4 * scale,
        )


def _draw_spacing_line(
    ax: Any,
    h1: Hole,
    h2: Hole,
    ox: float,
    oy: float,
    scale: float,
) -> None:
    """Draw spacing dimension between two holes.

    Renders a dotted line between centres with a label offset
    perpendicular to the connecting line.

    Args:
        ax: Matplotlib axes.
        h1: First Hole.
        h2: Second Hole.
        ox: Figure-space X origin.
        oy: Figure-space Y origin.
        scale: Inches per mm.
    """
    fx1, fy1 = plate_to_fig(h1.cx, h1.cy, ox, oy, scale)
    fx2, fy2 = plate_to_fig(h2.cx, h2.cy, ox, oy, scale)

    dist = np.sqrt((h1.cx - h2.cx) ** 2 + (h1.cy - h2.cy) ** 2)

    ax.plot(
        [fx1, fx2],
        [fy1, fy2],
        color=DIM_COLOR,
        linewidth=0.5,
        linestyle=":",
        zorder=3,
    )

    mid_x = (fx1 + fx2) / 2
    mid_y = (fy1 + fy2) / 2

    dx = fx2 - fx1
    dy = fy2 - fy1
    length = np.sqrt(dx**2 + dy**2)
    if length > 0:
        perp_x = -dy / length * 3 * scale
        perp_y = dx / length * 3 * scale
    else:
        perp_x, perp_y = 0, 3 * scale

    ax.text(
        mid_x + perp_x,
        mid_y + perp_y,
        f"{dist:.1f}",
        ha="center",
        va="center",
        fontsize=DIM_TEXT_SIZE - 1,
        color=DIM_COLOR,
        zorder=5,
        bbox=dict(
            boxstyle="round,pad=0.2",
            facecolor="white",
            edgecolor="none",
            alpha=0.8,
        ),
    )


if __name__ == "__main__":
    from sampler import sample_plate_with_retry

    output_dir = "/home/claude/test_renders"
    os.makedirs(output_dir, exist_ok=True)

    test_cases = [
        (0, "simple", 1, "full"),
        (10, "multi_rule", 2, "full"),
        (20, "conditional", 2, "full"),
        (0, "simple", 1, "partial"),
        (0, "simple", 1, "minimal"),
    ]

    for seed, comp, nv, annot in test_cases:
        cfg = sample_plate_with_retry(
            num_violations=nv,
            rule_complexity=comp,
            annotation_level=annot,
            seed=seed,
        )
        if not cfg:
            print(f"FAILED: seed={seed} {comp} nv={nv} {annot}")
            continue

        cfg.annotation_level = annot
        fname = f"plate_{comp}_{annot}_seed{seed}.png"
        path = os.path.join(output_dir, fname)
        render_plate(cfg, path, seed=seed)
        print(
            f"Rendered: {fname} ({len(cfg.holes)} holes, "
            f"{cfg.plate_width:.0f}x{cfg.plate_height:.0f}mm)"
        )
