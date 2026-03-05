"""Parameter sampler for mechanical plate compliance dataset.

Constructive sampling with compliance control:
    1. Decide desired outcome (how many violations, which rules)
    2. Generate rules/spec parameters
    3. Place holes constructively to achieve desired outcome
    4. Verify geometric validity and compliance correctness
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

PLATE_WIDTH_RANGE: tuple[float, float] = (80.0, 160.0)
PLATE_HEIGHT_RANGE: tuple[float, float] = (50.0, 100.0)
NUM_HOLES_RANGE: tuple[int, int] = (3, 8)
SMALL_NOMINAL_DIAMETERS: list[float] = [6.0, 8.0, 10.0]
LARGE_NOMINAL_DIAMETERS: list[float] = [12.0, 14.0, 16.0]
ALL_NOMINAL_DIAMETERS: list[float] = SMALL_NOMINAL_DIAMETERS + LARGE_NOMINAL_DIAMETERS
TOLERANCE_VALUES: list[float] = [0.3, 0.4, 0.5, 0.8, 1.0]
EDGE_MULTIPLIERS: list[float] = [1.5, 2.0, 2.5]
SPACING_MINIMUMS: list[float] = [20.0, 25.0, 30.0, 35.0]

MIN_PHYSICAL_MARGIN: float = 3.0
MIN_PHYSICAL_CLEARANCE: float = 2.0

MAX_PLACEMENT_ATTEMPTS: int = 400


@dataclass
class Hole:
    """A single hole with position, diameter, and zone assignment."""

    id: str
    cx: float
    cy: float
    diameter: float
    has_bolt: bool
    zone: str
    intended_violations: list[str] = field(default_factory=list)


@dataclass
class Rule:
    """A design rule constraining holes.

    Supported rule_type values: 'tolerance', 'edge_distance', 'spacing',
    'bolt_populated'.
    """

    id: str
    rule_type: str
    text: str
    params: dict = field(default_factory=dict)


@dataclass
class PlateConfig:
    """Complete plate configuration: geometry, holes, rules, and annotation settings."""

    plate_width: float
    plate_height: float
    holes: list[Hole]
    rules: list[Rule]
    zones: dict
    nominal_diameters: dict
    annotation_level: str
    rule_complexity: str


def hole_inside_plate(
    cx: float, cy: float, radius: float, pw: float, ph: float
) -> bool:
    """Check whether a hole fits entirely within the plate boundaries.

    Args:
        cx: Hole centre X.
        cy: Hole centre Y.
        radius: Hole radius.
        pw: Plate width.
        ph: Plate height.

    Returns:
        True if the hole clears all edges by at least MIN_PHYSICAL_MARGIN.
    """
    return (
        cx - radius >= MIN_PHYSICAL_MARGIN
        and cx + radius <= pw - MIN_PHYSICAL_MARGIN
        and cy - radius >= MIN_PHYSICAL_MARGIN
        and cy + radius <= ph - MIN_PHYSICAL_MARGIN
    )


def holes_dont_overlap(
    cx: float, cy: float, radius: float, existing_holes: list[Hole]
) -> bool:
    """Check that a new hole does not overlap any existing holes.

    Args:
        cx: Candidate hole centre X.
        cy: Candidate hole centre Y.
        radius: Candidate hole radius.
        existing_holes: Already-placed holes.

    Returns:
        True if the candidate clears all existing holes by MIN_PHYSICAL_CLEARANCE.
    """
    for h in existing_holes:
        dist = np.sqrt((cx - h.cx) ** 2 + (cy - h.cy) ** 2)
        if dist < radius + h.diameter / 2 + MIN_PHYSICAL_CLEARANCE:
            return False
    return True


def min_edge_distance(cx: float, cy: float, pw: float, ph: float) -> float:
    """Return the minimum distance from a point to any plate edge.

    Args:
        cx: Point X coordinate.
        cy: Point Y coordinate.
        pw: Plate width.
        ph: Plate height.

    Returns:
        Distance to the nearest plate edge in mm.
    """
    return min(cx, pw - cx, cy, ph - cy)


def pairwise_dist(h1: Hole, h2: Hole) -> float:
    """Euclidean distance between two hole centres.

    Args:
        h1: First hole.
        h2: Second hole.

    Returns:
        Centre-to-centre distance in mm.
    """
    return np.sqrt((h1.cx - h2.cx) ** 2 + (h1.cy - h2.cy) ** 2)


def dist_xy(x1: float, y1: float, x2: float, y2: float) -> float:
    """Euclidean distance between two points.

    Args:
        x1: First point X.
        y1: First point Y.
        x2: Second point X.
        y2: Second point Y.

    Returns:
        Distance in mm.
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def check_tolerance(hole: Hole, rule: Rule) -> tuple[Any, ...]:
    """Check whether a hole diameter is within the tolerance band.

    Args:
        hole: The hole to check.
        rule: A tolerance rule.

    Returns:
        (passes, details_dict) tuple.
    """
    nom = rule.params["nominal"][hole.zone]
    tol = rule.params["tolerance"][hole.zone]
    lo, hi = nom - tol, nom + tol
    passes = lo <= hole.diameter <= hi
    return passes, {
        "rule": rule.id,
        "nominal": nom,
        "tolerance": tol,
        "range": (round(lo, 1), round(hi, 1)),
        "actual": hole.diameter,
    }


def check_edge_distance(
    hole: Hole, rule: Rule, pw: float, ph: float
) -> tuple[Any, ...]:
    """Check whether a hole meets the minimum edge distance requirement.

    If the rule has 'use_nominal' set, the multiplier is applied to the
    nominal diameter (2-hop reasoning) instead of the actual hole diameter.

    Args:
        hole: The hole to check.
        rule: An edge distance rule.
        pw: Plate width.
        ph: Plate height.

    Returns:
        (passes, details_dict) tuple.
    """
    mult = rule.params["multiplier"][hole.zone]
    if rule.params.get("use_nominal"):
        ref_dia = rule.params["nominal_ref"][hole.zone]
    else:
        ref_dia = hole.diameter
    required = mult * ref_dia
    actual = min_edge_distance(hole.cx, hole.cy, pw, ph)
    return actual >= required, {
        "rule": rule.id,
        "multiplier": mult,
        "ref_diameter": round(ref_dia, 1),
        "required": round(required, 2),
        "actual": round(actual, 1),
    }


def check_spacing(hole: Hole, all_holes: list[Hole], rule: Rule) -> tuple[Any, ...]:
    """Check whether the minimum spacing to any neighbour meets the threshold.

    Args:
        hole: The hole to check.
        all_holes: All holes on the plate.
        rule: A spacing rule.

    Returns:
        (passes, details_dict) tuple.
    """
    min_req = rule.params["min_spacing"].get(hole.zone, 0)
    if min_req <= 0:
        return True, {"rule": rule.id, "note": "does not apply to this zone"}
    for other in all_holes:
        if other.id == hole.id or other.zone != hole.zone:
            continue
        d = pairwise_dist(hole, other)
        if d < min_req:
            return False, {
                "rule": rule.id,
                "required": min_req,
                "actual": round(d, 1),
                "other": other.id,
            }
    return True, {"rule": rule.id, "required": min_req}


def check_bolt(hole: Hole, rule: Rule) -> tuple[Any, ...]:
    """Check whether a hole is populated with a bolt as required.

    Args:
        hole: The hole to check.
        rule: A bolt_populated rule.

    Returns:
        (passes, details_dict) tuple.
    """
    return hole.has_bolt, {"rule": rule.id, "has_bolt": hole.has_bolt}


def check_one_rule(
    hole: Hole,
    all_holes: list[Hole],
    rule: Rule,
    pw: float,
    ph: float,
) -> tuple[Any, ...]:
    """Dispatch compliance check to the appropriate rule-type handler.

    Args:
        hole: The hole to check.
        all_holes: All holes on the plate.
        rule: The rule to check.
        pw: Plate width.
        ph: Plate height.

    Returns:
        (passes, details_dict) tuple.
    """
    if rule.rule_type == "tolerance":
        return check_tolerance(hole, rule)
    elif rule.rule_type == "edge_distance":
        return check_edge_distance(hole, rule, pw, ph)
    elif rule.rule_type == "spacing":
        return check_spacing(hole, all_holes, rule)
    elif rule.rule_type == "bolt_populated":
        return check_bolt(hole, rule)
    return True, {}


def check_all_rules(
    hole: Hole,
    all_holes: list[Hole],
    rules: list[Rule],
    pw: float,
    ph: float,
) -> list[tuple[Any, ...]]:
    """Run all rules against a single hole and return results.

    Args:
        hole: The hole to check.
        all_holes: All holes on the plate.
        rules: List of rules.
        pw: Plate width.
        ph: Plate height.

    Returns:
        List of (passes, details_dict) tuples, one per rule.
    """
    return [check_one_rule(hole, all_holes, r, pw, ph) for r in rules]


def generate_rules(
    complexity: str,
    zones: dict,
    nominal_diameters: dict[str, float],
    rng: np.random.Generator,
) -> list[Rule]:
    """Generate design rules with randomised parameters at the given complexity level.

    Args:
        complexity: 'simple', 'multi_rule', or 'conditional'.
        zones: Zone definitions dict.
        nominal_diameters: Nominal diameter per zone.
        rng: NumPy random generator.

    Returns:
        List of Rule objects.
    """
    rules: list[Rule] = []
    zone_names = sorted(zones.keys())

    if complexity == "simple":
        nom = nominal_diameters[zone_names[0]]
        tol = float(rng.choice(TOLERANCE_VALUES[:3]))
        mult = float(rng.choice(EDGE_MULTIPLIERS[:2]))

        rules.append(
            Rule(
                id="R1",
                rule_type="tolerance",
                text=f"All holes shall have diameter {nom:.1f} ± {tol:.1f} mm.",
                params={
                    "nominal": {z: nom for z in zones},
                    "tolerance": {z: tol for z in zones},
                },
            )
        )
        rules.append(
            Rule(
                id="R2",
                rule_type="edge_distance",
                text=f"Edge distance ≥ {mult:.1f}× hole diameter.",
                params={"multiplier": {z: mult for z in zones}},
            )
        )
        sp = float(rng.choice(SPACING_MINIMUMS))
        rules.append(
            Rule(
                id="R3",
                rule_type="spacing",
                text=f"Minimum hole spacing: {sp:.1f} mm.",
                params={"min_spacing": {z: sp for z in zones}},
            )
        )

    elif complexity == "multi_rule":
        nom = nominal_diameters[zone_names[0]]
        tol = float(rng.choice(TOLERANCE_VALUES))
        mult = float(rng.choice(EDGE_MULTIPLIERS))
        sp = float(rng.choice(SPACING_MINIMUMS))

        rules.append(
            Rule(
                id="R1",
                rule_type="tolerance",
                text=f"All holes shall have diameter {nom:.1f} ± {tol:.1f} mm.",
                params={
                    "nominal": {z: nom for z in zones},
                    "tolerance": {z: tol for z in zones},
                },
            )
        )
        rules.append(
            Rule(
                id="R2",
                rule_type="edge_distance",
                text=(
                    f"Edge distance ≥ {mult:.1f}× the nominal diameter "
                    f"specified in Rule R1."
                ),
                params={
                    "multiplier": {z: mult for z in zones},
                    "use_nominal": True,
                    "nominal_ref": {z: nom for z in zones},
                },
            )
        )
        rules.append(
            Rule(
                id="R3",
                rule_type="spacing",
                text=f"Minimum hole spacing: {sp:.1f} mm.",
                params={"min_spacing": {z: sp for z in zones}},
            )
        )
        rules.append(
            Rule(
                id="R4",
                rule_type="bolt_populated",
                text="All bolt holes must be populated with fasteners.",
                params={},
            )
        )

    elif complexity == "conditional":
        tol_nom: dict[str, float] = {}
        tol_tol: dict[str, float] = {}
        for z in zone_names:
            tol_nom[z] = nominal_diameters[z]
            tol_tol[z] = float(rng.choice(TOLERANCE_VALUES))
        parts = [
            f"Zone {z}: {tol_nom[z]:.1f} ± {tol_tol[z]:.1f} mm" for z in zone_names
        ]
        rules.append(
            Rule(
                id="R1",
                rule_type="tolerance",
                text=f'Diameter tolerances by zone. {". ".join(parts)}.',
                params={"nominal": tol_nom, "tolerance": tol_tol},
            )
        )

        edge_m: dict[str, float] = {}
        for z in zone_names:
            zone_h = zones[z]["y_max"] - zones[z]["y_min"]
            max_dia = tol_nom[z] + tol_tol[z]
            max_feasible = (
                (zone_h / 2 - MIN_PHYSICAL_MARGIN) / max_dia if max_dia > 0 else 3.0
            )
            feasible = [m for m in EDGE_MULTIPLIERS if m < max_feasible]
            edge_m[z] = float(rng.choice(feasible)) if feasible else 1.5
        parts = [f"Zone {z}: ≥ {edge_m[z]:.1f}× diameter" for z in zone_names]
        rules.append(
            Rule(
                id="R2",
                rule_type="edge_distance",
                text=f'Edge distance by zone. {". ".join(parts)}.',
                params={"multiplier": edge_m},
            )
        )

        sp_zone = rng.choice(zone_names)
        sp = float(rng.choice(SPACING_MINIMUMS))
        sp_dict = {z: (sp if z == sp_zone else 0.0) for z in zone_names}
        rules.append(
            Rule(
                id="R3",
                rule_type="spacing",
                text=(
                    f"Minimum spacing in Zone {sp_zone}: {sp:.1f} mm. "
                    f"Does not apply to other zones."
                ),
                params={"min_spacing": sp_dict},
            )
        )

        rules.append(
            Rule(
                id="R4",
                rule_type="bolt_populated",
                text="All bolt holes must be populated with fasteners.",
                params={},
            )
        )

    return rules


def place_compliant(
    hole_id: str,
    zone_name: str,
    zones: dict,
    rules: list[Rule],
    existing: list[Hole],
    pw: float,
    ph: float,
    nominal_diameters: dict[str, float],
    rng: np.random.Generator,
    max_attempts: int = MAX_PLACEMENT_ATTEMPTS,
) -> Hole | None:
    """Place a hole that satisfies ALL rules.

    Iteratively samples diameters and positions, checking tolerance,
    edge distance, spacing, and bolt rules until a valid placement is
    found or max_attempts is reached.

    Args:
        hole_id: Identifier for the new hole (e.g. 'H3').
        zone_name: Zone to place the hole in.
        zones: Zone definitions dict.
        rules: List of all rules.
        existing: Already-placed holes.
        pw: Plate width.
        ph: Plate height.
        nominal_diameters: Nominal diameter per zone.
        rng: NumPy random generator.
        max_attempts: Maximum placement attempts.

    Returns:
        A compliant Hole, or None if placement failed.
    """
    bounds = zones[zone_name]

    tol_rule = next((r for r in rules if r.rule_type == "tolerance"), None)
    edge_rule = next((r for r in rules if r.rule_type == "edge_distance"), None)
    spacing_rule = next((r for r in rules if r.rule_type == "spacing"), None)

    for _ in range(max_attempts):
        if tol_rule:
            nom = tol_rule.params["nominal"][zone_name]
            tol = tol_rule.params["tolerance"][zone_name]
            dia = round(float(rng.uniform(nom - tol * 0.8, nom + tol * 0.8)), 1)
        else:
            dia = nominal_diameters[zone_name]

        if edge_rule:
            mult = edge_rule.params["multiplier"][zone_name]
            if edge_rule.params.get("use_nominal"):
                ref_dia = edge_rule.params["nominal_ref"][zone_name]
            else:
                ref_dia = dia
            safe_margin = mult * ref_dia + 0.5
        else:
            safe_margin = dia / 2 + MIN_PHYSICAL_MARGIN

        x_lo = max(safe_margin, dia / 2 + MIN_PHYSICAL_MARGIN)
        x_hi = pw - x_lo
        y_lo = max(
            bounds["y_min"] + safe_margin,
            bounds["y_min"] + dia / 2 + MIN_PHYSICAL_MARGIN,
        )
        y_hi = min(
            bounds["y_max"] - safe_margin,
            bounds["y_max"] - dia / 2 - MIN_PHYSICAL_MARGIN,
        )

        if x_lo >= x_hi or y_lo >= y_hi:
            continue

        cx = round(float(rng.uniform(x_lo, x_hi)), 1)
        cy = round(float(rng.uniform(y_lo, y_hi)), 1)

        if not hole_inside_plate(cx, cy, dia / 2, pw, ph):
            continue
        if not holes_dont_overlap(cx, cy, dia / 2, existing):
            continue

        if spacing_rule:
            min_sp = spacing_rule.params["min_spacing"].get(zone_name, 0)
            if min_sp > 0:
                too_close = any(
                    dist_xy(cx, cy, h.cx, h.cy) < min_sp + 0.5
                    for h in existing
                    if h.zone == zone_name
                )
                if too_close:
                    continue

        hole = Hole(
            id=hole_id,
            cx=cx,
            cy=cy,
            diameter=dia,
            has_bolt=True,
            zone=zone_name,
        )

        temp = existing + [hole]
        results = check_all_rules(hole, temp, rules, pw, ph)
        if all(p for p, _ in results):
            return hole

    return None


def place_violating(
    hole_id: str,
    zone_name: str,
    target_rule_id: str,
    zones: dict,
    rules: list[Rule],
    existing: list[Hole],
    pw: float,
    ph: float,
    nominal_diameters: dict[str, float],
    rng: np.random.Generator,
    max_attempts: int = MAX_PLACEMENT_ATTEMPTS,
) -> Hole | None:
    """Place a hole that violates EXACTLY target_rule_id and passes all others.

    Uses rule-type-specific strategies: oversized/undersized diameter for
    tolerance violations, close-to-edge placement for edge distance
    violations, close-to-neighbour placement for spacing violations, and
    bolt removal for bolt violations.

    Args:
        hole_id: Identifier for the new hole.
        zone_name: Zone to place the hole in.
        target_rule_id: ID of the rule to violate.
        zones: Zone definitions dict.
        rules: List of all rules.
        existing: Already-placed holes.
        pw: Plate width.
        ph: Plate height.
        nominal_diameters: Nominal diameter per zone.
        rng: NumPy random generator.
        max_attempts: Maximum placement attempts.

    Returns:
        A Hole violating exactly the target rule, or None if placement failed.
    """
    bounds = zones[zone_name]
    target_rule = next(r for r in rules if r.id == target_rule_id)
    other_rules = [r for r in rules if r.id != target_rule_id]

    tol_rule = next((r for r in rules if r.rule_type == "tolerance"), None)
    edge_rule = next((r for r in rules if r.rule_type == "edge_distance"), None)

    for _ in range(max_attempts):
        if target_rule.rule_type == "tolerance":
            nom = target_rule.params["nominal"][zone_name]
            tol = target_rule.params["tolerance"][zone_name]
            if rng.random() < 0.5:
                dia = round(float(rng.uniform(nom + tol + 0.1, nom + tol + 2.0)), 1)
            else:
                dia = round(
                    float(max(2.0, rng.uniform(nom - tol - 2.0, nom - tol - 0.1))),
                    1,
                )

            if edge_rule:
                mult = edge_rule.params["multiplier"][zone_name]
                if edge_rule.params.get("use_nominal"):
                    ref_dia = edge_rule.params["nominal_ref"][zone_name]
                else:
                    ref_dia = dia
                safe = mult * ref_dia + 0.5
            else:
                safe = dia / 2 + MIN_PHYSICAL_MARGIN

            x_lo = max(safe, dia / 2 + MIN_PHYSICAL_MARGIN)
            x_hi = pw - x_lo
            y_lo = max(
                bounds["y_min"] + safe,
                bounds["y_min"] + dia / 2 + MIN_PHYSICAL_MARGIN,
            )
            y_hi = min(
                bounds["y_max"] - safe,
                bounds["y_max"] - dia / 2 - MIN_PHYSICAL_MARGIN,
            )
            if x_lo >= x_hi or y_lo >= y_hi:
                continue
            cx = round(float(rng.uniform(x_lo, x_hi)), 1)
            cy = round(float(rng.uniform(y_lo, y_hi)), 1)
            has_bolt = True

        elif target_rule.rule_type == "edge_distance":
            if tol_rule:
                nom = tol_rule.params["nominal"][zone_name]
                tol = tol_rule.params["tolerance"][zone_name]
                dia = round(float(rng.uniform(nom - tol * 0.5, nom + tol * 0.5)), 1)
            else:
                dia = nominal_diameters[zone_name]

            mult = target_rule.params["multiplier"][zone_name]
            if target_rule.params.get("use_nominal"):
                ref_dia = target_rule.params["nominal_ref"][zone_name]
            else:
                ref_dia = dia
            required_edge = mult * ref_dia
            phys_min = dia / 2 + MIN_PHYSICAL_MARGIN

            if phys_min >= required_edge:
                continue

            target_edge = float(rng.uniform(phys_min, required_edge - 0.3))

            edge = rng.choice(["left", "right", "top", "bottom"])
            y_lo = bounds["y_min"] + required_edge
            y_hi = bounds["y_max"] - required_edge
            x_lo_safe = required_edge
            x_hi_safe = pw - required_edge
            if edge in ("left", "right") and y_lo >= y_hi:
                continue
            if edge in ("top", "bottom") and x_lo_safe >= x_hi_safe:
                continue
            if edge == "left":
                cx, cy = target_edge, float(rng.uniform(y_lo, y_hi))
            elif edge == "right":
                cx, cy = pw - target_edge, float(rng.uniform(y_lo, y_hi))
            elif edge == "bottom":
                cy_candidate = bounds["y_min"] + target_edge
                if cy_candidate < bounds["y_min"] + dia / 2 + MIN_PHYSICAL_MARGIN:
                    continue
                cx, cy = (
                    float(rng.uniform(x_lo_safe, x_hi_safe)),
                    cy_candidate,
                )
            else:
                cy_candidate = bounds["y_max"] - target_edge
                if cy_candidate > bounds["y_max"] - dia / 2 - MIN_PHYSICAL_MARGIN:
                    continue
                cx, cy = (
                    float(rng.uniform(x_lo_safe, x_hi_safe)),
                    cy_candidate,
                )
            cx, cy = round(cx, 1), round(cy, 1)
            has_bolt = True

        elif target_rule.rule_type == "spacing":
            same_zone = [h for h in existing if h.zone == zone_name]
            if not same_zone:
                continue

            if tol_rule:
                nom = tol_rule.params["nominal"][zone_name]
                tol = tol_rule.params["tolerance"][zone_name]
                dia = round(float(rng.uniform(nom - tol * 0.5, nom + tol * 0.5)), 1)
            else:
                dia = nominal_diameters[zone_name]

            edge_req = dia / 2 + MIN_PHYSICAL_MARGIN
            if edge_rule:
                if edge_rule.params.get("use_nominal"):
                    ref_dia = edge_rule.params["nominal_ref"][zone_name]
                else:
                    ref_dia = dia
                edge_req = max(
                    edge_req,
                    edge_rule.params["multiplier"][zone_name] * ref_dia + 0.5,
                )

            shuffled_neighbors = list(same_zone)
            rng.shuffle(shuffled_neighbors)
            neighbor = shuffled_neighbors[0]

            min_sp = target_rule.params["min_spacing"][zone_name]
            phys_min = dia / 2 + neighbor.diameter / 2 + MIN_PHYSICAL_CLEARANCE
            if phys_min >= min_sp:
                continue
            target_d = float(rng.uniform(phys_min, min_sp - 0.3))

            placed = False
            angles = rng.uniform(0, 2 * np.pi, size=50)
            for angle in angles:
                cx = round(neighbor.cx + target_d * np.cos(angle), 1)
                cy = round(neighbor.cy + target_d * np.sin(angle), 1)

                if cy < bounds["y_min"] + edge_req or cy > bounds["y_max"] - edge_req:
                    continue
                if cx < edge_req or cx > pw - edge_req:
                    continue

                placed = True
                break

            if not placed:
                continue
            has_bolt = True

        elif target_rule.rule_type == "bolt_populated":
            h = place_compliant(
                hole_id,
                zone_name,
                zones,
                other_rules,
                existing,
                pw,
                ph,
                nominal_diameters,
                rng,
                max_attempts,
            )
            if h:
                h.has_bolt = False
                h.intended_violations = [target_rule_id]
                return h
            continue

        else:
            continue

        if not hole_inside_plate(cx, cy, dia / 2, pw, ph):
            continue
        if not holes_dont_overlap(cx, cy, dia / 2, existing):
            continue

        hole = Hole(
            id=hole_id,
            cx=cx,
            cy=cy,
            diameter=dia,
            has_bolt=has_bolt,
            zone=zone_name,
            intended_violations=[target_rule_id],
        )

        temp = existing + [hole]
        target_pass, _ = check_one_rule(hole, temp, target_rule, pw, ph)
        if target_pass:
            continue

        all_others_pass = all(
            check_one_rule(hole, temp, r, pw, ph)[0] for r in other_rules
        )
        if not all_others_pass:
            continue

        return hole

    return None


def sample_plate(
    num_violations: int = 1,
    rule_complexity: str = "multi_rule",
    annotation_level: str = "full",
    seed: int | None = None,
    allow_multi_violation: bool = False,
    max_placement_attempts: int | None = None,
) -> PlateConfig | None:
    """Generate one complete plate configuration with controlled compliance.

    Creates a plate with zones, rules, and holes such that exactly the
    requested number of holes violate exactly one rule each (unless
    allow_multi_violation is True, in which case some holes may also
    lose their bolt).

    Args:
        num_violations: Desired number of violating holes.
        rule_complexity: 'simple', 'multi_rule', or 'conditional'.
        annotation_level: 'full', 'partial', or 'minimal'.
        seed: Random seed for reproducibility.
        allow_multi_violation: If True, ~20 % of violating holes also lose
            their bolt, creating dual-violation holes for RL training.
        max_placement_attempts: Override per-hole placement attempts.

    Returns:
        PlateConfig or None if placement failed.
    """
    attempts = max_placement_attempts or MAX_PLACEMENT_ATTEMPTS
    rng = np.random.default_rng(seed)

    pw = round(float(rng.uniform(*PLATE_WIDTH_RANGE)), 0)
    ph = round(float(rng.uniform(*PLATE_HEIGHT_RANGE)), 0)

    if rule_complexity in ("simple", "multi_rule"):
        zones = {"A": {"y_min": 0, "y_max": ph}}
        nom_dias = {"A": float(rng.choice(ALL_NOMINAL_DIAMETERS))}
    else:
        split = round(ph / 2, 0)
        zones = {
            "A": {"y_min": split, "y_max": ph},
            "B": {"y_min": 0, "y_max": split},
        }
        nom_dias = {
            "A": float(rng.choice(SMALL_NOMINAL_DIAMETERS)),
            "B": float(rng.choice(LARGE_NOMINAL_DIAMETERS)),
        }

    rules = generate_rules(rule_complexity, zones, nom_dias, rng)

    n_holes = int(rng.integers(*NUM_HOLES_RANGE, endpoint=True))
    n_viol = min(num_violations, n_holes)
    n_comp = n_holes - n_viol

    violatable = [r.id for r in rules]
    spacing_ids = [r.id for r in rules if r.rule_type == "spacing"]
    weights = np.array([2.0 if rid in spacing_ids else 1.0 for rid in violatable])
    weights = weights / weights.sum()
    viol_plan = list(rng.choice(violatable, size=n_viol, p=weights))

    znames = list(zones.keys())
    hole_zones = [znames[i % len(znames)] for i in range(n_holes)]
    rng.shuffle(hole_zones)

    holes: list[Hole] = []
    idx = 0
    for i in range(n_comp):
        h = place_compliant(
            f"H{idx + 1}",
            hole_zones[idx],
            zones,
            rules,
            holes,
            pw,
            ph,
            nom_dias,
            rng,
            attempts,
        )
        if h is None:
            return None
        holes.append(h)
        idx += 1

    for i in range(n_viol):
        target = viol_plan[i]
        zone = hole_zones[idx]

        rule_obj = next(r for r in rules if r.id == target)
        if rule_obj.rule_type == "spacing":
            sp_val = rule_obj.params["min_spacing"].get(zone, 0)
            if sp_val <= 0:
                alt_zones = [
                    z for z in znames if rule_obj.params["min_spacing"].get(z, 0) > 0
                ]
                if alt_zones:
                    zone = rng.choice(alt_zones)
                else:
                    target = next(r.id for r in rules if r.rule_type == "tolerance")

        h = place_violating(
            f"H{idx + 1}",
            zone,
            target,
            zones,
            rules,
            holes,
            pw,
            ph,
            nom_dias,
            rng,
            attempts,
        )
        if h is None:
            return None
        holes.append(h)
        idx += 1

    if allow_multi_violation:
        bolt_rule = next((r for r in rules if r.rule_type == "bolt_populated"), None)
        if bolt_rule:
            for hole in holes:
                if (
                    hole.intended_violations
                    and bolt_rule.id not in hole.intended_violations
                    and rng.random() < 0.2
                ):
                    hole.has_bolt = False
                    hole.intended_violations.append(bolt_rule.id)

    holes.sort(key=lambda h: int(h.id[1:]))

    for hole in holes:
        results = check_all_rules(hole, holes, rules, pw, ph)
        failed = [det for passed, det in results if not passed]
        failed_ids = [d.get("rule", "") for d in failed]
        if not hole.intended_violations and failed:
            return None
        if hole.intended_violations:
            for vid in hole.intended_violations:
                if vid not in failed_ids:
                    return None
            if len(failed_ids) != len(hole.intended_violations):
                return None

    return PlateConfig(
        plate_width=pw,
        plate_height=ph,
        holes=holes,
        rules=rules,
        zones=zones,
        nominal_diameters=nom_dias,
        annotation_level=annotation_level,
        rule_complexity=rule_complexity,
    )


def sample_plate_with_retry(
    num_violations: int = 1,
    rule_complexity: str = "multi_rule",
    annotation_level: str = "full",
    seed: int | None = None,
    max_retries: int = 30,
    allow_multi_violation: bool = False,
    max_placement_attempts: int | None = None,
) -> PlateConfig | None:
    """Wrapper that retries sample_plate with different seeds on failure.

    Args:
        num_violations: Desired number of violating holes.
        rule_complexity: 'simple', 'multi_rule', or 'conditional'.
        annotation_level: 'full', 'partial', or 'minimal'.
        seed: Base random seed.
        max_retries: Maximum number of retry attempts.
        allow_multi_violation: If True, allow dual-violation holes.
        max_placement_attempts: Override per-hole placement attempts.

    Returns:
        PlateConfig or None if all retries fail.
    """
    for attempt in range(max_retries):
        s = seed * 1000 + attempt if seed is not None else None
        cfg = sample_plate(
            num_violations=num_violations,
            rule_complexity=rule_complexity,
            annotation_level=annotation_level,
            seed=s,
            allow_multi_violation=allow_multi_violation,
            max_placement_attempts=max_placement_attempts,
        )
        if cfg is not None:
            return cfg
    return None


if __name__ == "__main__":
    ok, fail = 0, 0
    complexities = ["simple", "multi_rule", "conditional"]

    for seed in range(100):
        comp = complexities[seed % 3]
        nv = seed % 4

        cfg = sample_plate_with_retry(
            num_violations=nv, rule_complexity=comp, seed=seed
        )

        if cfg:
            ok += 1
            viols = [
                (h.id, h.intended_violations)
                for h in cfg.holes
                if h.intended_violations
            ]
            print(
                f"seed={seed:3d} {comp:12s} nv={nv} holes={len(cfg.holes)} "
                f"plate={cfg.plate_width:.0f}x{cfg.plate_height:.0f} "
                f"violations={viols}"
            )
        else:
            fail += 1
            print(f"seed={seed:3d} {comp:12s} nv={nv} FAILED")

    print(f"\n{'=' * 50}")
    print(f"Success: {ok}/100 ({ok}%)")
    print(f"Failures: {fail}/100")
