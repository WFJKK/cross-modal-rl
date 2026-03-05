"""Question generator for mechanical plate compliance dataset.

Takes a PlateConfig (from sampler) and generates question-answer pairs with
step-by-step reasoning chains. All template-based, no LLM calls. Ground
truth is derived directly from sampler parameters.

Question types:
    1. per_component_compliance — "Does H3 comply with R1?"
    2. full_audit — "List all violations"
    3. measurement_extraction — "What is the distance between H1 and H4?"
    4. rule_selection — "What Material Class applies?" (conditional only)
    5. counterfactual — "What if the material were Steel?"
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from sampler import (
    Hole,
    PlateConfig,
    Rule,
    check_bolt,
    check_edge_distance,
    check_spacing,
    check_tolerance,
    min_edge_distance,
    pairwise_dist,
)


class Question:
    """A question-answer pair with reasoning chain for a compliance check."""

    def __init__(
        self,
        qtype: str,
        question: str,
        answer: str | list[str],
        reasoning: str,
    ) -> None:
        self.qtype = qtype
        self.question = question
        self.answer = answer
        self.reasoning = reasoning

    def to_dict(self) -> dict[str, str | list[str]]:
        """Serialise the question to a JSON-compatible dictionary."""
        return {
            "type": self.qtype,
            "question": self.question,
            "answer": self.answer,
            "reasoning": self.reasoning,
        }


def tolerance_reasoning(
    hole: Hole,
    rule: Rule,
    annotation_level: str = "full",
    dia_visible: bool = True,
) -> str:
    """Build a reasoning chain for a tolerance check.

    Adapts wording depending on whether the diameter annotation is visible:
    exact values when labeled, approximate estimates when not.

    Args:
        hole: The hole being checked.
        rule: The tolerance rule.
        annotation_level: One of 'full', 'partial', 'minimal'.
        dia_visible: Whether this hole's diameter label is shown on the drawing.

    Returns:
        Multi-sentence reasoning string.
    """
    nom = rule.params["nominal"][hole.zone]
    tol = rule.params["tolerance"][hole.zone]
    lo, hi = round(nom - tol, 1), round(nom + tol, 1)
    passes = lo <= hole.diameter <= hi

    steps: list[str] = []
    if annotation_level == "minimal" or (
        annotation_level == "partial" and not dia_visible
    ):
        approx_dia = round(hole.diameter)
        steps.append(
            f"From the scale bar, {hole.id} diameter appears "
            f"approximately {approx_dia}mm."
        )
        steps.append(
            f"Rule {rule.id} specifies nominal {nom:.1f} ± {tol:.1f}mm, "
            f"giving acceptable range {lo}–{hi}mm."
        )
        if passes:
            steps.append(f"~{approx_dia}mm appears within {lo}–{hi}mm. Compliant.")
        else:
            if hole.diameter > hi:
                steps.append(
                    f"~{approx_dia}mm appears to exceed upper limit {hi}mm. "
                    f"Non-compliant."
                )
            else:
                steps.append(
                    f"~{approx_dia}mm appears below lower limit {lo}mm. "
                    f"Non-compliant."
                )
    else:
        steps.append(f"{hole.id} diameter is {hole.diameter}mm.")
        steps.append(
            f"Rule {rule.id} specifies nominal {nom:.1f} ± {tol:.1f}mm, "
            f"giving acceptable range {lo}–{hi}mm."
        )
        if passes:
            steps.append(f"{hole.diameter}mm is within {lo}–{hi}mm. Compliant.")
        else:
            if hole.diameter > hi:
                steps.append(
                    f"{hole.diameter}mm exceeds upper limit {hi}mm "
                    f"by {round(hole.diameter - hi, 1)}mm. Non-compliant."
                )
            else:
                steps.append(
                    f"{hole.diameter}mm is below lower limit {lo}mm "
                    f"by {round(lo - hole.diameter, 1)}mm. Non-compliant."
                )
    return " ".join(steps)


def edge_distance_reasoning(
    hole: Hole,
    rule: Rule,
    pw: float,
    ph: float,
    annotation_level: str = "full",
    edge_visible: bool = True,
    dia_visible: bool = True,
) -> str:
    """Build a reasoning chain for an edge distance check.

    For rules that reference a nominal diameter (use_nominal=True), the
    reasoning shows the two-hop lookup chain through the tolerance rule.

    Args:
        hole: The hole being checked.
        rule: The edge distance rule.
        pw: Plate width in mm.
        ph: Plate height in mm.
        annotation_level: One of 'full', 'partial', 'minimal'.
        edge_visible: Whether edge distance annotations are shown for this hole.
        dia_visible: Whether diameter annotations are shown for this hole.

    Returns:
        Multi-sentence reasoning string.
    """
    mult = rule.params["multiplier"][hole.zone]
    use_nominal = rule.params.get("use_nominal", False)
    if use_nominal:
        ref_dia = rule.params["nominal_ref"][hole.zone]
    else:
        ref_dia = hole.diameter
    required = round(mult * ref_dia, 2)
    actual = round(min_edge_distance(hole.cx, hole.cy, pw, ph), 1)

    edges = {
        "left edge": hole.cx,
        "right edge": pw - hole.cx,
        "bottom edge": hole.cy,
        "top edge": ph - hole.cy,
    }
    closest_edge = min(edges, key=lambda k: edges[k])
    passes = actual >= required

    steps: list[str] = []
    if annotation_level == "minimal" or (
        annotation_level == "partial" and not edge_visible
    ):
        approx_actual = round(actual)
        if use_nominal:
            steps.append(
                f"From the scale bar, {hole.id} appears approximately "
                f"{approx_actual}mm from the {closest_edge}."
            )
            steps.append(
                f"Rule {rule.id} requires ≥ {mult:.1f}× the nominal "
                f"diameter from Rule R1. Nominal is {ref_dia:.1f}mm, "
                f"so required = {mult:.1f} × {ref_dia:.1f} = {required}mm."
            )
        else:
            approx_dia = round(hole.diameter)
            approx_req = round(mult * approx_dia, 1)
            steps.append(
                f"From the scale bar, {hole.id} appears approximately "
                f"{approx_actual}mm from the {closest_edge}."
            )
            steps.append(
                f"Rule {rule.id} requires ≥ {mult:.1f} × ~{approx_dia}mm "
                f"≈ {approx_req}mm."
            )
            required = approx_req
        if passes:
            steps.append(f"~{approx_actual}mm ≥ {required}mm. Compliant.")
        else:
            steps.append(f"~{approx_actual}mm < {required}mm. Non-compliant.")
    else:
        steps.append(
            f"{hole.id} minimum edge distance is {actual}mm (to {closest_edge})."
        )
        if use_nominal:
            steps.append(
                f"Rule {rule.id} requires ≥ {mult:.1f}× the nominal "
                f"diameter from Rule R1. Nominal = {ref_dia:.1f}mm, "
                f"so required = {mult:.1f} × {ref_dia:.1f} = {required}mm."
            )
        else:
            steps.append(
                f"Rule {rule.id} requires ≥ {mult:.1f} × {hole.diameter}mm "
                f"= {required}mm."
            )
        if passes:
            steps.append(f"{actual}mm ≥ {required}mm. Compliant.")
        else:
            steps.append(f"{actual}mm < {required}mm. Non-compliant.")
    return " ".join(steps)


def spacing_reasoning(
    hole: Hole,
    all_holes: list[Hole],
    rule: Rule,
    annotation_level: str = "full",
) -> str:
    """Build a reasoning chain for a spacing check.

    Finds the closest same-zone neighbour and compares to the required
    minimum spacing.

    Args:
        hole: The hole being checked.
        all_holes: All holes on the plate.
        rule: The spacing rule.
        annotation_level: One of 'full', 'partial', 'minimal'.

    Returns:
        Multi-sentence reasoning string.
    """
    min_req = rule.params["min_spacing"].get(hole.zone, 0)

    if min_req <= 0:
        return (
            f"Rule {rule.id} does not apply to Zone {hole.zone}. "
            f"{hole.id} is compliant with this rule."
        )

    closest_dist = float("inf")
    closest_id: str | None = None
    for other in all_holes:
        if other.id == hole.id or other.zone != hole.zone:
            continue
        d = round(pairwise_dist(hole, other), 1)
        if d < closest_dist:
            closest_dist = d
            closest_id = other.id

    if closest_id is None:
        return (
            f"{hole.id} is the only hole in Zone {hole.zone}. "
            f"Spacing rule {rule.id} is satisfied by default."
        )

    passes = closest_dist >= min_req
    steps: list[str] = []
    if annotation_level == "minimal":
        approx_dist = round(closest_dist)
        steps.append(
            f"From the scale bar, the closest same-zone hole to {hole.id} "
            f"appears to be {closest_id} at approximately {approx_dist}mm."
        )
        steps.append(f"Rule {rule.id} requires ≥ {min_req}mm.")
        if passes:
            steps.append(f"~{approx_dist}mm ≥ {min_req}mm. Compliant.")
        else:
            steps.append(f"~{approx_dist}mm < {min_req}mm. Non-compliant.")
    else:
        steps.append(
            f"Closest same-zone hole to {hole.id} is {closest_id} "
            f"at {closest_dist}mm."
        )
        steps.append(f"Rule {rule.id} requires ≥ {min_req}mm.")
        if passes:
            steps.append(f"{closest_dist}mm ≥ {min_req}mm. Compliant.")
        else:
            steps.append(f"{closest_dist}mm < {min_req}mm. Non-compliant.")
    return " ".join(steps)


def bolt_reasoning(hole: Hole, rule: Rule) -> str:
    """Build a reasoning chain for a bolt population check.

    Args:
        hole: The hole being checked.
        rule: The bolt population rule.

    Returns:
        Single-sentence reasoning string.
    """
    if hole.has_bolt:
        return f"{hole.id} has a fastener installed. " f"Compliant with Rule {rule.id}."
    else:
        return (
            f"{hole.id} does not have a fastener installed. "
            f"Rule {rule.id} requires all holes to be populated. "
            f"Non-compliant."
        )


def build_reasoning(
    hole: Hole,
    rule: Rule,
    all_holes: list[Hole],
    pw: float,
    ph: float,
    annotation_level: str = "full",
    annotations: dict | None = None,
) -> str:
    """Route to the correct reasoning builder with annotation awareness.

    Args:
        hole: The hole being checked.
        rule: The rule to check against.
        all_holes: All holes on the plate.
        pw: Plate width in mm.
        ph: Plate height in mm.
        annotation_level: One of 'full', 'partial', 'minimal'.
        annotations: Annotation visibility dict from decide_annotations.

    Returns:
        Reasoning string from the appropriate builder.
    """
    dia_visible = True
    edge_visible = True
    if annotations:
        dia_visible = hole.id in annotations.get("show_diameters", set())
        edge_visible = hole.id in annotations.get("show_edge_distances", set())

    if rule.rule_type == "tolerance":
        return tolerance_reasoning(hole, rule, annotation_level, dia_visible)
    elif rule.rule_type == "edge_distance":
        return edge_distance_reasoning(
            hole, rule, pw, ph, annotation_level, edge_visible, dia_visible
        )
    elif rule.rule_type == "spacing":
        return spacing_reasoning(hole, all_holes, rule, annotation_level)
    elif rule.rule_type == "bolt_populated":
        return bolt_reasoning(hole, rule)
    return ""


def check_passes(
    hole: Hole,
    rule: Rule,
    all_holes: list[Hole],
    pw: float,
    ph: float,
) -> bool:
    """Check whether a hole passes a rule.

    Args:
        hole: The hole to check.
        rule: The rule to check against.
        all_holes: All holes on the plate (needed for spacing).
        pw: Plate width in mm.
        ph: Plate height in mm.

    Returns:
        True if the hole passes the rule, False otherwise.
    """
    if rule.rule_type == "tolerance":
        return check_tolerance(hole, rule)[0]
    elif rule.rule_type == "edge_distance":
        return check_edge_distance(hole, rule, pw, ph)[0]
    elif rule.rule_type == "spacing":
        return check_spacing(hole, all_holes, rule)[0]
    elif rule.rule_type == "bolt_populated":
        return check_bolt(hole, rule)[0]
    return True


def gen_per_component(
    config: PlateConfig,
    rng: np.random.Generator,
    annotation_level: str = "full",
    annotations: dict | None = None,
) -> list[Question]:
    """Generate per-component compliance questions.

    Every hole is checked against every rule, producing one Yes/No
    question per (hole, rule) pair.

    Args:
        config: The plate configuration.
        rng: NumPy random generator.
        annotation_level: One of 'full', 'partial', 'minimal'.
        annotations: Annotation visibility dict.

    Returns:
        List of Question objects.
    """
    questions: list[Question] = []
    pw, ph = config.plate_width, config.plate_height

    for hole in config.holes:
        for rule in config.rules:
            passes = check_passes(hole, rule, config.holes, pw, ph)
            reasoning = build_reasoning(
                hole, rule, config.holes, pw, ph, annotation_level, annotations
            )

            questions.append(
                Question(
                    qtype="per_component_compliance",
                    question=f"Does hole {hole.id} comply with Rule {rule.id}?",
                    answer="Yes" if passes else "No",
                    reasoning=reasoning,
                )
            )

    return questions


def gen_full_audit(
    config: PlateConfig,
    annotation_level: str = "full",
    annotations: dict | None = None,
) -> Question:
    """Generate a full audit question listing all violations.

    Iterates over every (hole, rule) pair and collects failures into
    a violation list with per-violation reasoning.

    Args:
        config: The plate configuration.
        annotation_level: One of 'full', 'partial', 'minimal'.
        annotations: Annotation visibility dict.

    Returns:
        A single Question with answer as a list of violation strings.
    """
    pw, ph = config.plate_width, config.plate_height
    violation_list: list[str] = []
    reasoning_parts: list[str] = []

    for hole in config.holes:
        for rule in config.rules:
            passes = check_passes(hole, rule, config.holes, pw, ph)
            if not passes:
                reasoning = build_reasoning(
                    hole, rule, config.holes, pw, ph, annotation_level, annotations
                )
                violation_list.append(f"{hole.id}: Rule {rule.id} violation")
                reasoning_parts.append(f"{hole.id} vs {rule.id}: {reasoning}")

    if violation_list:
        answer: str | list[str] = violation_list
        full_reasoning = (
            f"Checked each hole against each rule. "
            f"Found {len(violation_list)} violation(s). " + " ".join(reasoning_parts)
        )
    else:
        answer = ["No violations found. Design is fully compliant."]
        full_reasoning = "Checked each hole against each rule. All pass."

    return Question(
        qtype="full_audit",
        question="List all rule violations in this design.",
        answer=answer,
        reasoning=full_reasoning,
    )


def gen_measurement_diameter(
    config: PlateConfig,
    rng: np.random.Generator,
    annotation_level: str = "full",
    annotations: dict | None = None,
) -> Question:
    """Generate a diameter measurement question for a random hole.

    Args:
        config: The plate configuration.
        rng: NumPy random generator (for hole selection).
        annotation_level: One of 'full', 'partial', 'minimal'.
        annotations: Annotation visibility dict.

    Returns:
        A Question asking for the diameter of a randomly selected hole.
    """
    hole = config.holes[rng.integers(len(config.holes))]
    dia_visible = True
    if annotations:
        dia_visible = hole.id in annotations.get("show_diameters", set())

    if annotation_level == "minimal" or (
        annotation_level == "partial" and not dia_visible
    ):
        approx_dia = round(hole.diameter)
        reasoning = (
            f"Using the scale bar, {hole.id} appears approximately "
            f"{approx_dia}mm in diameter."
        )
    else:
        reasoning = f"The annotation on {hole.id} reads ⌀{hole.diameter}mm."

    return Question(
        qtype="measurement_diameter",
        question=f"What is the diameter of hole {hole.id}?",
        answer=f"{hole.diameter}mm",
        reasoning=reasoning,
    )


def gen_measurement_edge_distance(
    config: PlateConfig,
    rng: np.random.Generator,
    annotation_level: str = "full",
    annotations: dict | None = None,
) -> Question:
    """Generate an edge distance measurement question for a random hole.

    Args:
        config: The plate configuration.
        rng: NumPy random generator.
        annotation_level: One of 'full', 'partial', 'minimal'.
        annotations: Annotation visibility dict.

    Returns:
        A Question asking for the minimum edge distance of a random hole.
    """
    hole = config.holes[rng.integers(len(config.holes))]
    pw, ph = config.plate_width, config.plate_height
    actual = round(min_edge_distance(hole.cx, hole.cy, pw, ph), 1)

    edges = {
        "left edge": round(hole.cx, 1),
        "right edge": round(pw - hole.cx, 1),
        "bottom edge": round(hole.cy, 1),
        "top edge": round(ph - hole.cy, 1),
    }
    closest_edge = min(edges, key=lambda k: edges[k])

    edge_visible = True
    if annotations:
        edge_visible = hole.id in annotations.get("show_edge_distances", set())

    if annotation_level == "minimal" or (
        annotation_level == "partial" and not edge_visible
    ):
        approx_actual = round(actual)
        reasoning = (
            f"Using the scale bar, {hole.id} appears approximately "
            f"{approx_actual}mm from the {closest_edge}."
        )
    else:
        reasoning = (
            f"{hole.id} is at ({hole.cx}, {hole.cy}) on a "
            f"{pw}×{ph}mm plate. Distances to edges: "
            f"left={edges['left edge']}mm, "
            f"right={edges['right edge']}mm, "
            f"bottom={edges['bottom edge']}mm, "
            f"top={edges['top edge']}mm. "
            f"Minimum is {actual}mm to {closest_edge}."
        )

    return Question(
        qtype="measurement_edge_distance",
        question=f"What is the minimum edge distance of hole {hole.id}?",
        answer=f"{actual}mm (to {closest_edge})",
        reasoning=reasoning,
    )


def gen_measurement_hole_to_hole(
    config: PlateConfig,
    rng: np.random.Generator,
    annotation_level: str = "full",
    annotations: dict | None = None,
) -> Optional[Question]:
    """Generate a hole-to-hole distance measurement question.

    Args:
        config: The plate configuration.
        rng: NumPy random generator.
        annotation_level: One of 'full', 'partial', 'minimal'.
        annotations: Annotation visibility dict.

    Returns:
        A Question, or None if the plate has fewer than 2 holes.
    """
    if len(config.holes) < 2:
        return None

    idx = rng.choice(len(config.holes), size=2, replace=False)
    h1 = config.holes[idx[0]]
    h2 = config.holes[idx[1]]
    dist = round(pairwise_dist(h1, h2), 1)

    pair_visible = False
    if annotations:
        show_spacing = annotations.get("show_spacing", set())
        pair_visible = (h1.id, h2.id) in show_spacing or (
            h2.id,
            h1.id,
        ) in show_spacing

    if annotation_level == "minimal" or (
        annotation_level == "partial" and not pair_visible
    ):
        approx_dist = round(dist)
        reasoning = (
            f"Using the scale bar to estimate positions: "
            f"{h1.id} and {h2.id} appear approximately "
            f"{approx_dist}mm apart center-to-center."
        )
    else:
        reasoning = (
            f"{h1.id} is at ({h1.cx}, {h1.cy}). "
            f"{h2.id} is at ({h2.cx}, {h2.cy}). "
            f"Distance = √(({h2.cx}-{h1.cx})² + ({h2.cy}-{h1.cy})²) "
            f"= {dist}mm."
        )

    return Question(
        qtype="measurement_hole_to_hole",
        question=f"What is the center-to-center distance between "
        f"{h1.id} and {h2.id}?",
        answer=f"{dist}mm",
        reasoning=reasoning,
    )


def gen_measurement_plate_dims(
    config: PlateConfig,
    annotation_level: str = "full",
    annotations: dict | None = None,
) -> Question:
    """Generate a plate dimensions measurement question.

    Args:
        config: The plate configuration.
        annotation_level: One of 'full', 'partial', 'minimal'.
        annotations: Annotation visibility dict.

    Returns:
        A Question asking for width × height.
    """
    pw, ph = config.plate_width, config.plate_height

    dims_visible = True
    if annotations:
        dims_visible = annotations.get("show_plate_dims", True)

    if annotation_level == "minimal" or (
        annotation_level == "partial" and not dims_visible
    ):
        approx_pw = round(pw)
        approx_ph = round(ph)
        reasoning = (
            f"Using the scale bar, the plate appears approximately "
            f"{approx_pw}×{approx_ph}mm."
        )
    else:
        reasoning = (
            f"The plate dimension annotations show width {pw}mm and height {ph}mm."
        )

    return Question(
        qtype="measurement_plate_dims",
        question="What are the plate dimensions (width × height)?",
        answer=f"{pw}×{ph}mm",
        reasoning=reasoning,
    )


def gen_all_measurements(
    config: PlateConfig,
    rng: np.random.Generator,
    annotation_level: str = "full",
    annotations: dict | None = None,
) -> list[Question]:
    """Generate all four measurement extraction questions for an example.

    Args:
        config: The plate configuration.
        rng: NumPy random generator.
        annotation_level: One of 'full', 'partial', 'minimal'.
        annotations: Annotation visibility dict.

    Returns:
        List of 3–4 Question objects (hole-to-hole may be skipped).
    """
    qs: list[Question] = []
    qs.append(gen_measurement_diameter(config, rng, annotation_level, annotations))
    qs.append(gen_measurement_edge_distance(config, rng, annotation_level, annotations))
    q = gen_measurement_hole_to_hole(config, rng, annotation_level, annotations)
    if q:
        qs.append(q)
    qs.append(gen_measurement_plate_dims(config, annotation_level, annotations))
    return qs


def gen_rule_selection(
    config: PlateConfig, rng: np.random.Generator
) -> Optional[Question]:
    """Generate a rule selection question (conditional complexity only).

    Two variants: asks about the acceptable diameter range for a zone,
    or whether a spacing rule applies to a zone.

    Args:
        config: The plate configuration (must be conditional complexity).
        rng: NumPy random generator.

    Returns:
        A Question, or None if complexity is not conditional.
    """
    if config.rule_complexity != "conditional":
        return None

    zone_names = sorted(config.zones.keys())
    variant = rng.choice(["material_class", "zone_rule"])

    if variant == "material_class":
        zone = rng.choice(zone_names)
        tol_rule = next((r for r in config.rules if r.rule_type == "tolerance"), None)
        if not tol_rule:
            return None

        nom = tol_rule.params["nominal"][zone]
        tol = tol_rule.params["tolerance"][zone]
        lo, hi = round(nom - tol, 1), round(nom + tol, 1)

        return Question(
            qtype="rule_selection",
            question=f"What is the acceptable diameter range for Zone {zone} holes?",
            answer=f"{lo}–{hi}mm (nominal {nom:.1f} ± {tol:.1f}mm)",
            reasoning=(
                f"Zone {zone} holes have nominal diameter {nom:.1f}mm. "
                f"From the spec, the tolerance for this zone is "
                f"±{tol:.1f}mm. Acceptable range: "
                f"{nom:.1f} - {tol:.1f} = {lo}mm to "
                f"{nom:.1f} + {tol:.1f} = {hi}mm."
            ),
        )

    else:
        spacing_rule = next((r for r in config.rules if r.rule_type == "spacing"), None)
        if not spacing_rule:
            return None

        zone = rng.choice(zone_names)
        sp = spacing_rule.params["min_spacing"].get(zone, 0)
        applies = sp > 0

        return Question(
            qtype="rule_selection",
            question=f"Does Rule {spacing_rule.id} (spacing) apply to Zone {zone}?",
            answer="Yes" if applies else "No",
            reasoning=(
                f"Rule {spacing_rule.id} specifies minimum spacing of "
                f"{sp}mm for Zone {zone}. "
                + (
                    f"This is > 0, so the rule applies."
                    if applies
                    else f"The spec states this rule does not apply to Zone {zone}."
                )
            ),
        )


def gen_counterfactual(
    config: PlateConfig,
    rng: np.random.Generator,
    annotation_level: str = "full",
    annotations: dict | None = None,
) -> Optional[Question]:
    """Generate a counterfactual question about a violating hole.

    Picks a random violating hole and one of its violated rules, then
    asks what value would be needed for compliance. Adapts reasoning
    to annotation visibility.

    Args:
        config: The plate configuration.
        rng: NumPy random generator.
        annotation_level: One of 'full', 'partial', 'minimal'.
        annotations: Annotation visibility dict.

    Returns:
        A Question, or None if there are no violations.
    """
    pw, ph = config.plate_width, config.plate_height
    violating = [h for h in config.holes if h.intended_violations]
    if not violating:
        return None

    hole = rng.choice(violating)
    viol_ids = hole.intended_violations
    non_bolt = [
        vid
        for vid in viol_ids
        if next(r for r in config.rules if r.id == vid).rule_type != "bolt_populated"
    ]
    target_id = rng.choice(non_bolt) if non_bolt else viol_ids[0]
    rule = next(r for r in config.rules if r.id == target_id)

    dia_visible = True
    edge_visible = True
    if annotations:
        dia_visible = hole.id in annotations.get("show_diameters", set())
        edge_visible = hole.id in annotations.get("show_edge_distances", set())

    use_approx = annotation_level == "minimal" or (
        annotation_level == "partial" and not dia_visible
    )

    if rule.rule_type == "tolerance":
        nom = rule.params["nominal"][hole.zone]
        tol = rule.params["tolerance"][hole.zone]
        lo, hi = round(nom - tol, 1), round(nom + tol, 1)

        if use_approx:
            approx_dia = round(hole.diameter)
            dia_text = f"~{approx_dia}mm (estimated from scale bar)"
        else:
            dia_text = f"{hole.diameter}mm"

        if hole.diameter > hi:
            return Question(
                qtype="counterfactual",
                question=f"What is the maximum allowable diameter for "
                f"{hole.id} to comply with Rule {rule.id}?",
                answer=f"{hi}mm",
                reasoning=(
                    f"Rule {rule.id} specifies {nom:.1f} ± {tol:.1f}mm. "
                    f"Upper limit = {nom:.1f} + {tol:.1f} = {hi}mm. "
                    f"{hole.id} is currently {dia_text}, which "
                    f"exceeds this by {round(hole.diameter - hi, 1)}mm."
                ),
            )
        else:
            return Question(
                qtype="counterfactual",
                question=f"What is the minimum allowable diameter for "
                f"{hole.id} to comply with Rule {rule.id}?",
                answer=f"{lo}mm",
                reasoning=(
                    f"Rule {rule.id} specifies {nom:.1f} ± {tol:.1f}mm. "
                    f"Lower limit = {nom:.1f} - {tol:.1f} = {lo}mm. "
                    f"{hole.id} is currently {dia_text}, which "
                    f"is below this by {round(lo - hole.diameter, 1)}mm."
                ),
            )

    elif rule.rule_type == "edge_distance":
        mult = rule.params["multiplier"][hole.zone]
        use_nominal = rule.params.get("use_nominal", False)
        if use_nominal:
            ref_dia = rule.params["nominal_ref"][hole.zone]
        else:
            ref_dia = hole.diameter
        required = round(mult * ref_dia, 2)
        actual = round(min_edge_distance(hole.cx, hole.cy, pw, ph), 1)

        use_approx_edge = annotation_level == "minimal" or (
            annotation_level == "partial" and not edge_visible
        )

        if use_approx_edge:
            actual_text = f"~{round(actual)}mm (estimated from scale bar)"
        else:
            actual_text = f"{actual}mm"

        if use_nominal:
            req_text = (
                f"Rule {rule.id} requires ≥ {mult:.1f}× nominal from Rule R1. "
                f"Nominal = {ref_dia:.1f}mm, so required = "
                f"{mult:.1f} × {ref_dia:.1f} = {required}mm."
            )
        else:
            req_text = (
                f"Rule {rule.id} requires ≥ {mult:.1f} × diameter. "
                f"{hole.id} diameter is {hole.diameter}mm, so required "
                f"edge distance = {mult:.1f} × {hole.diameter} = {required}mm."
            )

        return Question(
            qtype="counterfactual",
            question=f"What is the minimum edge distance {hole.id} would "
            f"need to comply with Rule {rule.id}?",
            answer=f"{required}mm",
            reasoning=(
                f"{req_text} Currently {actual_text}, which is "
                f"{round(required - actual, 1)}mm short."
            ),
        )

    elif rule.rule_type == "spacing":
        min_req = rule.params["min_spacing"].get(hole.zone, 0)

        closest_dist = float("inf")
        closest_id: str | None = None
        for other in config.holes:
            if other.id == hole.id or other.zone != hole.zone:
                continue
            d = pairwise_dist(hole, other)
            if d < closest_dist:
                closest_dist = d
                closest_id = other.id

        if closest_id is None:
            return None

        closest_dist = round(closest_dist, 1)

        if annotation_level == "minimal":
            dist_text = f"~{round(closest_dist)}mm (estimated from scale bar)"
        else:
            dist_text = f"{closest_dist}mm"

        return Question(
            qtype="counterfactual",
            question=f"By how much would the spacing between {hole.id} and "
            f"{closest_id} need to increase to comply with "
            f"Rule {rule.id}?",
            answer=f"{round(min_req - closest_dist, 1)}mm",
            reasoning=(
                f"Rule {rule.id} requires ≥ {min_req}mm spacing. "
                f"Current spacing between {hole.id} and {closest_id} "
                f"is {dist_text}. Need to increase by "
                f"{min_req} - {closest_dist} = "
                f"{round(min_req - closest_dist, 1)}mm."
            ),
        )

    elif rule.rule_type == "bolt_populated":
        other_rules = [r for r in config.rules if r.id != rule.id]
        other_failures: list[str] = []
        for r in other_rules:
            if not check_passes(hole, r, config.holes, pw, ph):
                other_failures.append(r.id)

        if other_failures:
            answer = (
                f"No. Installing a bolt would fix Rule {rule.id}, "
                f"but {hole.id} also violates {', '.join(other_failures)}."
            )
        else:
            answer = (
                f"Yes. {hole.id}'s only violation is the missing bolt. "
                f"Installing one would make it fully compliant."
            )

        return Question(
            qtype="counterfactual",
            question=f"If a fastener were installed in {hole.id}, would it "
            f"be fully compliant with all rules?",
            answer=answer,
            reasoning=(
                f"{hole.id} currently violates Rule {rule.id} "
                f"(no bolt). Checking other rules: "
                + (
                    f"also fails {', '.join(other_failures)}."
                    if other_failures
                    else "passes all other rules."
                )
            ),
        )

    return None


def generate_questions(
    config: PlateConfig,
    seed: int | None = None,
    annotations: dict | None = None,
) -> list[dict]:
    """Generate all questions for a plate configuration.

    Produces per-component compliance questions, a full audit, measurement
    extraction questions, and optionally rule selection and counterfactual
    questions depending on complexity and violations.

    Args:
        config: PlateConfig from sampler.
        seed: Random seed for reproducible question generation.
        annotations: Annotation visibility dict from decide_annotations.
            If None, reasoning assumes full annotation.

    Returns:
        List of question dicts with type, question, answer, reasoning.
    """
    rng = np.random.default_rng(seed)
    annotation_level = config.annotation_level
    questions: list[Question | dict] = []

    questions.extend(gen_per_component(config, rng, annotation_level, annotations))
    questions.append(gen_full_audit(config, annotation_level, annotations))
    questions.extend(gen_all_measurements(config, rng, annotation_level, annotations))

    if config.rule_complexity == "conditional":
        q = gen_rule_selection(config, rng)
        if q:
            questions.append(q)

    if any(h.intended_violations for h in config.holes):
        q = gen_counterfactual(config, rng, annotation_level, annotations)
        if q:
            questions.append(q)

    return [q.to_dict() if isinstance(q, Question) else q for q in questions]


if __name__ == "__main__":
    from sampler import sample_plate_with_retry
    from spec_generator import generate_spec

    for seed, comp, nv in [
        (0, "simple", 1),
        (10, "multi_rule", 2),
        (20, "conditional", 2),
    ]:
        print("=" * 65)
        print(f"EXAMPLE: {comp}, {nv} violations, seed={seed}")
        print("=" * 65)

        cfg = sample_plate_with_retry(
            num_violations=nv, rule_complexity=comp, seed=seed
        )
        if not cfg:
            print("FAILED to generate config\n")
            continue

        spec = generate_spec(cfg, seed=seed)
        print(f"Plate: {cfg.plate_width}×{cfg.plate_height}mm")
        print(
            f"Holes: {[(h.id, h.zone, h.diameter, h.intended_violations) for h in cfg.holes]}"
        )
        print(f"\nSpec:\n{spec}\n")

        qs = generate_questions(cfg, seed=seed)
        print(f"Questions ({len(qs)}):")
        for i, q in enumerate(qs):
            print(f"\n  Q{i + 1} [{q['type']}]")
            print(f"  {q['question']}")
            ans = q["answer"]
            if isinstance(ans, list):
                print("  Answer:")
                for a in ans:
                    print(f"    - {a}")
            else:
                print(f"  Answer: {ans}")
            print(f"  Reasoning: {q['reasoning']}")
        print()
