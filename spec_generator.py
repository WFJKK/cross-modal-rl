"""
Spec generator for mechanical plate compliance dataset.

Takes a PlateConfig from the sampler and produces a readable
specification document string. Same underlying rules, different
presentation complexity.

Presentation formats by rule_complexity:
- simple: direct statements, no tables
- multi_rule: direct statements, more rules
- conditional: tables, material/class mapping, zone-dependent rules
"""

import numpy as np
from sampler import PlateConfig, Rule
from typing import List, Optional


# ============================================================
# Fixed material/class mapping
# ============================================================

MATERIALS = [
    {'name': 'Steel AISI 1018', 'class': 'I'},
    {'name': 'Aluminum 6061-T6', 'class': 'II'},
    {'name': 'Composite CFRP', 'class': 'III'},
]

# Class → tolerance values for table presentation
# These are just the COLUMN HEADERS; actual values come from the sampler
CLASS_ORDER = ['I', 'II', 'III']


# ============================================================
# Spec ID / title generation
# ============================================================

PART_PREFIXES = ['MP', 'FP', 'BP', 'SP', 'GP']  # Mounting, Flange, Bracket, Support, Guide Plate
PART_NAMES = {
    'MP': 'Mounting Plate',
    'FP': 'Flange Plate',
    'BP': 'Bracket Plate',
    'SP': 'Support Plate',
    'GP': 'Guide Plate',
}


def generate_spec_id(rng: np.random.Generator) -> tuple[str, str]:
    """Generate a spec ID and part name."""
    prefix = rng.choice(PART_PREFIXES)
    number = rng.integers(100, 999)
    spec_id = f'{prefix}-{number}'
    part_name = f'{PART_NAMES[prefix]} {spec_id}'
    return spec_id, part_name


# ============================================================
# Simple / multi_rule spec generation
# ============================================================

def generate_simple_spec(config: PlateConfig, rng) -> str:
    """Direct statements, no tables, no conditionals."""
    spec_id, part_name = generate_spec_id(rng)

    lines = []
    lines.append(f'SPEC-{spec_id}: {part_name} Design Requirements')
    lines.append('')

    # Shuffle rule order in document, but keep original IDs
    shuffled_rules = list(config.rules)
    rng.shuffle(shuffled_rules)

    for rule in shuffled_rules:

        if rule.rule_type == 'tolerance':
            nom = list(rule.params['nominal'].values())[0]
            tol = list(rule.params['tolerance'].values())[0]
            lines.append(f'Rule {rule.id}: All holes shall have diameter {nom:.1f} ± {tol:.1f} mm.')

        elif rule.rule_type == 'edge_distance':
            mult = list(rule.params['multiplier'].values())[0]
            if rule.params.get('use_nominal'):
                # 2-hop: reference the tolerance rule's nominal
                tol_rule = next((r for r in config.rules if r.rule_type == 'tolerance'), None)
                if tol_rule:
                    tol_id = tol_rule.id
                    lines.append(f'Rule {rule.id}: Minimum edge distance for any hole '
                                 f'shall be no less than {mult:.1f}× the nominal diameter '
                                 f'specified in Rule {tol_id}.')
                else:
                    lines.append(f'Rule {rule.id}: Minimum edge distance for any hole '
                                 f'shall be no less than {mult:.1f}× the hole diameter.')
            else:
                lines.append(f'Rule {rule.id}: Minimum edge distance for any hole '
                             f'shall be no less than {mult:.1f}× the hole diameter.')

        elif rule.rule_type == 'spacing':
            sp = list(rule.params['min_spacing'].values())[0]
            lines.append(f'Rule {rule.id}: Minimum center-to-center hole spacing '
                         f'shall be {sp:.1f} mm.')

        elif rule.rule_type == 'bolt_populated':
            lines.append(f'Rule {rule.id}: All bolt holes must be populated '
                         f'with fasteners. Empty holes are not permitted.')

        lines.append('')

    return '\n'.join(lines).strip()


# ============================================================
# Conditional spec generation (with tables and indirection)
# ============================================================

def generate_conditional_spec(config: PlateConfig, rng) -> str:
    """
    Zone-dependent rules with table lookups and material/class mapping.
    Introduces multi-hop reasoning chains.
    """
    spec_id, part_name = generate_spec_id(rng)

    # Pick a material (determines class)
    material = rng.choice(MATERIALS)
    mat_name = material['name']
    mat_class = material['class']

    zone_names = sorted(config.zones.keys())

    lines = []
    lines.append(f'SPEC-{spec_id}: {part_name} Design Requirements')
    lines.append(f'Material: {mat_name}')
    lines.append('')

    # Build each rule as an independent block of lines
    rule_blocks = []

    # --- Tolerance rule ---
    tol_rule = next((r for r in config.rules if r.rule_type == 'tolerance'), None)
    if tol_rule:
        block = []
        block.append(f'Rule {tol_rule.id}: Hole diameters shall conform to the tolerances '
                     f'specified in Table A based on Material Class.')
        for z in zone_names:
            nom = tol_rule.params['nominal'][z]
            block.append(f'  Nominal diameter for Zone {z} holes: {nom:.1f} mm.')
        rule_blocks.append(block)

    # --- Edge distance rule ---
    edge_rule = next((r for r in config.rules if r.rule_type == 'edge_distance'), None)
    if edge_rule:
        block = []
        block.append(f'Rule {edge_rule.id}: Edge distance requirements depend on hole zone:')
        for i, z in enumerate(zone_names):
            mult = edge_rule.params['multiplier'][z]
            letter = chr(ord('a') + i)
            block.append(f'  ({letter}) Zone {z}: edge distance ≥ {mult:.1f}× hole diameter')
        rule_blocks.append(block)

    # --- Spacing rule ---
    spacing_rule = next((r for r in config.rules if r.rule_type == 'spacing'), None)
    if spacing_rule:
        active_zones = [z for z in zone_names
                        if spacing_rule.params['min_spacing'].get(z, 0) > 0]
        inactive_zones = [z for z in zone_names if z not in active_zones]

        if active_zones:
            block = []
            sp = spacing_rule.params['min_spacing'][active_zones[0]]
            zone_str = ', '.join(f'Zone {z}' for z in active_zones)
            block.append(f'Rule {spacing_rule.id}: For Material Class {mat_class}, '
                         f'the minimum hole-to-hole spacing in {zone_str} '
                         f'shall be {sp:.1f} mm.')
            if inactive_zones:
                inactive_str = ', '.join(f'Zone {z}' for z in inactive_zones)
                block.append(f'  Rule {spacing_rule.id} does not apply to {inactive_str}.')
            rule_blocks.append(block)

    # --- Bolt populated rule ---
    bolt_rule = next((r for r in config.rules if r.rule_type == 'bolt_populated'), None)
    if bolt_rule:
        block = [f'Rule {bolt_rule.id}: All bolt holes must be populated with fasteners. '
                 f'Empty holes are not permitted in the final assembly.']
        rule_blocks.append(block)

    # Shuffle rule blocks
    rng.shuffle(rule_blocks)

    for block in rule_blocks:
        for line in block:
            lines.append(line)
        lines.append('')

    # --- Table A (always at bottom) ---
    if tol_rule:
        lines.append(build_tolerance_table(tol_rule, config.zones, mat_class))
        lines.append('')

    # --- Material class note (always at bottom) ---
    lines.append('Note: Material Class is determined by material type:')
    mat_parts = [f'  {m["name"].split()[0]} → Class {m["class"]}' for m in MATERIALS]
    lines.append('  |  '.join(mat_parts))
    lines.append('')

    return '\n'.join(lines).strip()


def build_tolerance_table(tol_rule: Rule, zones: dict, active_class: str) -> str:
    """
    Build Table A mapping Material Class to tolerances.

    The table has rows for each class (I, II, III) and columns for
    small holes (nom ≤ 10mm) and large holes (nom > 10mm).

    Only the active class row has the actual values from the sampler.
    Other rows get plausible but different values.
    """
    zone_names = sorted(zones.keys())

    # Determine which zones are "small" (nom ≤ 10) vs "large" (nom > 10)
    nominals = {z: tol_rule.params['nominal'][z] for z in zone_names}
    tolerances = {z: tol_rule.params['tolerance'][z] for z in zone_names}

    # The active class tolerance for small and large holes
    small_zones = [z for z in zone_names if nominals[z] <= 10.0]
    large_zones = [z for z in zone_names if nominals[z] > 10.0]

    # Get actual tolerance values for the active class
    if small_zones:
        active_small_tol = tolerances[small_zones[0]]
    else:
        active_small_tol = 0.5  # default

    if large_zones:
        active_large_tol = tolerances[large_zones[0]]
    else:
        active_large_tol = 0.8  # default

    # Build table with plausible values for other classes
    # Class I is tightest, Class III is loosest
    class_tolerances = {
        'I':   {'small': 0.3, 'large': 0.4},
        'II':  {'small': 0.5, 'large': 0.8},
        'III': {'small': 1.0, 'large': 1.5},
    }

    # Override the active class with actual sampler values
    class_tolerances[active_class]['small'] = active_small_tol
    class_tolerances[active_class]['large'] = active_large_tol

    lines = []
    lines.append('Table A: Diameter tolerances by Material Class')
    lines.append('| Material Class | Small holes (nom ≤ 10mm) | Large holes (nom > 10mm) |')
    lines.append('|----------------|--------------------------|--------------------------|')
    for cls in CLASS_ORDER:
        s = class_tolerances[cls]['small']
        l = class_tolerances[cls]['large']
        lines.append(f'| Class {cls:<8s} | ± {s:.1f} mm{" " * 17}| ± {l:.1f} mm{" " * 17}|')

    return '\n'.join(lines)


# ============================================================
# Main entry point
# ============================================================

def generate_spec(config: PlateConfig, seed: int | None = None) -> str:
    """
    Generate a specification document for a plate configuration.

    Args:
        config: PlateConfig from the sampler
        seed: random seed for presentation choices

    Returns:
        Spec document as a string
    """
    rng = np.random.default_rng(seed)

    if config.rule_complexity in ('simple', 'multi_rule'):
        return generate_simple_spec(config, rng)
    elif config.rule_complexity == 'conditional':
        return generate_conditional_spec(config, rng)
    else:
        return generate_simple_spec(config, rng)


# ============================================================
# Test
# ============================================================

if __name__ == '__main__':
    from sampler import sample_plate_with_retry

    print("=" * 60)
    print("EXAMPLE 1: simple")
    print("=" * 60)
    cfg = sample_plate_with_retry(num_violations=1, rule_complexity='simple', seed=0)
    if cfg:
        spec = generate_spec(cfg, seed=0)
        print(spec)
        print()
        print(f"[Holes: {len(cfg.holes)}, Violations: "
              f"{[(h.id, h.intended_violations) for h in cfg.holes if h.intended_violations]}]")

    print()
    print("=" * 60)
    print("EXAMPLE 2: multi_rule")
    print("=" * 60)
    cfg = sample_plate_with_retry(num_violations=2, rule_complexity='multi_rule', seed=10)
    if cfg:
        spec = generate_spec(cfg, seed=10)
        print(spec)
        print()
        print(f"[Holes: {len(cfg.holes)}, Violations: "
              f"{[(h.id, h.intended_violations) for h in cfg.holes if h.intended_violations]}]")

    print()
    print("=" * 60)
    print("EXAMPLE 3: conditional")
    print("=" * 60)
    cfg = sample_plate_with_retry(num_violations=2, rule_complexity='conditional', seed=20)
    if cfg:
        spec = generate_spec(cfg, seed=20)
        print(spec)
        print()
        print(f"[Holes: {len(cfg.holes)}, Violations: "
              f"{[(h.id, h.intended_violations) for h in cfg.holes if h.intended_violations]}]")
        print(f"[Zones: {list(cfg.zones.keys())}, Nominals: {cfg.nominal_diameters}]")
