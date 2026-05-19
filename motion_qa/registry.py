# motion_qa/registry.py

from __future__ import annotations

from motion_qa import modules

MODULE_MAP: dict[str, callable] = {
    # Spatial / temporal
    "dominant_direction":        modules.dominant_direction,
    "global_displacement":       modules.global_displacement,
    "displacement_category":     modules.displacement_category,
    "clip_duration":             modules.clip_duration,
    "most_active_limb":          modules.most_active_limb,
    # Dance-specific
    "classify_dance_style":      modules.classify_dance_style,
    "detect_freeze":             modules.detect_freeze,
    "detect_jacking":            modules.detect_jacking,
    "compute_rhythm_regularity": modules.compute_rhythm_regularity,
}
