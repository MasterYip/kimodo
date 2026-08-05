"""Compose text prompts for G1 locomotion motion generation."""

from typing import Optional


def _torso_height_hint(height: float) -> str:
    """Map a normalized torso height to a qualitative descriptor.

    Normalized heights are relative to G1's default standing height.
    """
    if height < 0.50:
        return "crouching very low"
    elif height < 0.60:
        return "crouching"
    elif height < 0.70:
        return "slightly crouching"
    elif height < 0.85:
        return ""  # normal stance, no hint needed
    elif height < 0.95:
        return "standing tall"
    else:
        return "standing on tiptoes"


def _speed_hint(vel: Optional[dict[str, float]] = None) -> str:
    """Map velocity magnitude to a speed descriptor."""
    if vel is None:
        return ""
    vx = vel.get("vx", 0.0)
    vy = vel.get("vy", 0.0)
    speed = (vx ** 2 + vy ** 2) ** 0.5
    if speed < 0.05:
        return "very slowly"
    elif speed < 0.3:
        return "slowly"
    elif speed < 0.7:
        return ""
    elif speed < 1.5:
        return "at a brisk pace"
    else:
        return "at high speed"


def build_motion_prompt(
    description: str,
    style: str = "",
    torso_height: Optional[float] = None,
    vel: Optional[dict[str, float]] = None,
) -> str:
    """Compose a text prompt for Kimodo from sampled parameters."""
    prefix = ""
    suffix_parts = []

    # Add speed hint
    speed = _speed_hint(vel)
    if speed:
        suffix_parts.append(speed)

    # Add style
    if style.strip():
        suffix_parts.append(style.strip())

    # Add torso height hint — only as a suffix modifier, not a prefix
    # (avoids doubling "a robot with crouching a robot crouches...")
    if torso_height is not None:
        hint = _torso_height_hint(torso_height)
        if hint:
            suffix_parts.append(hint)

    base = description.strip()
    suffix = " ".join(suffix_parts)
    if suffix:
        result = f"{base} {suffix}."
    else:
        result = f"{base}."

    # Clean up multiple periods
    while ".." in result:
        result = result.replace("..", ".")
    return result


def build_multi_prompt(
    samples: list,
    separator: str = ".",
) -> tuple[str, str]:
    """Build a multi-prompt timeline from multiple sampled motions.

    Args:
        samples: List of objects with .prompt and .duration attributes
        separator: Separator between prompts

    Returns:
        (prompts_string, durations_string) for kimodo_gen CLI.
    """
    prompts = [s.prompt.rstrip(".") for s in samples]
    durations = [str(s.duration) for s in samples]
    return (
        f" {separator} ".join(prompts) + separator,
        " ".join(durations),
    )
