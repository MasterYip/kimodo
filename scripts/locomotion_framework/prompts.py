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
    """Map planar velocity magnitude to a speed descriptor.

    Uses planar speed ``hypot(vx, vy)`` (not just ``|vx|``) so that lateral
    and backward motions get the correct pace hint too.  For forward-dominant
    motions this is identical to the old ``|vx|`` behaviour; for pure-lateral
    motions it fixes the wrong "very slowly" hint that the vx-only metric gave.
    """
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


def resolve_arm_swing(mode: str, vel: Optional[dict[str, float]] = None) -> str:
    """Resolve prompt-only arm-swing intent from planar travel direction."""
    if mode not in {"auto", "sagittal", "lateral", "none"}:
        raise ValueError("arm_swing must be auto/sagittal/lateral/none")
    if mode == "none":
        return ""
    if mode == "auto":
        vx = 0.0 if vel is None else vel.get("vx", 0.0)
        vy = 0.0 if vel is None else vel.get("vy", 0.0)
        if abs(vx) < 1e-8 and abs(vy) < 1e-8:
            return ""
        mode = "sagittal" if abs(vx) >= abs(vy) else "lateral"
    if mode == "sagittal":
        return "with natural alternating forward and backward arm swing"
    return "with natural alternating left and right arm swing and slight fore-aft clearance"


def _exact_prompt(text: str) -> str:
    """Normalize only terminal punctuation for a literal prompt override."""
    result = text.strip()
    if not result.endswith((".", "!", "?")):
        result += "."
    return result


def build_motion_prompt(
    description: str,
    style: str = "",
    torso_height: Optional[float] = None,
    vel: Optional[dict[str, float]] = None,
    speed_hint: bool = True,
    exact_prompt: Optional[str] = None,
    arm_swing: str = "none",
) -> str:
    """Compose a text prompt for Kimodo from sampled parameters.

    Args:
        description: Base motion description (e.g. "A person walks forward").
        style: Style modifier (e.g. "casually"); empty string omits it.
        torso_height: Normalized height; maps to a crouch/tall hint.
        vel: Velocity command dict (vx/vy/wz); used only for the speed hint.
        speed_hint: When False, suppress the velocity-magnitude hint
            ("slowly"/"at a brisk pace"/...). Used for exact-prompt parity
            batches where ``description`` is already the final prompt.
        exact_prompt: Literal final prompt; when set it bypasses composition
            (only terminal punctuation is normalised).
        arm_swing: Text intent for arm swing ("none"/"auto"/"sagittal"/"lateral").
    """
    if exact_prompt is not None:
        return _exact_prompt(exact_prompt)
    prefix = ""
    suffix_parts = []

    # Add speed hint
    if speed_hint:
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

    # Text intent only; this does not create a physical arm constraint.
    arm_hint = resolve_arm_swing(arm_swing, vel)
    if arm_hint:
        suffix_parts.append(arm_hint)

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
