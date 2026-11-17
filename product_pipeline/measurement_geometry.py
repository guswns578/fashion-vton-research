"""Convert metric body and garment measurements into pilot geometry parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Mapping

DEFAULT_BODY_CM = {
    "height": 172.0,
    "neck": 37.0,
    "shoulder_breadth": 42.0,
    "chest": 92.0,
    "waist": 76.0,
    "hip": 94.0,
    "arm_length": 60.0,
    "upper_arm": 29.0,
    "wrist": 16.0,
    "back_waist_length": 44.0,
}

DEFAULT_GARMENT_CM = {
    "shoulder": 48.0,
    "chest_flat": 55.0,
    "body_length": 70.0,
    "sleeve_length": 60.0,
    "upper_arm_flat": 22.0,
    "cuff_flat": 12.0,
    "hem_flat": 55.0,
    "armhole": 25.0,
}


def _positive_measurements(
    values: Mapping[str, object], required: tuple[str, ...], label: str
) -> dict[str, float]:
    result: dict[str, float] = {}
    missing = []
    for name in required:
        value = values.get(name)
        if value is None:
            missing.append(name)
            continue
        number = float(value)
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{label}.{name} must be a positive finite number, got {value!r}")
        result[name] = number
    if missing:
        raise ValueError(f"Missing {label} measurements: {', '.join(missing)}")
    return result


def ellipse_radius_x(circumference_m: float, depth_to_width: float = 0.70) -> float:
    """Solve the semi-major radius for an ellipse with a fixed b/a ratio."""
    if circumference_m <= 0 or not 0 < depth_to_width <= 1:
        raise ValueError("Circumference and depth-to-width ratio must be positive.")
    ratio = depth_to_width
    coefficient = math.pi * (3 * (1 + ratio) - math.sqrt((3 + ratio) * (1 + 3 * ratio)))
    return circumference_m / coefficient


def vertex_mass_from_gsm(
    surface_area_m2: float, vertex_count: int, mass_gsm: float
) -> tuple[float, float]:
    """Return (mass per vertex, total mass) for Blender's vertex-mass setting."""
    if surface_area_m2 <= 0 or vertex_count <= 0 or mass_gsm <= 0:
        raise ValueError("Surface area, vertex count, and GSM must be positive.")
    total_mass_kg = surface_area_m2 * mass_gsm / 1000.0
    return total_mass_kg / vertex_count, total_mass_kg


def resolve_mass_gsm(
    material: Mapping[str, object] | None,
    surface_area_m2: float,
    fallback_gsm: float,
) -> tuple[float, str]:
    material = material or {}
    measured_gsm = material.get("mass_gsm")
    if measured_gsm is not None:
        value = float(measured_gsm)
        if value <= 0:
            raise ValueError("material.mass_gsm must be positive.")
        return value, "pilot.material.mass_gsm"
    garment_mass_g = material.get("garment_mass_g")
    if garment_mass_g is not None:
        value = float(garment_mass_g)
        if value <= 0 or surface_area_m2 <= 0:
            raise ValueError("Garment mass and surface area must be positive.")
        return value / surface_area_m2, "pilot.material.garment_mass_g/mesh_area"
    if fallback_gsm <= 0:
        raise ValueError("Fallback GSM must be positive.")
    return fallback_gsm, "preset_fallback"


def load_pilot_material(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    pilot = json.loads(path.read_text(encoding="utf-8"))
    return dict(pilot.get("garment", {}).get("material", {}))


@dataclass(frozen=True)
class BodyProxy:
    source_cm: dict[str, float]
    torso_radius_x_m: float
    torso_radius_y_m: float
    torso_half_height_m: float
    torso_center_z_m: float
    shoulder_z_m: float
    arm_half_length_m: float
    arm_radius_m: float
    arm_center_x_m: float
    arm_center_z_m: float
    arm_angle_deg: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PanelPattern:
    source_cm: dict[str, float]
    outline_xz_m: tuple[tuple[float, float], ...]
    shoulder_half_m: float
    chest_half_m: float
    hem_half_m: float
    sleeve_end_x_m: float
    shoulder_z_m: float
    hem_z_m: float
    neck_half_m: float
    neck_line_z_m: float
    cuff_half_height_m: float
    panel_offset_y_m: float
    inferred: dict[str, float]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["outline_xz_m"] = [list(point) for point in self.outline_xz_m]
        return data


def body_proxy_from_cm(values: Mapping[str, object]) -> BodyProxy:
    required = tuple(DEFAULT_BODY_CM)
    cm = _positive_measurements(values, required, "subject.measurements_cm")
    chest_m = cm["chest"] / 100.0
    torso_radius_x = ellipse_radius_x(chest_m)
    torso_radius_y = torso_radius_x * 0.70
    torso_height = (cm["back_waist_length"] + 14.0) / 100.0
    torso_half_height = torso_height * 0.5
    shoulder_z = torso_height * 0.43
    torso_center_z = shoulder_z - torso_half_height * 0.78
    arm_length = cm["arm_length"] / 100.0
    arm_half_length = arm_length * 0.5
    arm_angle_deg = 8.0
    arm_drop = math.sin(math.radians(arm_angle_deg)) * arm_half_length
    arm_center_x = (
        cm["shoulder_breadth"] / 200.0 + math.cos(math.radians(arm_angle_deg)) * arm_half_length
    )
    arm_radius = cm["upper_arm"] / 100.0 / (2 * math.pi)
    return BodyProxy(
        source_cm=cm,
        torso_radius_x_m=torso_radius_x,
        torso_radius_y_m=torso_radius_y,
        torso_half_height_m=torso_half_height,
        torso_center_z_m=torso_center_z,
        shoulder_z_m=shoulder_z,
        arm_half_length_m=arm_half_length,
        arm_radius_m=arm_radius,
        arm_center_x_m=arm_center_x,
        arm_center_z_m=shoulder_z - arm_drop,
        arm_angle_deg=arm_angle_deg,
    )


def panel_pattern_from_cm(values: Mapping[str, object], body: BodyProxy) -> PanelPattern:
    required = tuple(DEFAULT_GARMENT_CM)
    cm = _positive_measurements(values, required, "garment.measurements_cm")
    shoulder = cm["shoulder"] / 100.0
    chest_flat = cm["chest_flat"] / 100.0
    body_length = cm["body_length"] / 100.0
    sleeve_length = cm["sleeve_length"] / 100.0
    upper_arm_flat = cm["upper_arm_flat"] / 100.0
    cuff_flat = cm["cuff_flat"] / 100.0
    hem_flat = cm["hem_flat"] / 100.0
    armhole = cm["armhole"] / 100.0

    if chest_flat * 2 <= body.source_cm["chest"] / 100.0:
        raise ValueError(
            "Garment circumference must exceed the body chest circumference for this loose-top pilot."
        )
    if upper_arm_flat * 2 <= body.source_cm["upper_arm"] / 100.0:
        raise ValueError(
            "Garment upper-arm circumference must exceed the body upper-arm circumference."
        )

    shoulder_half = shoulder * 0.5
    chest_half = chest_flat * 0.5
    hem_half = hem_flat * 0.5
    shoulder_z = body_length * 0.5
    hem_z = -body_length * 0.5
    shoulder_drop = min(max(shoulder * 0.055, 0.018), 0.040)
    neck_half = min(max(shoulder * 0.21, 0.075), shoulder_half * 0.58)
    neck_depth = min(max(armhole * 0.30, 0.060), 0.110)
    underarm_drop = min(max(armhole * 0.66, upper_arm_flat * 0.70), body_length * 0.42)
    sleeve_drop = min(max(sleeve_length * 0.16, 0.055), sleeve_length * 0.28)
    sleeve_run = math.sqrt(max(sleeve_length**2 - sleeve_drop**2, 0.0))
    sleeve_end_x = shoulder_half + sleeve_run
    cuff_center_z = shoulder_z - shoulder_drop - sleeve_drop
    cuff_half_height = cuff_flat * 0.5
    underarm_z = shoulder_z - shoulder_drop - underarm_drop
    panel_offset = body.torso_radius_y_m + 0.035

    left = [
        (-neck_half, shoulder_z),
        (-shoulder_half, shoulder_z - shoulder_drop),
        (-sleeve_end_x, cuff_center_z + cuff_half_height),
        (-sleeve_end_x, cuff_center_z - cuff_half_height),
        (-chest_half, underarm_z),
        (-hem_half, hem_z),
    ]
    right = [(-x, z) for x, z in reversed(left)]
    outline = tuple(left + right + [(0.0, shoulder_z - neck_depth)])
    return PanelPattern(
        source_cm=cm,
        outline_xz_m=outline,
        shoulder_half_m=shoulder_half,
        chest_half_m=chest_half,
        hem_half_m=hem_half,
        sleeve_end_x_m=sleeve_end_x,
        shoulder_z_m=shoulder_z,
        hem_z_m=hem_z,
        neck_half_m=neck_half,
        neck_line_z_m=shoulder_z - neck_depth,
        cuff_half_height_m=cuff_half_height,
        panel_offset_y_m=panel_offset,
        inferred={
            "shoulder_drop_m": shoulder_drop,
            "neck_depth_m": neck_depth,
            "underarm_drop_m": underarm_drop,
            "sleeve_drop_m": sleeve_drop,
        },
    )


def load_pilot_geometry(path: Path | None) -> tuple[BodyProxy, PanelPattern, str]:
    if path is None:
        body_values = DEFAULT_BODY_CM
        garment_values = DEFAULT_GARMENT_CM
        source = "synthetic_defaults"
    else:
        pilot = json.loads(path.read_text(encoding="utf-8"))
        body_values = pilot["subject"]["measurements_cm"]
        garment_values = pilot["garment"]["measurements_cm"]
        source = str(path.resolve())
    body = body_proxy_from_cm(body_values)
    return body, panel_pattern_from_cm(garment_values, body), source
