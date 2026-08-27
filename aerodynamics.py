# aerodynamics.py
# derives jsbsim-ready aerodynamic coefficients and moments of inertia
# from the xtr-33 geometry/mass in config.py, using standard small-rc-
# aircraft estimation methods since no wind tunnel data exists for this
# airframe - every estimate here gets corrected against real flight
# test data once the plane is built and flown
# used by aircraft/generic_trainer.xml (next file)

import math
from dataclasses import dataclass

from config import config


AIR_DENSITY_SEA_LEVEL_KG_M3 = 1.225
GRAVITY_MPS2 = 9.81


@dataclass(frozen=True)
class AeroCoefficients:
    aspect_ratio: float
    oswald_efficiency: float
    induced_drag_factor_k: float
    cl_alpha_per_rad: float
    cd0_estimate: float
    cl_max_required: float
    wing_loading_kg_m2: float
    ixx_estimate_kg_m2: float
    iyy_estimate_kg_m2: float
    izz_estimate_kg_m2: float


def oswald_efficiency_estimate(aspect_ratio: float) -> float:
    # raymer's empirical fit for straight, untapered wings
    return 1.78 * (1 - 0.045 * aspect_ratio ** 0.68) - 0.64


def lift_curve_slope_finite_wing(aspect_ratio: float, oswald_e: float) -> float:
    # thin-airfoil 2d slope (2*pi/rad) corrected for a finite, low-ar wing
    cl_alpha_2d = 2 * math.pi
    return cl_alpha_2d / (1 + cl_alpha_2d / (math.pi * aspect_ratio * oswald_e))


def zero_lift_drag_estimate(aspect_ratio: float) -> float:
    # flat-profile 3d-printed airframe - no streamlined fuselage fairing,
    # exposed control horns and a blunt flat-plate wing push this well
    # above a clean sailplane's ~0.02, kept as a function so it is easy
    # to override with a real tuft-test or glide-ratio result later
    return 0.05


def required_cl_max(mass_kg: float, wing_area_m2: float, stall_speed_mps: float) -> float:
    # backs out the cl_max the wing needs to actually stall at the speed
    # config.py assumes - if this is implausible, revisit the stall
    # speed estimate, not just the wing
    weight_n = mass_kg * GRAVITY_MPS2
    dynamic_pressure = 0.5 * AIR_DENSITY_SEA_LEVEL_KG_M3 * stall_speed_mps ** 2
    return weight_n / (dynamic_pressure * wing_area_m2)


def estimate_moments_of_inertia(mass_kg: float, wingspan_m: float):
    # simplified small-rc-aircraft estimate, not a substitute for a real
    # bifilar pendulum measurement - treats span-wise mass as a rod for
    # roll, scales pitch/yaw off an assumed fuselage length fraction of
    # wingspan, typical for a profile 3d-aerobatic airframe
    k_roll = 0.40
    fuselage_length_m = 0.8 * wingspan_m
    ixx = k_roll * mass_kg * (wingspan_m / 2) ** 2
    iyy = k_roll * mass_kg * (fuselage_length_m / 2) ** 2
    izz = ixx + iyy
    return ixx, iyy, izz


def compute_aero_coefficients() -> AeroCoefficients:
    ac = config.aircraft
    ar = ac.aspect_ratio
    e = oswald_efficiency_estimate(ar)
    k = 1.0 / (math.pi * ar * e)
    cl_alpha = lift_curve_slope_finite_wing(ar, e)
    cd0 = zero_lift_drag_estimate(ar)
    cl_max = required_cl_max(ac.mass_kg, ac.wing_area_m2, ac.stall_speed_estimate_mps)
    wing_loading = ac.mass_kg / ac.wing_area_m2
    ixx, iyy, izz = estimate_moments_of_inertia(ac.mass_kg, ac.wingspan_m)

    return AeroCoefficients(
        aspect_ratio=ar,
        oswald_efficiency=e,
        induced_drag_factor_k=k,
        cl_alpha_per_rad=cl_alpha,
        cd0_estimate=cd0,
        cl_max_required=cl_max,
        wing_loading_kg_m2=wing_loading,
        ixx_estimate_kg_m2=ixx,
        iyy_estimate_kg_m2=iyy,
        izz_estimate_kg_m2=izz,
    )


def validate(coeffs: AeroCoefficients) -> list:
    warnings = []
    if not (0.8 <= coeffs.cl_max_required <= 1.8):
        warnings.append(
            f"cl_max_required={coeffs.cl_max_required:.2f} is outside the usual "
            f"0.8-1.8 range for small rc wings - revisit the stall speed estimate "
            f"or wing area in config.py"
        )
    if not (0.02 <= coeffs.induced_drag_factor_k <= 0.08):
        warnings.append(
            f"induced_drag_factor_k={coeffs.induced_drag_factor_k:.3f} is outside "
            f"the usual 0.02-0.08 range for small rc aircraft"
        )
    if not (1.5 <= coeffs.wing_loading_kg_m2 <= 6.0):
        warnings.append(
            f"wing_loading={coeffs.wing_loading_kg_m2:.1f} kg/m2 is outside the "
            f"usual 1.5-6.0 range for small 3d-aerobatic rc aircraft"
        )
    if coeffs.oswald_efficiency <= 0 or coeffs.oswald_efficiency >= 1:
        warnings.append(
            f"oswald_efficiency={coeffs.oswald_efficiency:.2f} is not physically "
            f"valid (must be between 0 and 1) - aspect ratio is too extreme for "
            f"this empirical formula"
        )
    return warnings


def print_jsbsim_snippet(coeffs: AeroCoefficients):
    print("values for aircraft/generic_trainer.xml:")
    print(f"  Ixx: {coeffs.ixx_estimate_kg_m2:.4f} kg-m2")
    print(f"  Iyy: {coeffs.iyy_estimate_kg_m2:.4f} kg-m2")
    print(f"  Izz: {coeffs.izz_estimate_kg_m2:.4f} kg-m2")
    print(f"  CLalpha: {coeffs.cl_alpha_per_rad:.3f} per rad")
    print(f"  CD0: {coeffs.cd0_estimate:.3f}")
    print(f"  induced drag factor k: {coeffs.induced_drag_factor_k:.4f}")
    print(f"  CLmax (design target): {coeffs.cl_max_required:.3f}")


if __name__ == "__main__":
    coeffs = compute_aero_coefficients()
    print_jsbsim_snippet(coeffs)
    print()
    warnings = validate(coeffs)
    if warnings:
        print("validation warnings:")
        for w in warnings:
            print(" -", w)
    else:
        print("all estimates within plausible range for a small 3d-aerobatic rc aircraft")