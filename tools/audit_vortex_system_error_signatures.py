from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "case_id",
    "family",
    "value",
    "nominal_value",
    "grid_n",
    "output_power_ratio_to_nominal",
    "xy_corr_nominal",
    "xz_corr_nominal",
    "yz_corr_nominal",
    "xz_center_slope_rad_approx",
    "yz_center_slope_rad_approx",
    "xz_active_length_m",
    "yz_active_length_m",
    "axial_peak_residual_rms",
    "exact_axicon_kr_m_inv",
    "min_rotated_plane_spectral_power_ratio",
}


def _case_rows(df: pd.DataFrame, family: str, case_id: str) -> pd.DataFrame:
    return df[(df["family"] == family) & (df["case_id"] == case_id)].copy()


def _nominal_rows(df: pd.DataFrame) -> pd.DataFrame:
    values = pd.to_numeric(df["value"], errors="coerce")
    nominal = pd.to_numeric(df["nominal_value"], errors="coerce")
    same_finite = np.isclose(values, nominal, rtol=0.0, atol=1e-15, equal_nan=False)
    same_inf = np.isinf(values) & np.isinf(nominal) & (np.sign(values) == np.sign(nominal))
    return df[same_finite | same_inf].copy()


def _endpoint_abs_metric(
    df: pd.DataFrame,
    family: str,
    case_id: str,
    metric: str,
) -> float | None:
    sub = _case_rows(df, family, case_id)
    if sub.empty or metric not in sub:
        return None
    nominal = pd.to_numeric(sub["nominal_value"], errors="coerce").iloc[0]
    values = pd.to_numeric(sub["value"], errors="coerce")
    if np.isfinite(nominal):
        idx = np.argmax(np.abs(values.to_numpy() - float(nominal)))
    else:
        finite = np.isfinite(values.to_numpy())
        if not np.any(finite):
            return None
        idx = np.where(finite)[0][np.argmax(np.abs(values.to_numpy()[finite]))]
    value = float(pd.to_numeric(sub.iloc[idx][metric], errors="coerce"))
    return value if np.isfinite(value) else None


def audit(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    hard_failures: list[str] = []
    warnings: list[str] = []

    if missing:
        hard_failures.append(f"missing required columns: {missing}")
        return {
            "outcome": "VORTEX-SYSTEM-ERROR-SIGNATURE-AUDIT",
            "hard_pass": False,
            "report_figures_authorised": False,
            "hard_failures": hard_failures,
            "warnings": warnings,
        }

    nominal = _nominal_rows(df)
    expected_nominal_count = df[["case_id", "family"]].drop_duplicates().shape[0]
    if len(nominal) != expected_nominal_count:
        hard_failures.append(
            f"expected {expected_nominal_count} nominal rows, found {len(nominal)}"
        )

    for metric in ("xy_corr_nominal", "xz_corr_nominal", "yz_corr_nominal"):
        bad = nominal[pd.to_numeric(nominal[metric], errors="coerce") < 0.999999]
        if not bad.empty:
            hard_failures.append(
                f"nominal self-correlation failed for {metric}: {len(bad)} rows"
            )

    nominal_power = pd.to_numeric(
        nominal["output_power_ratio_to_nominal"], errors="coerce"
    ).to_numpy()
    if nominal_power.size and np.nanmax(np.abs(nominal_power - 1.0)) > 1e-10:
        hard_failures.append("nominal output power ratio is not unity")

    # Rigid coordinate rotations are numerical coordinate transforms, not
    # absorbers. The scalar tilted-plane implementation must retain spectral
    # power to within 1.5% per transform at the declared <=0.5 degree sweeps.
    tilt_mask = df["family"].str.contains("tilt", case=False, na=False)
    tilt = df[tilt_mask].copy()
    if not tilt.empty:
        ratios = pd.to_numeric(
            tilt["min_rotated_plane_spectral_power_ratio"], errors="coerce"
        )
        minimum = float(np.nanmin(ratios))
        if minimum < 0.985:
            hard_failures.append(
                "rotated-plane numerical power gate failed: "
                f"minimum transform ratio={minimum:.6f} < 0.985"
            )
    else:
        minimum = 1.0

    # Exact-Snell parameter families must alter kr monotonically.
    monotonic_results: dict[str, bool] = {}
    for family in ("axicon_base_angle_scale", "axicon_index_scale"):
        for case_id in sorted(df["case_id"].unique()):
            sub = _case_rows(df, family, case_id)
            if sub.empty:
                continue
            sub = sub.sort_values("value")
            kr = pd.to_numeric(sub["exact_axicon_kr_m_inv"], errors="coerce").to_numpy()
            passed = bool(np.all(np.diff(kr) > 0.0))
            monotonic_results[f"{case_id}:{family}"] = passed
            if not passed:
                hard_failures.append(
                    f"exact axicon kr is not monotonic for {case_id}/{family}"
                )

    # Soft, physics-facing diagnostics. These do not authorise report claims;
    # they make the next human/literature review explicit.
    decentre_summary: dict[str, Any] = {}
    for family, axis_metric in (
        ("beam_lateral_decentre_x", "xz_center_slope_rad_approx"),
        ("beam_lateral_decentre_y", "yz_center_slope_rad_approx"),
        ("axicon_lateral_decentre_x", "xz_center_slope_rad_approx"),
        ("axicon_lateral_decentre_y", "yz_center_slope_rad_approx"),
    ):
        if family not in set(df["family"]):
            continue
        decentre_summary[family] = {
            case: _endpoint_abs_metric(df, family, case, axis_metric)
            for case in sorted(df["case_id"].unique())
        }

    round_tip_summary: dict[str, Any] = {}
    if "axicon_round_tip" in set(df["family"]):
        for case in sorted(df["case_id"].unique()):
            round_tip_summary[case] = _endpoint_abs_metric(
                df,
                "axicon_round_tip",
                case,
                "axial_peak_residual_rms",
            )

    radius_length_summary: dict[str, Any] = {}
    if "beam_radius_scale" in set(df["family"]):
        for case in sorted(df["case_id"].unique()):
            sub = _case_rows(df, "beam_radius_scale", case).sort_values("value")
            radius_length_summary[case] = [
                {
                    "radius_scale": float(row["value"]),
                    "xz_active_length_m": float(row["xz_active_length_m"]),
                }
                for _, row in sub.iterrows()
            ]

    if any(df["family"].str.startswith("axicon_rigid_tilt")):
        warnings.append(
            "axicon rigid tilt passes only a scalar rotated-plane numerical "
            "model; full refractive/vector surface validation remains blocked"
        )

    return {
        "outcome": "VORTEX-SYSTEM-ERROR-SIGNATURE-AUDIT",
        "hard_pass": not hard_failures,
        "report_figures_authorised": False,
        "row_count": int(len(df)),
        "family_count": int(df["family"].nunique()),
        "cases": sorted(str(v) for v in df["case_id"].unique()),
        "minimum_rotated_plane_spectral_power_ratio": minimum,
        "hard_failures": hard_failures,
        "warnings": warnings,
        "exact_kr_monotonic_checks": monotonic_results,
        "decentre_endpoint_line_slopes": decentre_summary,
        "round_tip_endpoint_axial_residual_rms": round_tip_summary,
        "beam_radius_active_length_summary": radius_length_summary,
        "policy": (
            "hard_pass means numerical/invariant gates passed. It does not mean "
            "all error families are physically validated or report-ready."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit numerical and signature metrics from vortex system-error sweeps."
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("outputs/validation/vortex_system_errors/system_error_metrics.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/vortex_system_errors/system_error_signature_audit.json"),
    )
    args = parser.parse_args()

    result = audit(args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["hard_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
