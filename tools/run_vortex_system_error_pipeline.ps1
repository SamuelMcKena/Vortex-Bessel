param(
    [int]$GridN = 1536,
    [double]$Zmm = 60.0,
    [switch]$SkipFullSuite
)

$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param([scriptblock]$Command, [string]$Label)
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

Invoke-Checked {
    python -m pytest `
      tests/test_phase2e_source_sampling_repair.py `
      tests/test_phase2e_production_repair.py `
      tests/test_vortex_error_reference_models.py `
      tests/test_vortex_round_tip_reference.py `
      tests/test_vortex_physical_errors.py `
      tests/test_vortex_visual_atlas.py `
      tests/test_vortex_system_errors.py `
      tests/test_vortex_explicit_system_extensions.py `
      -q
} "Focused physics tests"

Invoke-Checked {
    python -m compileall -q `
      vbb_study/digital_twin/vortex_beam_slm_errors.py `
      vbb_study/digital_twin/vortex_explicit_4f.py `
      vbb_study/digital_twin/vortex_rotated_plane.py `
      vbb_study/digital_twin/vortex_system_route.py `
      vbb_study/digital_twin/vortex_system_error_sweeps.py `
      vbb_study/digital_twin/vortex_wavefront_errors.py `
      tools/check_vortex_explicit_4f_parity.py `
      tools/run_vortex_system_error_suite.py `
      tools/audit_vortex_system_error_signatures.py `
      tools/run_vortex_declared_aberration_suite.py
} "Compile new system-error modules"

Invoke-Checked {
    python tools/check_vortex_error_reference_models.py
} "Analytic/reference error-model checks"

Invoke-Checked {
    python tools/check_vortex_explicit_4f_parity.py --grid-n 512 --cases B0 V1 V3
} "Fast explicit-4F parity gate"

Invoke-Checked {
    python tools/check_vortex_explicit_4f_parity.py --grid-n $GridN --cases B0 V1 V3
} "Screening-resolution explicit-4F parity gate"

if ($SkipFullSuite) {
    Write-Host "`nParity gates passed. Full sweeps skipped by request." -ForegroundColor Yellow
    exit 0
}

Invoke-Checked {
    python tools/run_vortex_system_error_suite.py `
      --cases B0 V1 V3 `
      --families all `
      --grid-n $GridN `
      --z-mm $Zmm
} "All executable physical system-error sweeps"

Invoke-Checked {
    python tools/audit_vortex_system_error_signatures.py
} "System-error numerical/signature audit gate"

Invoke-Checked {
    python tools/run_vortex_declared_aberration_suite.py `
      --cases B0 V1 V3 `
      --grid-n $GridN `
      --z-mm $Zmm
} "All declared-plane Zernike sensitivity sweeps"

Write-Host "`n=== PIPELINE COMPLETE ===" -ForegroundColor Green
Write-Host "Physical-system outputs: outputs/validation/vortex_system_errors and outputs/figures/vortex_system_errors"
Write-Host "Declared-aberration outputs: outputs/validation/vortex_declared_aberrations"
Write-Host "These are screening/research outputs. Final report panels still require selected N=3072 reruns and family-specific validation gates."
