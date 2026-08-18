"""Compatibility shim for the q=20 closed-loop controller.

The legacy implementation that consumed
`UNCALIBRATED_DO_NOT_APPLY_q20_modal_correction.npy` has been disabled.  All
callers are routed to the calibrated Miao-style controller in
`iterative_correction_controller_v2.py`.
"""
from iterative_correction_controller_v2 import propose_iteration, evaluate_experimental_update

__all__ = ["propose_iteration", "evaluate_experimental_update"]
