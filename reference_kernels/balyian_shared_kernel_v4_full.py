"""Compatibility shim for the full shared-kernel filename.

The active implementation lives in ``balyian_shared_kernel_v4.py``.  This file
executes that kernel in-place so notebooks that look for the older ``*_full``
name still load the same code path and physics conventions.
"""

from pathlib import Path

_shared = Path(__file__).with_name("balyian_shared_kernel_v4.py")
exec(compile(_shared.read_text(encoding="utf-8"), str(_shared), "exec"), globals(), globals())
