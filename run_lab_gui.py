"""Launch the v0.6 measurement GUI; optionally locate an existing HEDS install."""
from pathlib import Path
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dual SLM lab GUI with measurement sessions')
    parser.add_argument('--sdk-path', type=Path, help='Parent directory containing the existing HEDS/ package; SDK/hedslib must already be installed.')
    args, qt_args = parser.parse_known_args()
    if args.sdk_path:
        if not (args.sdk_path/'HEDS').is_dir():
            parser.error('--sdk-path must contain an HEDS subdirectory')
        sys.path.insert(0, str(args.sdk_path.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent/'lab_gui'))
    sys.argv = [sys.argv[0]]+qt_args
    from slm_lab_control.app import main
    raise SystemExit(main())
