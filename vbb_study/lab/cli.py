from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import core


def main(argv=None):
    parser = argparse.ArgumentParser(description='SLM2 measurement-to-correction workbench; manual camera acquisition.')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ['init', 'start', 'plan', 'capture', 'evaluate', 'verify-plan', 'accept', 'status', 'export', 'report']:
        cmd = sub.add_parser(name)
        cmd.add_argument('session', type=Path)
        if name == 'start':
            cmd.add_argument('--base', required=True, type=Path)
            cmd.add_argument('--dark', required=True, nargs='+', type=Path)
        elif name == 'plan':
            cmd.add_argument('--mode', choices=core.MODES, required=True)
            cmd.add_argument('--values', nargs='+', type=float, default=[-.3, -.15, .15, .3])
        elif name == 'capture':
            cmd.add_argument('trial')
            cmd.add_argument('--folder', type=Path)
            cmd.add_argument('--settings-id', required=True)
        elif name == 'export':
            cmd.add_argument('trial')
            cmd.add_argument('--lut', type=Path)
            cmd.add_argument('--raw-path-verified', action='store_true')
    audit = sub.add_parser('audit-calibration')
    audit.add_argument('folder', type=Path)
    audit.add_argument('--output', required=True, type=Path)
    demo = sub.add_parser('demo')
    demo.add_argument('folder', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            result = {'edit_profile': str(core.init_session(args.session)), 'next': 'Fill measured settings, then start with the GUI-exported base phase and dark frames.'}
        elif args.command == 'start':
            core.start_session(args.session, args.base, args.dark)
            result = {'status': 'SESSION_STARTED', 'next': 'plan --mode astig_x'}
        elif args.command == 'plan':
            result = core.plan_round(args.session, args.mode, args.values)
        elif args.command == 'capture':
            result = core.capture(args.session, args.trial, args.folder, args.settings_id)
            result = {'trial': args.trial, 'frames': len(result['frames']), 'status': 'CAPTURE_SEALED'}
        elif args.command == 'evaluate':
            result = core.evaluate(args.session)
        elif args.command == 'verify-plan':
            result = core.plan_verification(args.session)
        elif args.command == 'accept':
            result = core.accept(args.session)
        elif args.command == 'export':
            result = core.export_phase(args.session, args.trial, args.lut, args.raw_path_verified)
        elif args.command == 'report':
            from .report import build_report
            result = {'report': str(build_report(args.session))}
        elif args.command == 'audit-calibration':
            from .legacy import audit_calibration
            result = audit_calibration(args.folder, args.output)
        elif args.command == 'demo':
            from .demo import run_demo
            result = run_demo(args.folder)
        else:
            path, state, profile = core.session(args.session)
            result = {'accepted_trial': state['accepted_trial'], 'phase_owner': profile['slm']['phase_owner'],
                      'rounds': [{**r, 'pending': [tid for tid in r['trials']+r.get('verification', [])
                        if not (path/'trials'/tid/'measurement.json').exists()]} for r in state['rounds']]}
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, IndexError) as exc:
        parser.exit(2, f'Cannot complete: {exc}\n')
