"""
neurobrix validate — NBX integrity check command.
"""

import sys
from pathlib import Path


def cmd_validate(args):
    """Validate NBX file(s) integrity."""
    from neurobrix.core.validators.nbx_validator import NBXValidator, ValidationLevel, NBXValidationError
    import json

    level_map = {
        'structure': ValidationLevel.STRUCTURE,
        'schema': ValidationLevel.SCHEMA,
        'coherence': ValidationLevel.COHERENCE,
        'deep': ValidationLevel.DEEP,
    }
    level = level_map[args.level]

    validator = NBXValidator(level=level)

    results = []
    all_valid = True

    for nbx_path in args.nbx_files:
        nbx_path = Path(nbx_path)

        if not nbx_path.exists():
            print(f"ERROR: File not found: {nbx_path}")
            all_valid = False
            continue

        if args.verbose:
            print(f"\nValidating: {nbx_path}")
            print(f"  Level: {args.level}")

        try:
            if args.strict:
                result = validator.validate_strict(nbx_path)
            else:
                result = validator.validate(nbx_path)

            results.append({
                'path': str(nbx_path),
                'valid': result.is_valid,
                'errors': [str(e) for e in result.errors],
                'warnings': result.warnings,
                'stats': result.stats,
            })

            if not result.is_valid:
                all_valid = False

            if not args.json:
                status = "VALID" if result.is_valid else f"INVALID ({len(result.errors)} errors)"
                print(f"{nbx_path.name}: {status}")

                if args.verbose or not result.is_valid:
                    if result.stats:
                        model_name = result.stats.get('model_name', 'unknown')
                        components = result.stats.get('neural_components', [])
                        print(f"  Model: {model_name}")
                        print(f"  Components: {components}")

                    for error in result.errors[:10]:
                        print(f"  ERROR: {error}")
                    if len(result.errors) > 10:
                        print(f"  ... and {len(result.errors) - 10} more errors")

                    for warning in result.warnings[:5]:
                        print(f"  WARN: {warning}")

        except NBXValidationError as e:
            print(f"VALIDATION FAILED: {nbx_path}")
            for error in e.result.errors[:10]:
                print(f"  ERROR: {error}")
            all_valid = False
            sys.exit(1)

    if args.json:
        print(json.dumps(results, indent=2))

    if len(args.nbx_files) > 1 and not args.json:
        valid_count = sum(1 for r in results if r['valid'])
        print(f"\nSummary: {valid_count}/{len(results)} files valid")

    sys.exit(0 if all_valid else 1)
