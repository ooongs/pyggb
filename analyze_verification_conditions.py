#!/usr/bin/env python3
"""
Comprehensive analysis of verification_conditions in geoqa3_dataset.json
Compares dataset formats with dsl_validator.py expectations.
"""

import json
import re
from collections import defaultdict
from typing import Dict, Set


def load_validator_params(validator_path: str) -> Dict[str, Set[str]]:
    """Extract expected parameters from dsl_validator.py"""
    with open(validator_path, 'r', encoding='utf-8') as f:
        content = f.read()

    validator_params = {}
    check_methods = re.findall(r'def (_check_\w+)\(self, data: Dict\)', content)

    for method_name in check_methods:
        pattern = rf'def {method_name}\(self, data: Dict\).*?(?:\n    def |\nclass |\Z)'
        match = re.search(pattern, content, re.DOTALL)

        if match:
            method_body = match.group(0)[:2000]
            cond_type = method_name.replace('_check_', '')
            params = set(re.findall(r'data\.get\(["\'](\w+)["\']', method_body))
            validator_params[cond_type] = params

    return validator_params


def load_dataset_params(dataset_path: str) -> Dict[str, Set[str]]:
    """Extract actual parameters from dataset verification_conditions"""
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset_params = defaultdict(set)

    for problem in data['problems']:
        if 'verification_conditions' in problem:
            for cond in problem['verification_conditions']:
                cond_type = cond.get('type', 'UNKNOWN')
                for key in cond.keys():
                    if key != 'type':
                        dataset_params[cond_type].add(key)

    return dict(dataset_params)


def load_condition_aliases(validator_path: str) -> Dict[str, str]:
    """Extract condition type aliases from check_condition method"""
    with open(validator_path, 'r', encoding='utf-8') as f:
        content = f.read()

    aliases = {}

    # Find all elif statements with multiple condition types
    pattern = r'elif condition_type in \[(.*?)\]:\s*return self\.(_check_\w+)'
    matches = re.findall(pattern, content, re.DOTALL)

    for condition_list, method_name in matches:
        canonical = method_name.replace('_check_', '')
        types = re.findall(r'"(\w+)"', condition_list)
        for t in types:
            if t != canonical:
                aliases[t] = canonical

    # Also find direct mappings
    pattern = r'elif condition_type == "(\w+)":\s*return self\.(_check_\w+)'
    matches = re.findall(pattern, content)

    for cond_type, method_name in matches:
        canonical = method_name.replace('_check_', '')
        if cond_type != canonical:
            aliases[cond_type] = canonical

    return aliases


def analyze_compatibility(validator_params, dataset_params, aliases):
    """Compare validator expectations with dataset reality"""
    results = {
        'compatible': [],
        'parameter_mismatch': [],
        'missing_in_validator': [],
        'unused_in_dataset': []
    }

    all_dataset_types = set(dataset_params.keys())
    all_validator_types = set(validator_params.keys())

    for cond_type in all_dataset_types:
        # Resolve aliases
        canonical_type = aliases.get(cond_type, cond_type)

        if canonical_type not in validator_params:
            results['missing_in_validator'].append({
                'type': cond_type,
                'canonical': canonical_type if canonical_type != cond_type else None,
                'params': sorted(dataset_params[cond_type])
            })
            continue

        dataset_p = dataset_params[cond_type]
        validator_p = validator_params[canonical_type]

        only_in_dataset = dataset_p - validator_p
        only_in_validator = validator_p - dataset_p

        if only_in_dataset or only_in_validator:
            results['parameter_mismatch'].append({
                'type': cond_type,
                'canonical': canonical_type if canonical_type != cond_type else None,
                'extra_in_dataset': sorted(only_in_dataset),
                'missing_in_dataset': sorted(only_in_validator),
                'common': sorted(dataset_p & validator_p)
            })
        else:
            results['compatible'].append({
                'type': cond_type,
                'canonical': canonical_type if canonical_type != cond_type else None,
                'params': sorted(dataset_p)
            })

    for cond_type in all_validator_types:
        if cond_type not in dataset_params and cond_type not in aliases.values():
            results['unused_in_dataset'].append({
                'type': cond_type,
                'params': sorted(validator_params[cond_type])
            })

    return results


def print_report(results, dataset_params):
    """Print comprehensive compatibility report"""
    print("="*80)
    print("VERIFICATION CONDITIONS COMPATIBILITY REPORT")
    print("="*80)

    # Summary
    total_types = len(dataset_params)
    compatible = len(results['compatible'])
    mismatches = len(results['parameter_mismatch'])
    missing = len(results['missing_in_validator'])

    print(f"\n📊 SUMMARY")
    print(f"   Total condition types in dataset: {total_types}")
    print(f"   ✓ Fully compatible: {compatible}")
    print(f"   ⚠ Parameter mismatch: {mismatches}")
    print(f"   ❌ Missing in validator: {missing}")
    print(f"   📝 Unused by dataset: {len(results['unused_in_dataset'])}")

    # Compatible types
    print(f"\n{'='*80}")
    print(f"✓ FULLY COMPATIBLE TYPES ({compatible})")
    print(f"{'='*80}")
    for item in sorted(results['compatible'], key=lambda x: x['type']):
        type_name = item['type']
        if item['canonical']:
            type_name += f" (alias for {item['canonical']})"
        print(f"\n  {type_name}")
        print(f"    Parameters: {', '.join(item['params'])}")

    # Parameter mismatches
    if results['parameter_mismatch']:
        print(f"\n{'='*80}")
        print(f"⚠ PARAMETER MISMATCHES ({mismatches})")
        print(f"{'='*80}")
        for item in sorted(results['parameter_mismatch'], key=lambda x: x['type']):
            type_name = item['type']
            if item['canonical']:
                type_name += f" (alias for {item['canonical']})"
            print(f"\n  {type_name}")

            if item['extra_in_dataset']:
                print(f"    ❌ In dataset but NOT in validator:")
                for param in item['extra_in_dataset']:
                    print(f"       - {param}")

            if item['missing_in_dataset']:
                print(f"    📝 Expected by validator but NOT in dataset:")
                for param in item['missing_in_dataset']:
                    print(f"       - {param}")

            if item['common']:
                print(f"    ✓ Common: {', '.join(item['common'])}")

    # Missing in validator
    if results['missing_in_validator']:
        print(f"\n{'='*80}")
        print(f"❌ TYPES IN DATASET BUT NOT IN VALIDATOR ({missing})")
        print(f"{'='*80}")
        for item in sorted(results['missing_in_validator'], key=lambda x: x['type']):
            type_name = item['type']
            if item['canonical']:
                type_name += f" → {item['canonical']} (alias not found)"
            print(f"\n  {type_name}")
            print(f"    Parameters: {', '.join(item['params'])}")

    # Unused in dataset
    if results['unused_in_dataset']:
        print(f"\n{'='*80}")
        print(f"📝 TYPES IN VALIDATOR BUT NOT USED IN DATASET ({len(results['unused_in_dataset'])})")
        print(f"{'='*80}")
        for item in sorted(results['unused_in_dataset'], key=lambda x: x['type']):
            print(f"\n  {item['type']}")
            print(f"    Expected parameters: {', '.join(item['params'])}")

    print(f"\n{'='*80}")


def main():
    dataset_path = 'data/geoqa3_dataset.json'
    validator_path = 'src/dsl/dsl_validator.py'

    print("Loading validator parameters...")
    validator_params = load_validator_params(validator_path)
    print(f"  Found {len(validator_params)} check methods")

    print("Loading dataset parameters...")
    dataset_params = load_dataset_params(dataset_path)
    print(f"  Found {len(dataset_params)} condition types")

    print("Loading condition aliases...")
    aliases = load_condition_aliases(validator_path)
    print(f"  Found {len(aliases)} type aliases")

    print("\nAnalyzing compatibility...\n")
    results = analyze_compatibility(validator_params, dataset_params, aliases)

    print_report(results, dataset_params)


if __name__ == '__main__':
    main()
