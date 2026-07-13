#!/usr/bin/env python3
"""Provenance check: compare saved model metadata to pinned pipeline feature lists."""
from pathlib import Path
import json

from pipeline import ALL_FEATURES, MULTI_ASSET_FEATURES, MACRO_FEATURES, MICRO_FEATURES

MODEL_DIR = Path(__file__).parent / 'models'
OUT = MODEL_DIR / 'feature_provenance_report.json'

FEATURE_SETS = {
    'ALL_FEATURES': ALL_FEATURES,
    'MULTI_ASSET_FEATURES': MULTI_ASSET_FEATURES,
    'MACRO_FEATURES': MACRO_FEATURES,
    'MICRO_FEATURES': MICRO_FEATURES,
}

results = {}
for p in MODEL_DIR.glob('latest_meta.json'):
    try:
        obj = json.loads(p.read_text())
        stored = obj.get('features', [])
    except Exception as e:
        results[p.name] = {'status': 'load_error', 'error': str(e)}
        continue
    entry = {
        'status': 'ok',
        'saved_feature_count': len(stored),
        'timestamp': obj.get('timestamp'),
    }
    for name, expected_list in FEATURE_SETS.items():
        stored_set = set(stored)
        expected_set = set(expected_list)
        entry[name + '_count'] = len(expected_list)
        entry[name + '_missing_in_saved'] = sorted(expected_set - stored_set)
        entry[name + '_extra_in_saved'] = sorted(stored_set - expected_set)
        entry[name + '_matches'] = sorted(stored_set) == sorted(expected_set)
    if not stored:
        entry['note'] = 'saved model has empty feature_names; provenance cannot be verified until model_trainer.py sets/persists feature names'
    results[p.name] = entry
    print('FILE:', p.name)
    print('  saved_feature_count:', len(stored))
    print('  matches ALL_FEATURES:', entry.get('ALL_FEATURES_matches'))
    print('  missing in saved:', len(entry.get('ALL_FEATURES_missing_in_saved', [])))

report = {
    'expected_feature_count': len(ALL_FEATURES),
    'expected_feature_hash': str(hash(tuple(ALL_FEATURES))),
    'checked_files': len(results),
    'results': results,
}
OUT.write_text(json.dumps(report, indent=2))
print('[FP] saved', OUT)
