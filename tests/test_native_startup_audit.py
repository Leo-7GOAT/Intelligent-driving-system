import json

import pytest

from scripts.audit_interrupted_native import audit


def test_prebridge_failure_is_recoverable_but_unknown_activity_is_not(tmp_path):
    (tmp_path/'infrastructure').mkdir()
    (tmp_path/'H0_global').mkdir()
    (tmp_path/'protocol.json').write_text(json.dumps({'methods':['H0_global','H0_independent']}))
    (tmp_path/'H0_global/config.json').write_text('{}')
    (tmp_path/'failure.json').write_text(json.dumps({
        'error': 'FileNotFoundError: CARLA client interpreter not found: isolated-python'}))
    # An unexpected activity artifact must not be discarded as a zero-cost run.
    trace = tmp_path/'H0_global/transitions.jsonl.gz'
    trace.write_bytes(b'')
    with pytest.raises(ValueError, match='unverified interaction'):
        audit(tmp_path)
    assert not (tmp_path/'interruption_audit.json').exists()
    trace.unlink()
    audit(tmp_path)
    result = json.loads((tmp_path/'interruption_audit.json').read_text())
    assert all(row['state']=='not_started' and row['consumed_controls_lower_bound']==0
               for row in result['methods'].values())
    assert (tmp_path/'H0_global/config.json').exists()
