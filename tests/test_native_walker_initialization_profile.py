from pathlib import Path

import pytest

from scripts.prepare_native_supervised_ppo import copy_walker_spawn_harness

BRIDGE = Path(__file__).resolve().parents[1]/'scripts/carla_bridge37.py'


def test_already_registered_walker_profile_keeps_complete_bridge_bytes(tmp_path):
    target = tmp_path/'copied.py'
    proof = copy_walker_spawn_harness(BRIDGE, BRIDGE, target)
    assert target.read_bytes() == BRIDGE.read_bytes()
    assert proof['version'] == 'native_walker_clearance_020_v1'


@pytest.mark.parametrize('fault', ['extra_code', 'wrong_height', 'removed_evidence'])
def test_walker_profile_never_hides_additional_behavior_changes(tmp_path, fault):
    source = BRIDGE.read_text()
    if fault == 'extra_code':changed = source+'\nunrelated_actor_behavior = True\n'
    elif fault == 'wrong_height':changed = source.replace('walker_transform.location.z+=.2', 'walker_transform.location.z+=2.')
    else:changed = source.replace("'clearance_added_m':.2", "'clearance_added_m':0.")
    assert changed != source
    current = tmp_path/'changed.py';current.write_text(changed)
    target = tmp_path/'never.py'
    with pytest.raises(ValueError):copy_walker_spawn_harness(BRIDGE, current, target)
    assert not target.exists() and BRIDGE.read_text() == source
