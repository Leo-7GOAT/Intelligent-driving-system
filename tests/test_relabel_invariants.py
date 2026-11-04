import numpy as np
import pytest

from scripts.verify_relabel_teacher import assert_action_only_change, verify


def test_incomplete_relabel_cannot_authorize_fitting(tmp_path):
    (tmp_path/'protocol.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='fitting requires all groups'):
        verify(tmp_path)


def test_relabel_rejects_changed_visits_returns_splits_and_invalid_actions():
    old = {'observations': np.zeros((2, 3), dtype=np.float32),
           'actions': np.zeros((2, 2), dtype=np.float32),
           'returns': np.zeros(2, dtype=np.float32),
           'train_mask': np.array([True, False])}
    new = {key: value.copy() for key, value in old.items()}
    new['actions'][0] = [.5, -.2]
    assert_action_only_change(old, new)
    for key in ('observations', 'returns', 'train_mask'):
        changed = {name: value.copy() for name, value in new.items()}
        changed[key].flat[0] = not changed[key].flat[0] if key == 'train_mask' else 1
        with pytest.raises(AssertionError):
            assert_action_only_change(old, changed)
    for value in (float('nan'), 1.01):
        new['actions'][0, 0] = value
        with pytest.raises(AssertionError):
            assert_action_only_change(old, new)
