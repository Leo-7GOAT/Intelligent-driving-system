import json
import numpy as np
import pytest
from scripts.replay_native_first_ppo_update import scalar_json


def test_observed_sb3_numpy_metrics_can_be_preserved_as_json():
    metrics={'loss':np.float32(.5),'steps':np.int64(40)}
    assert json.loads(json.dumps(metrics,default=scalar_json))=={'loss':.5,'steps':40}


def test_unknown_objects_are_not_silently_stringified_as_numeric_evidence():
    with pytest.raises(TypeError):json.dumps({'metric':object()},default=scalar_json)
