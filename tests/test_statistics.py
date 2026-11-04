import pytest
from navstack.statistics import paired_bootstrap,learning_curve_auc


def test_paired_seed_bootstrap_preserves_pairing_and_refuses_episode_pseudoreplication():
    result=paired_bootstrap({11:.1,22:.4,33:.7},{11:0,22:.3,33:.6})
    assert result["n"]==3 and result["ci95"]==pytest.approx([.1,.1])
    assert paired_bootstrap({11:.9},{11:0})["ci95"] is None
    with pytest.raises(ValueError):paired_bootstrap({1:1},{1:0},repetitions=100)


def test_registered_auc_does_not_extrapolate_unrun_milestones():
    assert learning_curve_auc([(100000,.5),(300000,.8)])["value"] is None
    assert learning_curve_auc([(100000,.5),(300000,.5),(1000000,.5)])["value"]==.5
