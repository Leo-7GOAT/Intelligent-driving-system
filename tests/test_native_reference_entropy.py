"""Explicit training-only entropy changes must fail before staging invalid jobs."""
import pytest
from scripts.prepare_native_reference_context import prepare,validate_entropy
from scripts.prepare_native_reference_learning import prepare as prepare_learning


@pytest.mark.parametrize('value',[True,False,-.01,1.01,float('nan'),float('inf'),'0.02',[]])
def test_invalid_entropy_is_rejected_before_creating_initializer(tmp_path,value):
    output=tmp_path/'initial'
    with pytest.raises(ValueError,match='Entropy coefficient'):
        prepare(output,methods=['H'],entropy_coef=value)
    assert not output.exists()


def test_entropy_defaults_preserve_old_setting_and_zero_is_explicit():
    assert validate_entropy(None) is None
    assert validate_entropy(0)==0.0
    assert validate_entropy(.02)==.02


def test_memo_cannot_be_enabled_by_a_truthy_non_boolean(tmp_path):
    output=tmp_path/'training'
    with pytest.raises(ValueError,match='boolean'):
        prepare_learning(output,exact_failure_memo='false')
    assert not output.exists()
