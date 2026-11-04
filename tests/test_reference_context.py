from copy import deepcopy
import numpy as np
import pytest
import torch
from navstack.reference_context import reference_candidate_context,LearnedReferenceBias
from test_reference_reuse import prepared


def test_context_identifies_only_exact_valid_reference_without_changing_state_or_selection():
    env,reference=prepared()
    try:
        state=deepcopy(env.state);selection=deepcopy(env._reference_selection)
        result=reference_candidate_context(env)
        assert result.shape==(31,) and result[26]==1 and result[0]==0
        for index in np.flatnonzero(result):assert env.decode_action(int(index))==selection
        assert env.state==state and env._reference_selection==selection and env.reference is reference
        env.world.static.append((16.,15.,17.,17.))
        np.testing.assert_array_equal(reference_candidate_context(env),np.zeros(31))
    finally:env.close()


def test_zero_initialized_term_is_exact_and_the_available_bit_is_learned_not_an_override():
    layer=LearnedReferenceBias();logits=torch.zeros(2,31);logits[:,2]=2.
    context=torch.zeros_like(logits);context[:,26]=1
    assert torch.equal(layer(logits,context),logits)
    # A real gradient update can learn the coefficient;availability alone does
    # not choose26over2at initialization.
    assert layer(logits,context).argmax(-1).tolist()==[2,2]
    optimizer=torch.optim.Adam(layer.parameters(),lr=.01)
    loss=torch.nn.functional.cross_entropy(layer(logits,context),torch.tensor([26,26]))
    loss.backward();optimizer.step()
    assert layer.gain.item()>0 and layer(logits,context)[0,26]>0
    assert torch.equal(layer(logits,context,ablated=True),logits)
    with pytest.raises(ValueError):layer(logits,torch.ones_like(logits))
