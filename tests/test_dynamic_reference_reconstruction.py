from types import SimpleNamespace
import pytest
from scripts.audit_native_reference_inference import audit,verify_accepted_reference
from navstack.reference_road_refinement import pose_digest


def test_dynamic_rejection_needs_no_road_query_but_cannot_retain_an_unaccepted_reference():
    verify_accepted_reference('dynamic_brake',None,{'probes':[]})
    with pytest.raises(ValueError,match='unaccepted'):verify_accepted_reference('dynamic_brake',object(),{'probes':[]})


def test_accepted_reference_still_requires_exact_native_pose_certificate():
    reference=SimpleNamespace(x=[1.,2.],y=[3.,4.],yaw=[0.,0.])
    with pytest.raises(ValueError,match='lacks native'):verify_accepted_reference('reference',reference,{'probes':[]})
    proof=dict(pose_sha256=pose_digest([[1.,3.,0.],[2.,4.,0.]]),rejected_world=[],truncated=False)
    verify_accepted_reference('reference',reference,{'probes':[{'reference_road_probe':proof}]})
    for changed in ({'pose_sha256':'0'*64},{'rejected_world':[[1,2,3]]},{'truncated':True}):
        with pytest.raises(ValueError):verify_accepted_reference('reference',reference,{'probes':[{'reference_road_probe':{**proof,**changed}}]})


def test_ambiguous_replay_scope_is_rejected_before_reading_or_loading_models():
    with pytest.raises(ValueError,match='not both'):audit('missing','H',all_static=True,all_actions=True)
