import pytest
from scripts.audit_native_candidates import match_scenes


def test_resume_candidate_audit_matches_geometry_not_training_order():
    def scene(key):return {'scene':{'native_layout':{'geometry_id':key}}}
    routes=[{'geometry_id':key} for key in ('a','b','c','d')]
    scenes=[scene(key) for key in ('c','d','a','b','c')]
    assert match_scenes(routes,scenes)==[scenes[2],scenes[3],scenes[0],scenes[1]]
    with pytest.raises(ValueError,match='Missing measured training geometry'):
        match_scenes(routes,scenes[:3])
