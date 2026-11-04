import hashlib
import json
import zipfile

import pytest

from scripts.reproduce_native_asset import reproduce,inference_auditor


@pytest.mark.parametrize('failure', ['archive_hash', 'file_hash', 'path_escape'])
def test_reproduction_rejects_untrusted_content_before_extracting_or_loading(tmp_path, failure):
    archive = tmp_path/'release.zip'
    name = '../outside.pkl' if failure=='path_escape' else 'batch/model.pkl'
    payload = b'must never be deserialized'
    manifest = {'files':{name:{'bytes':len(payload), 'sha256':
        '0'*64 if failure=='file_hash' else hashlib.sha256(payload).hexdigest()}}}
    with zipfile.ZipFile(archive,'w') as stream:
        stream.writestr(name,payload)
        stream.writestr('manifest.json',json.dumps(manifest))
    expected = '0'*64 if failure=='archive_hash' else hashlib.sha256(archive.read_bytes()).hexdigest()
    output = tmp_path/'reproduction'
    with pytest.raises(ValueError):
        reproduce(archive,expected,output)
    assert not output.exists()
    assert not (tmp_path/'outside.pkl').exists()


@pytest.mark.parametrize('outcome',[False,True])
def test_reference_context_uses_its_registered_observation_auditor(outcome):
    from navstack.reference_context_policy import VERSION
    if outcome:
        from navstack.outcome_context_policy import VERSION
    assert inference_auditor({'reference_context':{'version':VERSION}})=='audit_native_reference_inference.py'
    assert inference_auditor({})=='audit_native_policy_inference.py'


@pytest.mark.parametrize('context',[{}, {'version':'unknown'}, {'version':'appended_reference_context_v1','extra':True},
                                  {'version':'action_aligned_outcome_context_v1','extra':True}])
def test_unknown_context_does_not_fall_back_to_legacy_observations(context):
    with pytest.raises(ValueError,match='Unsupported'):
        inference_auditor({'reference_context':context})


def test_legacy_replay_does_not_require_a_module_absent_from_frozen_sources(monkeypatch):
    import builtins
    from navstack.reference_context_policy import VERSION
    from scripts.audit_native_reference_inference import outcome_codec,OUTCOME_VERSION
    original=builtins.__import__
    def without_outcome(name,*args,**kwargs):
        if name=='navstack.outcome_context_policy':
            raise ModuleNotFoundError('Old frozen source has no outcome codec')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',without_outcome)
    assert outcome_codec(VERSION) is None
    with pytest.raises(ModuleNotFoundError):
        outcome_codec(OUTCOME_VERSION)
