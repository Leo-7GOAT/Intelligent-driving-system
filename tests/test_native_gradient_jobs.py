import pytest
from pathlib import Path
import shutil
from navstack.independent_clip_ppo import GLOBAL_CLIP, INDEPENDENT_CLIP
from scripts.native_road_learning import registered_jobs
from navstack.experiments import write_json
import scripts.prepare_native_gradient_pair as pairing


def test_legacy_jobs_keep_original_methods_without_intervention():
    assert registered_jobs({'methods':['H0','H','B1']}) == [
        {'name':method,'method':method} for method in ('H0','H','B1')]


def test_paired_jobs_require_explicit_protocol_and_unique_scopes():
    jobs=[{'name':'H_global','method':'H','gradient_clip_mode':GLOBAL_CLIP},
          {'name':'H_independent','method':'H','gradient_clip_mode':INDEPENDENT_CLIP}]
    protocol={'methods':[j['name'] for j in jobs],'jobs':jobs,'gradient_clipping_experiment':True}
    assert registered_jobs(protocol)==jobs
    with pytest.raises(ValueError,match='registered'):
        registered_jobs({**protocol,'gradient_clipping_experiment':False})
    with pytest.raises(ValueError,match='uniquely'):
        registered_jobs({**protocol,'methods':['H','H']})
    with pytest.raises(ValueError,match='Invalid'):
        registered_jobs({**protocol,'methods':['../H','H_independent'],
            'jobs':[{**jobs[0],'name':'../H'},jobs[1]]})
    with pytest.raises(ValueError,match='declare'):
        registered_jobs({**protocol,'jobs':[{**jobs[0],'gradient_clip_mode':'typo'},jobs[1]]})


def test_gradient_pair_cannot_hide_active_native_continuation(tmp_path, monkeypatch):
    (tmp_path/'runs/active').mkdir(parents=True)
    write_json(tmp_path/'runs/native_continuation.json',{'batch':str(tmp_path/'runs/active')})
    monkeypatch.setattr(pairing,'ROOT',tmp_path)
    monkeypatch.setattr(pairing,'resources_ok',lambda path:(True,None))
    with pytest.raises(RuntimeError,match='active or non-terminal'):
        pairing.prepare(tmp_path/'runs/active')
    assert not (tmp_path/'runs/native_gradient_pair.json').exists()
    assert [p.name for p in (tmp_path/'runs').iterdir() if p.is_dir()]==['active']


@pytest.fixture
def completed_pair_parent(tmp_path, monkeypatch):
    """Synthetic file fixture tests registration, never simulator/learner evidence."""
    source_root = pairing.ROOT
    parent = tmp_path/'runs/parent'
    (parent/'source/navstack').mkdir(parents=True)
    (parent/'source/scripts').mkdir()
    (parent/'calibration').mkdir()
    (parent/'infrastructure').mkdir()
    (parent/'source/navstack/env.py').write_text('# immutable task fixture\n')
    # Copy only source text needed by staging, without executing it.
    for directory in ('scripts', 'navstack'):
        (tmp_path/directory).mkdir()
        for pattern in ('*.py', '*.ps1'):
            for file in (source_root/directory).glob(pattern):
                shutil.copy2(file, tmp_path/directory/file.name)
    write_json(parent/'protocol.json', {'methods':['H0','H','B1'],
        'resource_limits':{'wall_seconds':7200}, 'seed':11,
        'combined_endpoint_manifest_id':'obsolete-parent-id'})
    write_json(parent/'summary.json', {'status':'complete'})
    write_json(parent/'guard_final.json', {'exit_code':0,'reason':None})
    write_json(parent/'infrastructure/server_stop.json', {'status':'owned_empty_server_stopped'})
    records = {}
    for index, method in enumerate(('H0','H','B1')):
        checkpoint = parent/method/'checkpoint_000100000'
        checkpoint.mkdir(parents=True)
        for name in ('model.zip','normalization.pkl','resume_state.pkl'):
            (checkpoint/name).write_bytes((method+name).encode())
        lineage = 100000+index*10
        write_json(parent/method/'summary.json', {'lineage_control_steps':lineage})
        records[method] = {'method':method, 'source_batch':{'absolute':str(parent)},
            'checkpoint':checkpoint.name,
            'checkpoint_hashes':{p.name:pairing.sha(p) for p in checkpoint.iterdir()},
            'lineage_controls':lineage, 'charged_controls_lower_bound':lineage+9545,
            'optimizer_steps_total_lower_bound':626,
            'unknown_unflushed_ancestor_tail':True, 'recovery_chain':['ancestor']}
    manifest = {'manifest_id':'verified-fixture', 'target_controls':120020,'sources':records}
    monkeypatch.setattr(pairing,'ROOT',tmp_path)
    monkeypatch.setattr(pairing,'resources_ok',lambda path:(True,None))
    monkeypatch.setattr(pairing,'build_manifest',lambda *args,**kwargs:manifest)
    return parent, records


def test_pair_staging_preserves_both_arms_and_charges_parent_once(completed_pair_parent):
    parent, records = completed_pair_parent
    child = pairing.prepare(parent)
    protocol = pairing.read(child/'protocol.json')
    assert len(registered_jobs(protocol)) == 6
    assert 'combined_endpoint_manifest_id' not in protocol
    assert protocol['parent_endpoint_manifest_id'] == 'verified-fixture'
    assert (child/'source/navstack/env.py').read_bytes() == (parent/'source/navstack/env.py').read_bytes()
    for method, record in records.items():
        for arm in ('global','independent'):
            name = method+'_'+arm
            assert protocol['method_budgets'][name] == 20000
            assert protocol['lineage_targets'][name] == record['lineage_controls']+20000
            recovery = protocol['recovery'][name]
            assert recovery['consumed_controls_lower_bound'] == record['charged_controls_lower_bound']
            assert recovery['unknown_unflushed_parent_tail'] is True
            assert recovery['recovery_chain'] == ['ancestor','parent']
            assert {p.name:pairing.sha(p) for p in (child/recovery['checkpoint']).iterdir()} == record['checkpoint_hashes']
    ledger = pairing.read(parent.parent/'native_endpoint_claims.json')
    assert len(ledger['claims']) == 3
    assert {row['child_batch'] for row in ledger['claims'].values()} == {child.name}
    assert Path(pairing.read(parent.parent/'native_gradient_pair.json')['batch']) == child
    with pytest.raises(RuntimeError, match='terminal guard'):
        pairing.prepare(parent)


def test_corrupt_pair_checkpoint_never_claims_parent(completed_pair_parent):
    parent, records = completed_pair_parent
    (parent/'H0'/records['H0']['checkpoint']/'model.zip').write_bytes(b'corrupt after verification')
    with pytest.raises(ValueError, match='checkpoint changed'):
        pairing.prepare(parent)
    assert not (parent.parent/'native_endpoint_claims.json').exists()
    assert not (parent.parent/'native_gradient_pair.json').exists()
    aborted = list(parent.parent.glob('*_native_gradient_pair/aborted.json'))
    assert len(aborted) == 1
    assert pairing.read(aborted[0])['status'] == 'ABORTED_UNREGISTERED_DO_NOT_RUN'
