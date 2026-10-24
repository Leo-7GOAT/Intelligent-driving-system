from pathlib import Path
import shutil

import pytest

import scripts.prepare_native_rollout_pair as pairing
from navstack.experiments import write_json


@pytest.fixture
def staged_parent(tmp_path,monkeypatch):
    """Synthetic staging only; real PPO state retention has a separate plant test."""
    original_root=pairing.ROOT
    parent=tmp_path/'runs/parent'
    for folder in ('source/navstack','source/scripts','calibration','infrastructure'):
        (parent/folder).mkdir(parents=True)
    (parent/'source/navstack/env.py').write_text('# frozen environment fixture')
    for name in pairing.REFRESH_FILES:
        (tmp_path/name).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(original_root/name,tmp_path/name)
    write_json(parent/'protocol.json',{'resource_limits':{},'rollout_actions':32,'ppo_epochs':2,
        'gradient_clipping_experiment':True,'seed':11})
    records={}
    for method in ('H0','H','B1'):
        name=method+'_global'
        checkpoint=parent/name/'checkpoint_end'
        checkpoint.mkdir(parents=True)
        for filename in ('model.zip','normalization.pkl','resume_state.pkl'):
            (checkpoint/filename).write_bytes((method+filename).encode())
        records[method]={'method':name,'source_batch':{'root_relative':'runs/parent'},
            'source_method_folder':name,'checkpoint':checkpoint.name,
            'checkpoint_hashes':{p.name:pairing.sha(p) for p in checkpoint.iterdir()},
            'lineage_controls':120000,'charged_controls_lower_bound':129545,
            'optimizer_steps_total_lower_bound':752,'unknown_unflushed_ancestor_tail':True,
            'recovery_chain':['ancestor']}
    monkeypatch.setattr(pairing,'ROOT',tmp_path)
    monkeypatch.setattr(pairing,'resources_ok',lambda root:(True,None))
    monkeypatch.setattr(pairing,'parent_manifest',lambda *args:{'manifest_id':'fixture',
        'target_controls':222400,'sources':records})
    return parent,records


def test_rollout_staging_retains_six_arms_and_claims_exact_parent_jobs(staged_parent):
    parent,records=staged_parent
    child=pairing.prepare(parent)
    protocol=pairing.read(child/'protocol.json')
    assert protocol['rollout_length_experiment'] and 'gradient_clipping_experiment' not in protocol
    assert len(protocol['jobs'])==6 and set(protocol['method_budgets'].values())=={102400}
    assert (child/'source/navstack/env.py').read_bytes()==(parent/'source/navstack/env.py').read_bytes()
    for job in protocol['jobs']:
        inherited=protocol['recovery'][job['name']]
        assert inherited['consumed_controls_lower_bound']==129545 and inherited['unknown_unflushed_parent_tail']
        assert {p.name:pairing.sha(p) for p in (child/inherited['checkpoint']).iterdir()}==records[job['method']]['checkpoint_hashes']
    ledger=pairing.read(parent.parent/'native_endpoint_claims.json')
    assert len(ledger['claims'])==3
    assert Path(pairing.read(parent.parent/'native_rollout_pair.json')['batch'])==child
    with pytest.raises(RuntimeError,match='terminal guard'):
        pairing.prepare(parent)


def test_corrupt_copy_does_not_consume_parent_claim(staged_parent):
    parent,_=staged_parent
    (parent/'H_global/checkpoint_end/model.zip').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):
        pairing.prepare(parent)
    assert not (parent.parent/'native_endpoint_claims.json').exists()
    assert not (parent.parent/'native_rollout_pair.json').exists()
    assert len(list(parent.parent.glob('*_native_rollout_pair/aborted.json')))==1
