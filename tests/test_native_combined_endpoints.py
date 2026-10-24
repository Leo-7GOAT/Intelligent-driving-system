"""Multi-batch endpoint composition preserves semantics, costs and claims."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from navstack.experiments import write_json
import scripts.combine_native_endpoints as combined
import scripts.continue_native_learning as continuation


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _config(method):
    env={"method":"H" if method=="H0" else method,
         "execution_feedback":"ablated" if method=="H0" else "enabled",
         "shared_task":"same","vehicle":{"wheelbase":2.875},"flat_hold":10}
    signature={"method":env["method"],"execution_feedback":env["execution_feedback"],
               "action_catalog":"hierarchy" if method!="B1" else "direct",
               "observation":"shared","road_geometry":"v3"}
    return {"env":env,"signature":signature,
            "scope":"recovery provenance" if method=="B1" else "original paired provenance"}


def _make_source(root,method,charged=20160,prior=None,shared="same"):
    batch=root/f"{method}_batch";folder=batch/method;checkpoint=folder/'checkpoint_000020000'
    checkpoint.mkdir(parents=True);(batch/'source/navstack').mkdir(parents=True);(batch/'calibration').mkdir()
    (batch/'source/navstack/core.py').write_text('LEARNING_SEMANTICS = 1\n',encoding='utf-8')
    write_json(batch/'calibration/calibration.json',{"vehicle":"same"})
    for name,value in (("model.zip",method.encode()),("normalization.pkl",b'norm'+method.encode()),
                       ("resume_state.pkl",b'rng'+method.encode())):
        (checkpoint/name).write_bytes(value)
    config=_config(method);write_json(folder/'config.json',config);write_json(checkpoint/'config.json',config)
    metadata={"physics_steps":20160,"lineage_control_steps":20160,"optimizer_steps":62 if prior else 126,
              "model_sha256":_digest(checkpoint/'model.zip'),
              "normalization_sha256":_digest(checkpoint/'normalization.pkl'),
              "resume_state_sha256":_digest(checkpoint/'resume_state.pkl')}
    write_json(checkpoint/'metadata.json',metadata)
    write_json(folder/'latest_checkpoint.json',{"folder":checkpoint.name})
    summary={"physics_steps":20160,"lineage_control_steps":20160,
             "charged_control_steps_lower_bound":charged,"optimizer_steps":metadata["optimizer_steps"],
             "optimization_phases":63,
             "reload_exact":True,"stopped_reason":None}
    write_json(folder/'summary.json',summary)
    write_json(folder/'evaluation.json',{"model_sha256":metadata["model_sha256"],
                                          "normalization_sha256":metadata["normalization_sha256"],
                                          "episodes":[{"success":False}]})
    protocol={"methods":[method],"seed":11,"controlled_budget_per_method":20000,
              "rollout_actions":32,"ppo_epochs":2,"episode_seconds":45,
              "road_candidate_policy":True,"road_geometry_version":"v3",
              "deterministic_training_route_evaluation":True,
              "reset_ticks_counted_separately":True,"hypothesis":"shared",
              "routes":[{"geometry_id":shared}],"resource_limits":{"wall_seconds":7200},
              "source_hashes":{"navstack/core.py":_digest(batch/'source/navstack/core.py')},
              "calibration_sha256":_digest(batch/'calibration/calibration.json')}
    if prior:protocol['recovery']={method:prior}
    write_json(batch/'protocol.json',protocol)
    write_json(batch/'interruption_audit.json',{"methods":{method:{"state":"complete",
        "readable_trace_controls":20160,"consumed_controls_lower_bound":20160,
        "trace_errors":[],"optimization_error":None,"unknown_unflushed_tail":False,
        "last_optimization":{"physics_steps":20160,"optimization_phases":63,
                             "optimizer_steps":metadata["optimizer_steps"]}}}})
    return folder


@pytest.fixture
def fake_endpoints(tmp_path,monkeypatch):
    root=tmp_path/'repo';(root/'runs').mkdir(parents=True)
    sources={
        'H0':_make_source(root,'H0'),
        'H':_make_source(root,'H'),
        'B1':_make_source(root,'B1',charged=29545,prior={
            "retained_optimizer_steps":64,"unknown_unflushed_parent_tail":True,
            "recovery_chain":["original","recovery_a"]}),
    }
    def verified(folder,target):
        folder=Path(folder);latest=json.loads((folder/'latest_checkpoint.json').read_text())
        checkpoint=folder/latest['folder']
        return checkpoint,json.loads((folder/'summary.json').read_text()),json.loads((checkpoint/'metadata.json').read_text())
    monkeypatch.setattr(combined,'verified_endpoint',verified)
    def raw_audit(batch,method):
        summary=json.loads((Path(batch)/method/'summary.json').read_text())
        protocol=json.loads((Path(batch)/'protocol.json').read_text())
        prior=protocol.get('recovery',{}).get(method,{})
        return {"schema_version":1,"method":method,
                "training_control_frames_verified":summary['physics_steps'],
                "evaluation_control_frames_verified":4,
                "total_applied_controls_verified":summary['physics_steps']+4,
                "bridge_response_sha256":"bridge-"+method,
                "lineage_controls":summary['lineage_control_steps'],
                "charged_controls_lower_bound":summary['charged_control_steps_lower_bound'],
                "segment_optimizer_steps":summary['optimizer_steps'],
                "optimizer_steps_total_lower_bound":prior.get('retained_optimizer_steps',0)+summary['optimizer_steps'],
                "unknown_unflushed_ancestor_tail":bool(prior.get('unknown_unflushed_parent_tail',False))}
    monkeypatch.setattr(combined,'audit_method_endpoint',raw_audit)
    for method,folder in sources.items():
        write_json(folder/'endpoint_raw_audit.json',raw_audit(folder.parent,method))
    return root,sources


def test_manifest_combines_exact_sources_and_preserves_recovery_cost(fake_endpoints,tmp_path):
    root,sources=fake_endpoints;output=tmp_path/'combined.json'
    manifest=combined.build_manifest(sources,100000,output,root)
    assert combined.verify_manifest(output,root)['manifest_id']==manifest['manifest_id']
    assert manifest['sources']['H']['charged_controls_lower_bound']==20160
    b1=manifest['sources']['B1']
    assert b1['charged_controls_lower_bound']==29545
    assert b1['optimizer_steps_total_lower_bound']==126
    assert b1['unknown_unflushed_ancestor_tail'] is True
    assert b1['recovery_chain']==['original','recovery_a']
    assert b1['source_config_scope']=='recovery provenance'
    changed=deepcopy(json.loads(output.read_text()))
    changed['sources']['B1']['charged_controls_lower_bound']=20160
    write_json(output,changed)
    with pytest.raises(ValueError,match='content hash mismatch'):
        combined.verify_manifest(output,root)


def test_manifest_rejects_semantic_drift(fake_endpoints,tmp_path):
    root,sources=fake_endpoints
    protocol=sources['B1'].parent/'protocol.json';value=json.loads(protocol.read_text())
    value['routes']=[{"geometry_id":"different"}];write_json(protocol,value)
    with pytest.raises(ValueError,match='training/task semantics'):
        combined.build_manifest(sources,100000,tmp_path/'should_not_exist.json',root)
    assert not (tmp_path/'should_not_exist.json').exists()


def test_all_endpoint_claims_publish_in_one_atomic_ledger(fake_endpoints,tmp_path):
    root,sources=fake_endpoints
    manifest=combined.build_manifest(sources,100000,root=root)
    child=root/'runs/child';child.mkdir()
    receipt=combined._claim_endpoints_atomic(root/'runs',manifest,child)
    ledger=json.loads((root/'runs/native_endpoint_claims.json').read_text())
    assert set(receipt['claim_keys'])==set(combined.METHODS)
    assert len(ledger['claims'])==3
    assert {claim['child_batch'] for claim in ledger['claims'].values()}=={'child'}
    with pytest.raises(FileExistsError,match='already has'):
        combined._claim_endpoints_atomic(root/'runs',manifest,root/'runs/other')
    assert json.loads((root/'runs/native_endpoint_claims.json').read_text())==ledger
    # Content hashes are verified evidence, not a way to claim the same source
    # location a second time after in-place mutation.
    changed=deepcopy(manifest)
    changed['sources']['B1']['checkpoint_hashes']['model.zip']='changed'
    changed['manifest_id']='changed-manifest'
    with pytest.raises(FileExistsError,match='already has'):
        combined._claim_endpoints_atomic(root/'runs',changed,root/'runs/other')


def test_single_parent_prepare_honors_combined_claim_ledger(fake_endpoints,monkeypatch):
    root,sources=fake_endpoints
    manifest=combined.build_manifest(sources,100000,root=root)
    child=root/'runs/child';child.mkdir()
    combined._claim_endpoints_atomic(root/'runs',manifest,child)
    monkeypatch.setattr(continuation,'ROOT',root)
    with pytest.raises(FileExistsError,match='already claimed'):
        continuation.assert_parent_endpoints_unclaimed(root/'runs',sources['H'].parent)


def test_prepare_from_manifest_copies_each_checkpoint_and_claims_atomically(fake_endpoints,tmp_path,monkeypatch):
    root,sources=fake_endpoints;manifest_path=root/'runs/endpoints.json'
    manifest=combined.build_manifest(sources,100000,manifest_path,root)
    (root/'scripts').mkdir()
    (root/'navstack').mkdir()
    (root/'navstack/independent_clip_ppo.py').write_text('# optional learner loader fixture\n',encoding='utf-8')
    # Combined preparation refreshes only orchestration files; fixture content
    # is enough because this test verifies staging/registration, not execution.
    monkeypatch.setattr(combined,'ORCHESTRATION_FILES',())
    (root/'scripts/combine_native_endpoints.py').write_text('# frozen fixture\n',encoding='utf-8')
    (root/'scripts/audit_native_method_endpoint.py').write_text('# frozen fixture\n',encoding='utf-8')
    monkeypatch.setattr(combined,'resources_ok',lambda unused:(True,'ok'))
    child=combined.prepare_from_manifest(manifest_path,900,root)
    protocol=json.loads((child/'protocol.json').read_text())
    assert protocol['methods']==list(combined.METHODS)
    assert protocol['method_budgets']=={'H0':79840,'H':79840,'B1':79840}
    assert protocol['recovery']['B1']['consumed_controls_lower_bound']==29545
    assert protocol['recovery']['B1']['unknown_unflushed_parent_tail'] is True
    assert protocol['combined_endpoint_manifest_id']==manifest['manifest_id']
    assert 'parent_batch' not in protocol
    assert all((child/'resume'/method/'resume_state.pkl').is_file() for method in combined.METHODS)
    registered=json.loads((child/'registered.json').read_text())
    assert registered['status']=='REGISTERED_READY' and len(registered['claim_keys'])==3
    pointer=json.loads((root/'runs/native_continuation.json').read_text())
    assert Path(pointer['batch'])==child
