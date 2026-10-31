import gzip,json
import pytest
from scripts.analyze_native_control import bind_controller_frames

def gz(path,rows):
    with gzip.open(path,'wt',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row)+'\n')

@pytest.mark.parametrize('fault',[None,'measurement','requested_command','reset_geometry','extra_step'])
def test_controller_raw_bindings_reject_nonmeasured_data_and_unaccounted_controls(tmp_path,fault):
    infrastructure=tmp_path/'infrastructure';infrastructure.mkdir();requests=[];responses=[];episodes=[];routes=[]
    for i in range(4):
        folder=tmp_path/f'case_{i:02d}';folder.mkdir();scene={'native_layout':{'geometry_id':f'g{i}'}}
        routes.append({'geometry_id':f'g{i}'});episode=f'episode{i}'
        begin={'op':'begin','episode_id':episode,'request_id':1};requests.append({'request':begin})
        responses.append({'response':{'episode_id':episode,'request_id':1,'scene':scene}})
        measurement={'episode_id':episode,'request_id':2,'frame':10+i,'state':{'x':1}}
        command={'acceleration':0.,'steering':0.}
        requests.append({'request':{'op':'step','episode_id':episode,'request_id':2,'expected_frame':9+i,**command}})
        responses.append({'response':measurement})
        trace={'measurement':measurement.copy(),'requested_command':command.copy()}
        if i==0 and fault=='measurement':trace['measurement']['state']={'x':99}
        if i==0 and fault=='requested_command':trace['requested_command']['acceleration']=1.
        if i==0 and fault=='reset_geometry':scene={'native_layout':{'geometry_id':'wrong'}}
        (folder/'scene.json').write_text(json.dumps(scene),encoding='utf-8')
        gz(folder/'transitions.jsonl.gz',[trace]);episodes.append({'case_id':i,'control_ticks':1})
    if fault=='extra_step':requests.append({'request':{'op':'step','episode_id':'extra','request_id':4}})
    gz(infrastructure/'bridge_fixture_requests.jsonl.gz',requests);gz(infrastructure/'bridge_fixture_responses.jsonl.gz',responses)
    if fault:
        with pytest.raises(ValueError):bind_controller_frames(tmp_path,{'routes':routes},episodes)
    else:assert bind_controller_frames(tmp_path,{'routes':routes},episodes)['actual_control_frames_bound']==4
