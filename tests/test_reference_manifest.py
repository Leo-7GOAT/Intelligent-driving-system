import hashlib,json
import pytest
from navstack.reference_manifest import load_fixed_reference

def fixture(folder):
    rows=[]
    for i in range(4):
        p={'case_id':i,'geometry_id':f'geometry{i}','road_sha256':f'road{i}','goal':[2,0,0],
           'reference':{'x':[0,1,2],'y':[0,0,0],'yaw':[0,0,0],'s':[0,1,2],'kappa':[0,0,0],'v':[0,1,0],'a':[1,0,-1],'t':[0,1,2],'start_time':0}}
        path=folder/f'case_{i:02d}.json';path.write_text(json.dumps(p),encoding='utf-8')
        rows.append({'case_id':i,'file':path.name,'geometry_id':p['geometry_id'],'road_sha256':p['road_sha256'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    (folder/'manifest.json').write_text(json.dumps({'references':rows}),encoding='utf-8')

def test_fixed_reference_is_exact_bound_to_scene_and_read_only(tmp_path):
    fixture(tmp_path);ref,payload,record=load_fixed_reference(tmp_path,0,'geometry0','road0',[2,0,0])
    assert list(ref.v)==[0,1,0] and ref.start_time==0
    with pytest.raises(ValueError):ref.x[0]=4
    with pytest.raises(ValueError):load_fixed_reference(tmp_path,0,'geometry0','road1',[2,0,0])
    with pytest.raises(ValueError):load_fixed_reference(tmp_path,0,'geometry0','road0',[1,0,0])

@pytest.mark.parametrize('fault',['changed_bytes','escape','duplicate','nonfinite','moving_terminal','clock'])
def test_invalid_reference_or_manifest_rejects_before_any_plant(tmp_path,fault):
    fixture(tmp_path);path=tmp_path/'case_00.json';manifest=json.loads((tmp_path/'manifest.json').read_text());p=json.loads(path.read_text())
    if fault=='escape':manifest['references'][0]['file']='../case_00.json'
    elif fault=='duplicate':manifest['references'][1]['case_id']=0
    elif fault=='nonfinite':p['reference']['x'][0]=float('nan')
    elif fault=='moving_terminal':p['reference']['v'][-1]=.5
    elif fault=='clock':p['reference']['start_time']=4
    else:p['reference']['x'][0]=.1
    path.write_text(json.dumps(p),encoding='utf-8')
    if fault!='changed_bytes':manifest['references'][0]['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    with pytest.raises(ValueError):load_fixed_reference(tmp_path,0,'geometry0','road0',[2,0,0])
