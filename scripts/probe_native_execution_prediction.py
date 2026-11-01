"""Exploratory TRAIN-only executability prediction;no plant or registered-policy updates."""
import gzip,hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from navstack.config import EnvConfig
from navstack.planning import analytic_connection
from navstack.env import candidate_goal_witness
from scripts.audit_native_policy_sequence import observation_env,advance_observation
from scripts.audit_native_method_endpoint import read,sha


def features(env,action):
    s=env.state;co,si=math.cos(s.yaw),math.sin(s.yaw)
    def local(x,y):return co*x+si*y,-si*x+co*y
    gx,gy=local(env.goal[0]-s.x,env.goal[1]-s.y);ga=env.goal[2]-s.yaw
    if action:
        target,speed=env.decode_action(action);dx,dy=local(target[0]-s.x,target[1]-s.y)
        angle=target[2]-s.yaw;clear=env.world.free(target,env.config.vehicle)
        witness=analytic_connection((s.x,s.y,s.yaw),target,env.world,env.config.vehicle) is not None if clear else False
        continuation=candidate_goal_witness(target,env.goal,env.world,env.config.vehicle) if clear else False
    else:
        target=(s.x,s.y,s.yaw);speed=0.;dx=dy=angle=0.;clear=True;witness=True;continuation=False
    geometry=[s.v/3,s.steering/.5,s.acceleration/2.5,env.plant.previous_command.acceleration/2.5,
        env.plant.previous_command.steering/.5,gx/40,gy/40,math.sin(ga),math.cos(ga),
        s.time/45,float(action==0),dx/15,dy/15,math.sin(angle),math.cos(angle),speed/3,
        math.dist(target[:2],env.goal[:2])/40,float(clear),float(witness),float(continuation)]
    neighbors=[]
    actors=sorted((o for o in env.world.dynamic if math.hypot(o.x-s.x,o.y-s.y)<=15),key=lambda o:(o.x-s.x)**2+(o.y-s.y)**2)[:3]
    for o in actors:
        x,y=local(o.x-s.x,o.y-s.y);vx,vy=local(o.vx-s.v*co,o.vy-s.v*si)
        neighbors.extend([x/15,y/15,vx/3,vy/3,1.])
    geometry+=neighbors+[0.]*(15-len(neighbors))
    return np.array(geometry,dtype=np.float32),np.array(env.execution_outcome,dtype=np.float32)


def dataset(batch,out,methods=('H','H0')):
    p=read(batch/'protocol.json');xs=[];feedback=[];ys=[];groups=[];refs=[];counts={}
    for name in ('env','config','planning','world','road_grid','road_grid_batch','vehicle','native_route','native_hazards','measured_demo'):
        name='navstack/'+name+'.py'
        if sha(ROOT/name)!=p['source_hashes'][name]:raise ValueError('Actual feature source differs from frozen source:'+name)
    for method in methods:
        folder=batch/method;audit=read(folder/'endpoint_raw_audit.json')
        if sha(batch/'protocol.json')!=audit['protocol_sha256']:raise ValueError('Protocol changed')
        if sha(folder/'config.json')!=audit['config_sha256']:raise ValueError('Config changed')
        bridge=batch/'infrastructure'/audit['bridge_response_stream']
        if sha(bridge)!=audit['bridge_response_sha256']:raise ValueError('Bridge changed')
        begins={}
        with gzip.open(bridge,'rt',encoding='utf-8') as stream:
            for line in stream:
                r=json.loads(line)['response']
                if 'reset_ticks' in r:begins[r['episode_id']]=r
        cfg=EnvConfig.from_dict(read(folder/'config.json')['env']);env=None;current=None;controls=0;decisions=0
        try:
            for name,digest in sorted(audit['transition_hashes'].items()):
                if sha(folder/name)!=digest:raise ValueError('TRAIN trace changed')
                with gzip.open(folder/name,'rt',encoding='utf-8') as stream:
                    for line in stream:
                        row=json.loads(line);episode=row['trace'][0]['backend_evidence']['episode_id']
                        if episode!=current:
                            if env is not None:env.close()
                            begin=begins[episode];scene=begin['scene']
                            initial={'initial':begin['state'],'goal':scene['goal'],'static':scene['static'],
                                'dynamic':begin['dynamic'],'road_sha256':scene['road_grid']['sha256'],
                                'actuator_evidence':{k:begin[k] for k in ('actuator_observation','speed_setpoint','actuation_state')}}
                            env=observation_env(cfg,scene,initial,begin=begin);env.world.road_collision_backend=cfg.road_collision_backend
                            current=episode
                        x,f=features(env,int(row['policy_action']))
                        assert not hasattr(env.plant,'step')
                        xs.append(x);feedback.append(f);ys.append(int(row['mode']=='plan_failed'))
                        groups.append(scene['native_layout']['geometry_id'])
                        refs.append((method,row['episode_id'],int(row['policy_action']),row['trace'][0]['backend_frame']))
                        wrapped={'policy_action':row['policy_action'],'info':{'mode':row['mode'],'trace':row['trace']}}
                        advance_observation(env,wrapped)
                        controls+=len(row['trace']);decisions+=1
                        if decisions%256==0:print(json.dumps({'method':method,'TRAIN_decisions':decisions,'new_controls':0}),flush=True)
            if controls!=audit['training_control_frames_verified']:raise ValueError('Incomplete TRAIN physical inventory')
            counts[method]={'TRAIN_controls':controls,'decisions':decisions,'raw_audit_sha256':sha(folder/'endpoint_raw_audit.json')}
        finally:
            if env is not None:env.close()
    x,f,y,g=map(np.asarray,(xs,feedback,ys,groups))
    np.savez_compressed(out/'TRAIN_selected_executions.npz',geometry=x,previous_execution_feedback=f,plan_failed=y,route_group=g,
        source_method=np.array([r[0] for r in refs]),episode=np.array([r[1] for r in refs]),selected_action=np.array([r[2] for r in refs]),first_frame=np.array([r[3] for r in refs]))
    return x,f,y,g,counts


def metrics(p,y):
    positive=y==1;negative=~positive
    auc=float(np.mean([(p[positive]>v).sum()+.5*(p[positive]==v).sum() for v in p[negative]])/positive.sum()) if positive.any() and negative.any() else None
    return {'n':len(y),'failures':int(y.sum()),'Brier':float(np.square(p-y).mean()),'AUC':auc,
        'log_loss':float(-(y*np.log(np.clip(p,1e-7,1))+(1-y)*np.log(np.clip(1-p,1e-7,1))).mean())}


def fit(x,y,train,test):
    mean=x[train].mean(0);std=np.maximum(x[train].std(0),.05)
    z=torch.tensor(np.clip((x-mean)/std,-10,10),dtype=torch.float32);target=torch.tensor(y,dtype=torch.float32)
    torch.manual_seed(0)
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.Tanh(),torch.nn.Linear(32,16),torch.nn.Tanh(),torch.nn.Linear(16,1))
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    for epoch in range(300):
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(z[train]).flatten(),target[train]);loss.backward();opt.step()
    with torch.no_grad():return torch.sigmoid(model(z[test]).flatten()).numpy()


def history_only_probabilities(feedback,y,train,test):
    previous=feedback[:,1]>=.5
    # Fit both transition probabilities on TRAIN only,with fixed Beta(1,1)
    # smoothing. No heldout label sets a prediction or threshold.
    probabilities=[]
    for value in (False,True):
        mask=train&(previous==value)
        probabilities.append((float(y[mask].sum())+1)/(int(mask.sum())+2))
    return np.asarray(probabilities)[previous[test].astype(int)]


def action_switches(method,episode,action):
    result=np.zeros(len(action),dtype=bool)
    if len(action)>1:
        result[1:]=(method[1:]==method[:-1])&(episode[1:]==episode[:-1])&(action[1:]!=action[:-1])
    return result


def validation_masks(seed,route,heldout):
    return np.isin(seed,[11,22,33])&(route!=heldout),np.isin(seed,[44,55])&(route==heldout)


def run_validation(output,pilot_cache=None):
    import shutil
    from navstack.research import resources_ok
    output=Path(output);output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    started=time.monotonic();shutil.copy2(__file__,output/'executed_probe.py')
    pairs=read(ROOT/'protocols/native_feedback_development_pairs_20261031.json')['pairs']
    if [r['seed'] for r in pairs]!=[11,22,33,44,55]:raise ValueError('Require every canonical trained policy seed')
    protocol={'scope':'Exploratory auxiliary prediction validation on existing selected TRAINdecisions;not a new PPOmethod or driving result',
        'train_policy_seeds':[11,22,33],'validation_policy_seeds':[44,55],
        'group_split':'For each heldout route,fit only other routes from train seeds;test only that route from validation seeds. No validation seed enters fitting or input normalization.',
        'capacity_matching':'Both MLPs have41inputs,identical seeded parameter initialization,32tanh16tanh1architecture,300full-batch BCE Adam1e-3updates. Geometry arm zeros the6history inputs.',
        'strong_history_baseline':'TRAIN-fitted Beta(1,1)-smoothed P(next plan_failed|previous plan_failed)',
        'predeclared_promising_rule':'Mean route Brier improves at least10percent against BOTH geometry-only MLP and history-only baseline;at least3of4route folds improve against geometry;action-switch subgroup Brier improves against geometry.',
        'limits':'Four known route geometries and previously collected TRAINdata. No final conditions,unselected-action labels,policy updates,or closed-loop claims. Prior seed11pilot was exploratory and had unmatched input-layer parameter counts.',
        'script_sha256':sha(__file__),'wall_budget_seconds':600,'CPU_threads':1,'new_controls':0}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    arrays=[];sources=[]
    try:
        for pair in pairs:
            inventory={}
            for method,key in (('H','h'),('H0','h0')):inventory.setdefault(pair[key],[]).append(method)
            for name,methods in inventory.items():
                okay,reason=resources_ok(ROOT)
                if not okay:raise RuntimeError(str(reason))
                if time.monotonic()-started>600:raise TimeoutError('Finite auxiliary probe budget exceeded')
                batch=ROOT/name
                if read(batch/'protocol.json')['seed']!=pair['seed']:raise ValueError('Canonical source seed changed')
                destination=output/('seed'+str(pair['seed'])+'_'+'_'.join(methods));destination.mkdir()
                if pair['seed']==11 and pilot_cache is not None:
                    cache=Path(pilot_cache);report=read(cache/'result.json')
                    path=cache/'TRAIN_selected_executions.npz'
                    if sha(path)!=report['TRAIN_dataset_sha256']:raise ValueError('Pilot TRAINdataset changed')
                    if report['input_sources']!={m:{'TRAIN_controls':read(batch/m/'endpoint_raw_audit.json')['training_control_frames_verified'],
                            'decisions':2080,'raw_audit_sha256':sha(batch/m/'endpoint_raw_audit.json')} for m in methods}:
                        raise ValueError('Cached TRAINdata no longer binds the actual raw audits')
                    shutil.copy2(path,destination/path.name);counts=report['input_sources']
                else:
                    _,_,_,_,counts=dataset(batch,destination,methods)
                path=destination/'TRAIN_selected_executions.npz'
                with np.load(path,allow_pickle=False) as data:
                    x=data['geometry'];f=data['previous_execution_feedback'];y=data['plan_failed'];g=data['route_group']
                    switches=action_switches(data['source_method'],data['episode'],data['selected_action'])
                if x.shape!=(len(y),35) or f.shape!=(len(y),6):raise ValueError('Unexpected feature contract')
                arrays.append((x,f,y,g,np.full(len(y),pair['seed']),switches))
                sources.append({'seed':pair['seed'],'batch':batch.name,'methods':methods,'sources':counts,
                    'dataset_sha256':sha(path),'relative_dataset':path.relative_to(output).as_posix()})
        x,f,y,g,seeds,switches=[np.concatenate([a[i] for a in arrays]) for i in range(6)]
        geometry=np.c_[x,np.zeros_like(f)];full=np.c_[x,f];folds=[]
        routes=np.unique(g)
        if len(routes)!=4:raise ValueError('Require all four route groups')
        for index,group in enumerate(routes):
            train,test=validation_masks(seeds,g,group)
            if not train.any() or not test.any() or np.any(train&test):raise ValueError('Invalid crossed seed/route partition')
            a=fit(geometry,y,train,test);b=fit(full,y,train,test);h=history_only_probabilities(f,y,train,test)
            switched=switches[test]
            row={'heldout_route':str(group),'train_decisions':int(train.sum()),'validation_decisions':int(test.sum()),
                'geometry':metrics(a,y[test]),'geometry_and_previous_execution':metrics(b,y[test]),
                'history_only':metrics(h,y[test]),'action_switches':int(switched.sum()),
                'switch_geometry':metrics(a[switched],y[test][switched]),
                'switch_geometry_and_previous_execution':metrics(b[switched],y[test][switched])}
            np.savez_compressed(output/('validation_predictions_fold'+str(index)+'.npz'),row_indices=np.flatnonzero(test),
                plan_failed=y[test],geometry=a,geometry_and_previous_execution=b,history_only=h,action_switched=switched)
            folds.append(row);print(json.dumps(row),flush=True)
        means={name:float(np.mean([r[name]['Brier'] for r in folds])) for name in
            ('geometry','geometry_and_previous_execution','history_only','switch_geometry','switch_geometry_and_previous_execution')}
        improved=sum(r['geometry_and_previous_execution']['Brier']<r['geometry']['Brier'] for r in folds)
        promising=(means['geometry_and_previous_execution']<=.9*min(means['geometry'],means['history_only'])
            and improved>=3 and means['switch_geometry_and_previous_execution']<means['switch_geometry'])
        result={'scope':protocol['scope'],'protocol_sha256':sha(output/'protocol.json'),'sources':sources,
            'all_TRAIN_decisions':len(y),'all_TRAIN_controls':sum(r['TRAIN_controls'] for s in sources for r in s['sources'].values()),
            'train_policy_seeds':[11,22,33],'validation_policy_seeds':[44,55],'folds':folds,'mean_route_Brier':means,
            'improved_route_folds':improved,'predeclared_promising_rule_passed':bool(promising),
            'applied_to_registered_policies':False,'diagnostic_predictor_Adam_steps':2400,
            'new_controls':0,'registered_policy_updates':0,'wall_seconds':time.monotonic()-started,'research_acceptance':'UNMET'}
        (output/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in result.items() if k not in ('folds','sources')}),flush=True)
        return result
    except BaseException as error:
        (output/'failure.json').write_text(json.dumps({'error':repr(error),'wall_seconds':time.monotonic()-started,'new_controls':0}),encoding='utf-8');raise


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    parser.add_argument('--pilot-cache');args=parser.parse_args();run_validation(args.output,args.pilot_cache)


