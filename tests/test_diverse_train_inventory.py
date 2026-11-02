from copy import deepcopy
import pytest
from scripts.prepare_diverse_native_train_inventory import select


def pool():
    rows=[]
    for i,heading in enumerate((-15,15,-30,30,-45,45,-60,60,-75,75,-20,20,-40,40,-65,65)):
        rows.append(dict(geometry_id=str(i),heading_change_degrees=heading,
            points=[dict(x=50.*i,y=0.),dict(x=50.*i+20,y=0.)]))
    return dict(candidates=rows,map='measured_map',opendrive_sha256='original_map')


def test_geometry_selection_is_deterministic_separated_and_not_prefix_augmentation():
    p=pool();excluded=[deepcopy(p['candidates'][0])]
    selected=select(p,excluded)
    p['candidates'].reverse()
    assert select(p,excluded)==selected and len(selected)==12
    assert len({r['geometry_id'] for r in selected})==12
    assert excluded[0]['geometry_id'] not in {r['geometry_id'] for r in selected}
    assert all(r['map']=='measured_map' for r in selected)


def test_insufficient_diversity_does_not_relax_distance_or_duplicate_a_road():
    p=pool();p['candidates']=p['candidates'][:4]
    with pytest.raises(ValueError):select(p,[])


def test_sampling_balances_base_geometry_and_family_before_episode_count():
    from scripts.fit_diverse_native_actor import balanced_masses
    cells={0:(0,'curve'),1:(0,'curve'),2:(0,'walker'),3:(1,'curve')}
    mass=balanced_masses(cells,[0,1,2,3])
    assert sum(mass[g] for g in [0,1,2])==.5 and mass[3]==.5
    assert mass[0]==mass[1]==.125 and mass[2]==.25
    with pytest.raises(ValueError):balanced_masses(cells,[0,4])
