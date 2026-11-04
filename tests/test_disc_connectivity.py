from navstack.disc_connectivity import necessary_connectivity
from navstack.native_route import sample_road_grid,REFINED_GRID_VERSION
from navstack.road_grid import RoadGrid
from navstack.config import VehicleConfig


def test_full_width_barrier_disconnects_even_the_conservative_center_space():
    road=RoadGrid(sample_road_grid(dict(width=32.,height=32.,geometry_id='fixture'),lambda x,y:True,REFINED_GRID_VERSION))
    result=necessary_connectivity(road,[(15.,0.,17.,32.)],(6.,16.),(26.,16.),VehicleConfig(),.8)
    assert result['model_infeasibility_certificate'] and not result['necessary_connected']
    assert result['start_component']!=0


def test_open_bypass_and_relaxed_goal_region_cannot_be_declared_infeasible():
    road=RoadGrid(sample_road_grid(dict(width=32.,height=32.,geometry_id='fixture'),lambda x,y:True,REFINED_GRID_VERSION))
    assert necessary_connectivity(road,[(15.,8.,17.,24.)],(6.,16.),(26.,16.),VehicleConfig(),.8)['necessary_connected']
    assert necessary_connectivity(road,[(15.,0.,17.,32.)],(6.,16.),(20.,16.),VehicleConfig(),8.)['necessary_connected']


def test_tiny_inscribed_body_preserves_all_uncertain_cells_instead_of_claiming_too_much():
    road=RoadGrid(sample_road_grid(dict(width=32.,height=32.,geometry_id='fixture'),lambda x,y:True))
    vehicle=VehicleConfig(front=.1,rear=.1,width=.1)
    assert necessary_connectivity(road,[(15.,0.,17.,32.)],(6.,16.),(26.,16.),vehicle,.8)['necessary_connected']
