"""Explicit geometry-separated native training/validation,never final testing."""
import math
from .native_route import local_layout

VERSION='native_training_validation_v1'


def validate_native_partitions(protocol):
    from .native_cases import registered_native_cases
    validation=protocol.get('native_validation')
    if (not isinstance(validation,dict) or set(validation)!={'version','routes','manifest','minimum_route_separation_m'}
            or validation['version']!=VERSION or not protocol.get('native_case_manifest')
            or protocol.get('native_hazard') is not None
            or not isinstance(validation['routes'],list) or not validation['routes']):
        raise ValueError('Require explicit native training and validation inventories')
    gap=validation['minimum_route_separation_m']
    if isinstance(gap,bool) or not isinstance(gap,(int,float)) or not math.isfinite(gap) or gap<8:
        raise ValueError('Validation roads require at least8m sampled centreline separation')
    training={key:protocol[key] for key in ('routes','native_case_manifest')}
    registered_native_cases(training)
    registered_native_cases({'routes':validation['routes'],'native_case_manifest':validation['manifest']})
    originals=protocol['routes'];heldout=validation['routes']
    # Validate geometry hashes before computing any separation assertion.
    for route in originals+heldout:
        local_layout(route,route['map'],route['opendrive_sha256'])
        if any(math.hypot(a['x']-b['x'],a['y']-b['y'])>1.25 for a,b in zip(route['points'],route['points'][1:])):
            raise ValueError('Partition separation requires densely sampled native routes')
    for a in originals:
        for b in heldout:
            if (a['map'],a['opendrive_sha256'])!=(b['map'],b['opendrive_sha256']):
                raise ValueError('Native validation must use the same registered map')
            if a['geometry_id']==b['geometry_id']:
                raise ValueError('Native validation repeats a training geometry')
            # Survey spacing is at most1.25m. Requiring gap+1.25 between samples
            # conservatively covers the two adjacent half segments as well.
            if any(math.hypot(p['x']-q['x'],p['y']-q['y'])<gap+1.25
                   for p in a['points'] for q in b['points']):
                raise ValueError('Native validation route overlaps or approaches training roads')
    return {'version':VERSION,'training_conditions':len(registered_native_cases(training)),
            'validation_conditions':len(registered_native_cases({'routes':heldout,'native_case_manifest':validation['manifest']})),
            'minimum_sampled_route_separation_m':gap+1.25,
            'scope':'Distinct development validation roads,not final confirmation or proof of unvisited historical roads'}
