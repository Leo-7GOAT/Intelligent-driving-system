"""Explicit paired native development conditions; no policy or actor simulation."""
from copy import deepcopy

from .native_hazards import resolve_native_hazard, verify_native_hazard_begin
from .native_route import local_layout


VERSION = 'native_development_cases_v1'


def registered_native_cases(protocol):
    routes = protocol['routes']
    manifest = protocol.get('native_case_manifest')
    if manifest is None:
        cases = [{'scenario': 'native_curve', 'condition': i, 'native_route': route}
                 for i, route in enumerate(routes)]
        if protocol.get('native_hazard') is not None:
            spec = protocol['native_hazard']
            for case in cases:
                route = case['native_route']
                resolve_native_hazard(local_layout(route, route['map'], route['opendrive_sha256']), spec)
                case['native_hazard'] = deepcopy(spec)
        return cases
    if (not isinstance(manifest, dict) or set(manifest) != {'version', 'split', 'cases'}
            or manifest['version'] != VERSION or manifest['split'] != 'development'
            or protocol.get('native_hazard') is not None
            or not isinstance(manifest['cases'], list) or not 1 <= len(manifest['cases']) <= 64):
        raise ValueError('Invalid explicit native development manifest or ambiguous global hazard')
    cases, identities, used_routes = [], set(), set()
    for index, item in enumerate(manifest['cases']):
        if (not isinstance(item, dict) or set(item) != {'route_index', 'hazard'}
                or type(item['route_index']) is not int or not 0 <= item['route_index'] < len(routes)):
            raise ValueError('Invalid registered native route/case specification')
        route_index = item['route_index']; route = routes[route_index]
        layout = local_layout(route, route['map'], route['opendrive_sha256'])
        spec = item['hazard']
        resolved = resolve_native_hazard(layout, spec) if spec is not None else None
        actor_identity = None if resolved is None else (
            resolved['kind'], float(resolved['distance_m']), float(resolved['lateral_m']),
            float(resolved['walker_speed_m_s']), float(resolved['start_delay_s']))
        identity = (route['map'], route['opendrive_sha256'], route['geometry_id'], actor_identity)
        if identity in identities:
            raise ValueError('Duplicate physical native development condition')
        identities.add(identity); used_routes.add(route_index)
        case = {'scenario': 'native_curve', 'condition': index, 'native_route': deepcopy(route)}
        if spec is not None: case['native_hazard'] = deepcopy(spec)
        cases.append(case)
    if used_routes != set(range(len(routes))):
        raise ValueError('Explicit development manifest omits an original road')
    return cases


def verify_registered_native_begin(begin, protocol, expected_case_id=None):
    """Bind actual reset condition, route geometry and measured actor inventory."""
    if protocol.get('native_case_manifest') is None:
        verify_native_hazard_begin(begin, protocol.get('native_hazard'))
        return
    cases = registered_native_cases(protocol)
    scene = begin.get('scene', {}); case_id = scene.get('condition')
    if (type(case_id) is not int or not 0 <= case_id < len(cases)
            or expected_case_id is not None and case_id != expected_case_id):
        raise ValueError('Measured reset condition differs from registered native case')
    case = cases[case_id]
    if scene.get('native_layout', {}).get('geometry_id') != case['native_route']['geometry_id']:
        raise ValueError('Measured reset route differs from registered native case')
    verify_native_hazard_begin(begin, case.get('native_hazard'))
    if 'native_hazard' not in case and (scene.get('static') != [] or begin.get('dynamic') != []
                                      or begin.get('native_dynamic_actors', []) != []):
        raise ValueError('Obstacle-free registered case contains measured actors')
