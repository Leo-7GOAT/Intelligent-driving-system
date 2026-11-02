"""Explicit repeated TRAIN exposures; unique scene inventory is unchanged."""
VERSION='native_basic_rehearsal_quarter_v1'
BASIC=(0,3,6,9)
EXTRA_CURVES=(12,15,18,21)


def rehearsal_schedule(cases):
    if len(cases)!=48 or any(c.get('condition')!=i for i,c in enumerate(cases)):
        raise ValueError('Rehearsal requires the registered unique TRAIN48 inventory')
    if any(cases[i].get('native_hazard') is not None for i in (*BASIC,*EXTRA_CURVES)):
        raise ValueError('Basic and extra curve exposures must actually be obstacle-free')
    remaining=[i for i in range(48) if i not in BASIC]+list(EXTRA_CURVES)
    order=[]
    for group in range(16):order.extend([BASIC[group%4],*remaining[3*group:3*group+3]])
    return dict(version=VERSION,indices=order,unique_conditions=48,exposures_per_cycle=64,
        repeated_exposures_are_not_independent_conditions=True)


def scheduled_native_cases(cases,schedule):
    expected=rehearsal_schedule(cases)
    if schedule!=expected:raise ValueError('TRAIN rehearsal schedule differs from its fixed definition')
    return [cases[i] for i in expected['indices']]


def install_schedule_dispatch(text):
    marker='            cases = registered_native_cases(protocol)\n'
    if text.count(marker)!=1 or "protocol.get('training_case_schedule')" in text:
        raise ValueError('Require the original one-time native training case dispatch')
    return text.replace(marker,marker+
        "            if protocol.get('training_case_schedule') and not protocol.get('evaluation_only'):\n"
        "                from navstack.native_case_schedule import scheduled_native_cases\n"
        "                cases = scheduled_native_cases(cases,protocol['training_case_schedule'])\n")
