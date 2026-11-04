"""Evaluation-only admission of the shared-lowlevel strong rule comparator."""
VERSION='shared_native_lowlevel_rule_v1'


def registered(protocol):
    value=protocol.get('fixed_shared_rule',{})
    return (isinstance(value,dict) and value.get('version')==VERSION
        and value.get('goal_wait_requires_heading') is True
        and type(value.get('failed_action_penalty_m')) is float and value['failed_action_penalty_m']==5.
        and protocol.get('methods')==['B3'] and protocol.get('evaluation_only') is True
        and all(type(protocol.get(k)) is int for k in ('controlled_budget_per_method','training_physics_steps','optimizer_updates'))
        and protocol.get('controlled_budget_per_method')==0 and protocol.get('method_budgets')=={'B3':0}
        and type(protocol.get('method_budgets',{}).get('B3')) is int
        and protocol.get('training_physics_steps')==0 and protocol.get('optimizer_updates')==0)
