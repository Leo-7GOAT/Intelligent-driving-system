from scripts.prepare_native_corrective_bc import corrective_input_cost


def test_first_corrective_fit_charges_teacher_and_two_actual_B1_sources():
    protocol={'extra_supervision':{'teacher_controls':4329},'recovery':{'B1':{'consumed_controls_lower_bound':4329}}}
    cost,batches=corrective_input_cost(protocol,{'sources':[{'batch':'teacher'}]}, {'phase4':7123,'ordered':5265})
    assert cost==16717 and batches==['ordered','phase4','teacher']


def test_later_correction_retains_ancestor_cost_and_does_not_recharge_old_source():
    protocol={'extra_supervision':{'teacher_controls':4329},'recovery':{'B1':{'consumed_controls_lower_bound':16717}},
        'corrective_initialization':{'parent':'phase4','oracle_source':'ordered'}}
    cost,batches=corrective_input_cost(protocol,{'sources':[{'batch':'teacher'}]}, {'new_parent':7000,'ordered':5265})
    assert cost==23717 and len(batches)==4
