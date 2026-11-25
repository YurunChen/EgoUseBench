from copy import deepcopy

import pytest

from egocua.final_state import evaluate_final_state, goal_entity_ids, goal_signature


@pytest.fixture
def goal():
    return {
        'acceptable_states': [{'selected_entity_ids': ['six', 'press'], 'fields': {}}],
        'unit_equivalences': {'silver_spoon': {'single': 1, 'two': 2, 'four': 4, 'six': 6}},
    }


@pytest.mark.parametrize('cart', [
    ['six', 'press'], ['two', 'two', 'two', 'press'],
    ['four', 'two', 'press'], ['four', 'single', 'single', 'press'],
    ['single'] * 6 + ['press'],
])
def test_reviewed_packages_can_be_mixed(goal, cart):
    assert evaluate_final_state(goal, {'selected_entity_ids': cart, 'fields': {}})


@pytest.mark.parametrize('cart', [
    ['four', 'single', 'press'], ['six', 'single', 'press'],
    ['six'], ['six', 'press', 'extra'], ['six', 'other_press'],
])
def test_exact_quantities_and_unmapped_items_are_preserved(goal, cart):
    assert not evaluate_final_state(goal, {'selected_entity_ids': cart, 'fields': {}})


def test_fields_remain_exact(goal):
    assert not evaluate_final_state(goal, {'selected_entity_ids': ['six', 'press'], 'fields': {'extra': 'yes'}})


def test_legal_ids_exclude_packages_that_cannot_fit():
    goal = {'acceptable_states': [{'selected_entity_ids': ['six'], 'fields': {}}],
            'unit_equivalences': {'spoon': {'four': 4, 'six': 6, 'eight': 8}}}
    assert goal_entity_ids(goal) == {'six'}


def test_single_object_capacity_is_not_accumulated():
    goal = {'acceptable_states': [{'selected_entity_ids': ['large_pan'], 'fields': {}}]}
    assert not evaluate_final_state(goal, {'selected_entity_ids': ['small_pan', 'small_pan'], 'fields': {}})


def test_same_witness_with_different_package_choices_is_a_different_goal(goal):
    narrowed = deepcopy(goal)
    del narrowed['unit_equivalences']['silver_spoon']['single']
    assert goal_signature(goal) != goal_signature(narrowed)


@pytest.mark.parametrize('quantity', [0, -1, True, 1.5, '2'])
def test_bad_package_counts_fail_early(goal, quantity):
    goal['unit_equivalences']['silver_spoon']['two'] = quantity
    with pytest.raises(ValueError, match='positive integers'):
        goal_entity_ids(goal)


def test_one_package_cannot_claim_two_unit_classes(goal):
    goal['unit_equivalences']['other'] = {'single': 1}
    with pytest.raises(ValueError, match='exactly one'):
        goal_entity_ids(goal)


def test_no_mapping_keeps_exact_original_sku_behavior():
    goal = {'acceptable_states': [{'selected_entity_ids': ['a', 'a', 'b'], 'fields': {}}]}
    assert evaluate_final_state(goal, {'selected_entity_ids': ['b', 'a', 'a'], 'fields': {}})
    assert not evaluate_final_state(goal, {'selected_entity_ids': ['a', 'b'], 'fields': {}})


def test_compiler_projects_reviewed_units_to_both_phases(goal):
    from egocua.construction import compile_quantity_task

    source = {
        'instructions': {'local': {'a': 'Initial', 'b': 'More', 'bundled': 'Initial More'}},
        'initial_goal': {'acceptable_states': [{'selected_entity_ids': ['two', 'press'], 'fields': {}}]},
        'final_goals': {'local': {'acceptable_states': goal['acceptable_states']}},
        'unit_equivalences': {**goal['unit_equivalences'], 'unused': {'unused': 1}},
    }
    original = {'graft': {}, 'visual_packets': [{'id': 'V0'}], 'source': {'id': 9}}
    compiled = compile_quantity_task(original, source, 'local')
    assert compiled['visual_packets'] == original['visual_packets']
    assert compiled['source'] == original['source']
    assert compiled['goal']['unit_equivalences'] == goal['unit_equivalences']
    assert compiled['graft']['initial_goal']['unit_equivalences'] == goal['unit_equivalences']
    assert original['graft'] == {}


def test_full_goal_survives_checker_and_attribution(goal):
    from egocua.checker import EVALUATOR_VERSION, check_run
    from egocua.failure_attribution import attribute_run
    from egocua.truth import Truth
    from tests.conditions import CONDITIONS
    from tests.helpers import complete_run_record, task

    definition = task('graft')
    definition['goal'] = deepcopy(goal)
    definition['graft']['initial_goal'] = {
        'acceptable_states': [{'selected_entity_ids': ['two', 'press'], 'fields': {}}],
        'unit_equivalences': deepcopy(goal['unit_equivalences']),
    }
    definition['graft']['transition'] = 'replace'
    state = {'selected_entity_ids': ['four', 'two', 'press'], 'fields': {}}
    record = complete_run_record({
        'task_id': definition['id'], 'track': 'graft',
        'condition': CONDITIONS['upfront'].as_dict(), 'task_snapshot': definition,
        'meta': {'run_id': 'quantity-fixture'}, 'steps': 1,
        'actions': [], 'live_state': state,
    })
    original = deepcopy(record)
    checked = check_run(definition, record)
    assert checked['e2e_success']
    assert checked['evaluator']['version'] == EVALUATOR_VERSION == '3.3'
    assert checked['evaluator']['goal'] == goal
    row = attribute_run(checked, record)
    assert row['scoring_goal'] == goal
    assert record == original
    assert Truth.of(definition).final_referent_ids == goal_entity_ids(goal)
    checked['evaluator']['goal']['unit_equivalences'].clear()
    assert row['scoring_goal'] == goal


def test_acceptance_replays_every_feasible_package_and_no_infeasible_one(goal):
    from tools.audit.audit_task_acceptance import _positive_states

    goal['unit_equivalences']['silver_spoon']['too_many'] = 8
    states = _positive_states(goal)
    assert all(evaluate_final_state(goal, state) for state in states)
    assert {sku for state in states for sku in state['selected_entity_ids']} == goal_entity_ids(goal)
    assert any('single' in state['selected_entity_ids'] for state in states)
    assert any('four' in state['selected_entity_ids'] and state['selected_entity_ids'].count('single') == 2 for state in states)


@pytest.mark.parametrize('phase', ['initial', 'final'])
def test_public_scope_must_include_every_feasible_alias(goal, phase):
    from egocua.schema import ValidationError, _validate_truth_block

    initial = deepcopy(goal)
    final = deepcopy(goal)
    if phase == 'initial':
        final = {'acceptable_states': [{'selected_entity_ids': ['press'], 'fields': {}}]}
    definition = {
        'id': 'scope-quantity-fixture', 'track': 'graft', 'goal': final,
        'graft': {'initial_goal': initial},
        'truth': {'site_scope_entity_ids': ['six', 'press']},
    }
    with pytest.raises(ValidationError, match='every task-referenced entity'):
        _validate_truth_block(definition, {'single', 'two', 'four', 'six', 'press'})


@pytest.mark.parametrize('version', ['3.0', '3.1', '3.2', '3.3'])
def test_historical_final_state_protocols_remain_supported(version):
    from egocua.failure_attribution import _require_final_state_checker

    _require_final_state_checker(version)
