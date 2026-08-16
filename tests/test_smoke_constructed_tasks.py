from __future__ import annotations

from tools.smoke_constructed_tasks import _negative_checks

from .helpers import item_id, site_data, task


def test_decoy_excludes_every_acceptable_option_for_current_arm() -> None:
    graft_task = task("graft")
    record = {
        "arm": "a_only",
        "actions": [],
        "final_state": {
            "selected_entity_ids": [item_id("graft", 0)],
            "fields": {},
            "submission_completed": True,
        },
    }

    checks = _negative_checks(graft_task, record, site_data("graft")["entities"])
    decoy_check = next(check for check in checks if check["mutation"] == "decoy_only")

    assert decoy_check["evidence_status"] == "FAIL"
    assert decoy_check["e2e_success"] is False
