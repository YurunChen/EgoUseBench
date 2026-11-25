from tools.render.render_all_site_audits import _render_ids


TASKS = [
    {"id": "mart-a", "site": "egomart"},
    {"id": "mart-b", "site": "egomart"},
    {"id": "media-a", "site": "egomedia"},
]


def test_full_render_selects_every_task() -> None:
    assert _render_ids(TASKS, requested_ids=set(), requested_sites=set()) == {
        "mart-a",
        "mart-b",
        "media-a",
    }


def test_incremental_render_does_not_pull_in_stale_unrequested_tasks() -> None:
    assert _render_ids(
        TASKS,
        requested_ids={"media-a"},
        requested_sites={"egomart"},
    ) == {"mart-a", "mart-b", "media-a"}


def test_site_render_selects_only_that_site() -> None:
    assert _render_ids(
        TASKS,
        requested_ids=set(),
        requested_sites={"egomart"},
    ) == {"mart-a", "mart-b"}
