import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from egocua.review_session import ReviewSession
from egocua.review_web import create_review_app
from egocua.review_workspace import ReviewWorkspace
from egocua.synthesis_workflow import SynthesisWorkflow
from tests.helpers import materialize_assets, products, task
from tests.helpers import unchanged_task_owners as unchanged_task_owners

pytestmark = pytest.mark.usefixtures("unchanged_task_owners")


@pytest.fixture
def session(tmp_path, monkeypatch):
    definitions = [task("graft"), task("refresh")]
    for definition in definitions:
        materialize_assets(definition, products(definition["track"]), tmp_path)
    authored = tmp_path / "data/instructions/authored.json"
    authored.parent.mkdir(parents=True, exist_ok=True)
    authored.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "instructions": {d["id"]: d["instructions"] for d in definitions},
            }
        )
    )
    workspace = ReviewWorkspace.open(tmp_path, "reader", include_all_tasks=True)
    original = SynthesisWorkflow.status
    calls = []

    def counted(self, task_ids=None, **kwargs):
        calls.append(set(task_ids))
        return original(self, task_ids, **kwargs)

    monkeypatch.setattr(SynthesisWorkflow, "status", counted)
    view = ReviewSession(workspace)
    view.current()
    assert calls == [{t["id"] for t in definitions}]
    calls.clear()
    return view, definitions, calls


def test_browsing_reuses_the_same_projection(session):
    view, _, calls = session
    _, first = view.current()
    _, second = view.current()
    assert first is second
    assert calls == []


def test_submitting_refreshes_only_that_task(session):
    view, definitions, calls = session
    workspaces, lifecycle = view.current()
    tid = definitions[0]["id"]
    case = workspaces["all"].view(tid, lifecycle=lifecycle)
    workspaces["all"].submit(
        tid,
        {"decision": "admit", "note": ""},
        expected_revision=case["revision"],
        lifecycle=lifecycle,
    )
    _, updated = view.current()
    assert calls == [{tid}]
    other = definitions[1]["id"]
    assert updated["tasks"][other] == lifecycle["tasks"][other]
    assert updated["summary"]["tasks"] == 2
    view.current()
    assert calls == [{tid}]


def test_changed_frame_refreshes_only_its_owner(session):
    view, definitions, calls = session
    path = view._all.root / definitions[0]["visual_packets"][0]["frame_paths"][0]
    path.write_bytes(path.read_bytes() + b"\n")
    view.current()
    assert calls == [{definitions[0]["id"]}]


def test_changed_task_refreshes_only_its_owner(session):
    view, definitions, calls = session
    definition = definitions[0]
    path = (
        view._all.root / "data/tasks" / definition["track"] / f"{definition['id']}.json"
    )
    definition["instructions"]["a"] += " Please."
    path.write_text(json.dumps(definition))
    workspaces, _ = view.current()
    assert calls == [{definition["id"]}]
    assert (
        workspaces["all"].tasks[definition["id"]]["instructions"]
        == definition["instructions"]
    )


def test_concurrent_reads_refresh_changed_inputs_once(session):
    view, definitions, calls = session
    path = view._all.root / "artifacts/reports/task-acceptance.json"
    document = json.loads(path.read_text())
    document["tasks"][definitions[0]["id"]]["display_note"] = "updated"
    path.write_text(json.dumps(document))
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(workers.map(lambda _: view.current()[1], range(4)))
    assert len(calls) == 1
    assert all(result is results[0] for result in results)


def test_refresh_failure_is_not_replaced_with_cached_success(session, monkeypatch):
    view, definitions, _ = session
    path = view._all.root / "data/tasks/graft" / f"{definitions[0]['id']}.json"
    path.write_text(path.read_text() + "\n")
    calls = []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise ValueError("missing current evidence")

    monkeypatch.setattr(SynthesisWorkflow, "status", unavailable)
    for _ in range(2):
        with pytest.raises(ValueError, match="missing current evidence"):
            view.current()
    assert len(calls) == 2


def test_asset_route_serves_media_and_archives_not_repository_secrets(tmp_path):
    definition = task("graft")
    materialize_assets(definition, products("graft"), tmp_path)
    client = create_review_app(
        ReviewWorkspace.open(tmp_path, "reader", include_all_tasks=True)
    ).test_client()
    (tmp_path / ".env").write_text("SECRET=private")
    archive = tmp_path / "artifacts/runs/example/run.json"
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps({"task_id": definition["id"]}))
    assert client.get("/review/asset?path=.env").status_code == 404
    assert client.get("/review/asset?path=missing.jpg").status_code == 404
    assert client.get("/review/asset?path=data/selection.json").status_code == 404
    assert (
        client.get("/review/asset?path=artifacts/runs/example/run.json").status_code
        == 200
    )
    frame = definition["visual_packets"][0]["frame_paths"][0]
    response = client.get("/review/asset", query_string={"path": frame})
    assert response.status_code == 200
    assert response.headers["X-Content-Type-Options"] == "nosniff"


def test_whitespace_and_summary_changes_do_not_recompute_tasks(session):
    view, _, calls = session
    path = view._all.root / "artifacts/reports/task-acceptance.json"
    document = json.loads(path.read_text())
    document.setdefault("summary", {})["display_note"] = "fresh summary, same evidence"
    path.write_text(json.dumps(document, indent=4))
    view.current()
    assert calls == []


def test_one_authored_request_refreshes_only_its_task(session):
    view, definitions, calls = session
    path = view._all.root / "data/instructions/authored.json"
    document = json.loads(path.read_text())
    tid = definitions[0]["id"]
    document["instructions"][tid]["a"] += " Please."
    path.write_text(json.dumps(document))
    view.current()
    assert calls == [{tid}]


def test_one_qualification_record_refreshes_only_its_task(session):
    view, definitions, calls = session
    path = view._all.root / "data/reviews/task-qualification.json"
    document = json.loads(path.read_text()) if path.exists() else {"tasks": {}}
    tid = definitions[0]["id"]
    document["tasks"].setdefault(tid, {})["note"] = "review evidence updated"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))
    view.current()
    assert calls == [{tid}]


def test_unrelated_website_entity_does_not_recompute_tasks(session):
    view, definitions, calls = session
    # Both tasks explicitly use disjoint result ranges.
    for definition in definitions:
        definition.setdefault("truth", {})["site_scope_entity_ids"] = [
            p["item_id"] for p in products(definition["track"])
        ]
        path = (
            view._all.root
            / "data/tasks"
            / definition["track"]
            / f"{definition['id']}.json"
        )
        path.write_text(json.dumps(definition))
    view.current()
    calls.clear()
    path = view._all.root / "sites/egomart/products.json"
    document = json.loads(path.read_text())
    document["products"][0]["name"] += " updated"
    path.write_text(json.dumps(document))
    view.current()
    assert calls == [{definitions[0]["id"]}]


def test_source_record_projection_does_not_mix_other_tasks(session):
    from types import SimpleNamespace

    view, _, _ = session
    path = view._all.root / "source-manifest.json"
    document = {
        "schema_version": 1,
        "dataset": "shared",
        "tasks": [
            {"task_id": "one", "note": "old"},
            {"task_id": "two", "note": "unchanged"},
        ],
    }
    tasks = {
        tid: {"source": {"dossier": path.name, "task_id": tid}}
        for tid in ("one", "two")
    }
    workspace = SimpleNamespace(root=view._all.root, tasks=tasks)
    path.write_text(json.dumps(document))
    before_global, before = view._task_records(path, workspace)
    document["tasks"][0]["note"] = "new"
    path.write_text(json.dumps(document))
    after_global, after = view._task_records(path, workspace)
    assert before_global == after_global
    assert before["one"] != after["one"]
    assert before["two"] == after["two"]
