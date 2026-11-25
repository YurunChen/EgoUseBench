"""The project site renders the paper copy from one content dictionary."""

from __future__ import annotations

import json
from itertools import pairwise

import pytest
from markupsafe import escape

from tools.render import render_website_data
from website import app as site_app
from website.app import create_app
from website.chart import LABEL_GAP, slope_chart
from website.content import ROOT, load_content

@pytest.fixture
def public_content(tmp_path):
    (tmp_path / "data.json").write_bytes((ROOT / "data.json").read_bytes())
    return load_content(tmp_path)


@pytest.fixture
def content(tmp_path, public_content):
    results = {
        "abstract": ["Fixture abstract opening.", "Fixture abstract conclusion."],
        "leaderboard": [
            {
                "model": f"Fixture model {i}",
                "group": "closed" if i < 3 else "open",
                "computer_use": 99 - i,
                "offline": 80 - i,
                "online": 70 - i,
            }
            for i in range(6)
        ],
        "means": [
            {"setting": row["name"], "value": str(value)}
            for row, value in zip(public_content["settings"], (96.5, 77.5, 67.5), strict=True)
        ],
        "gap": "Fixture comparison.",
        "source_note": "Synthetic scores for display tests.",
    }
    replay = {
        **public_content["data"],
        "seconds_per_step": 3.2,
        "samples": [{
            "task_id": "fixture-task",
            "track": "graft",
            "transition": "replace",
            "source": "Synthetic source",
            "license": "Synthetic asset",
            "runs": [{
                "id": "fixture-run",
                "success": True,
                "model": "Fixture model 0",
                "final_cart": ["Fixture item"],
                "steps": [{
                    "step": 1,
                    "path": "/",
                    "messages": [{"who": "wearer", "text": "Fixture request"}],
                }],
            }],
        }],
    }
    (tmp_path / "results.local.json").write_text(json.dumps(results))
    (tmp_path / "replays.local.json").write_text(json.dumps(replay))
    return load_content(tmp_path)


@pytest.fixture
def sample_bundle(tmp_path_factory, content):
    root = tmp_path_factory.mktemp("website-samples")
    for sample in content["data"]["samples"]:
        for run in sample["runs"]:
            folder = root / run["id"]
            folder.mkdir()
            (folder / "wearer.mp4").touch()
            (folder / "poster.jpg").touch()
            for step in run["steps"]:
                (folder / f"page-{step['step']}.jpg").touch()
                (folder / f"view-{step['step']}.jpg").touch()
    return root


@pytest.fixture
def client(monkeypatch, sample_bundle, content):
    monkeypatch.setattr(site_app, "SAMPLE_ROOT", sample_bundle)
    return create_app(content).test_client()


def pages(client):
    return {
        path: client.get(path).get_data(as_text=True)
        for path in ("/", "/benchmark", "/leaderboard", "/docs", "/data")
    }


def test_pages_are_available(client):
    for path in ("/", "/benchmark", "/leaderboard", "/docs", "/data"):
        assert client.get(path).status_code == 200


def test_every_page_links_the_dataset(client, content):
    for html in pages(client).values():
        assert content["dataset_url"] in html


def test_leaderboard_prints_content_scores(client, content):
    html = client.get("/leaderboard").get_data(as_text=True)
    closed_at = html.index("Closed-weight")
    open_at = html.index("Open-weight")
    assert closed_at < open_at
    for row in content["leaderboard"]:
        assert row["model"] in html
        for key in ("computer_use", "offline", "online"):
            assert f"{row[key]:.1f}" in html
    assert "UI-TARS" not in html
    assert content["source_note"] in html


def test_home_shows_one_replay_then_abstract_leaderboard_and_citation(client, content):
    html = client.get("/").get_data(as_text=True)
    expected_sections = ("player", "abstract-title", "leaderboard-title", "citation")
    positions = [html.index(f'id="{section}"') for section in expected_sections]
    assert positions == sorted(positions)
    assert html.index('class="hero-intro"') < html.index('class="optical-floor"')
    assert html.index('class="optical-floor"') < html.index('class="deck"')
    assert 'id="cases-title"' not in html
    assert 'id="catalog-title"' not in html
    assert 'id="overview-title"' not in html
    for paragraph in content["abstract"]:
        assert f"<p>{escape(paragraph)}</p>" in html
    for row in content["means"]:
        assert row["value"] not in html
    first = content["data"]["samples"][0]["runs"][0]
    assert escape(first["steps"][0]["messages"][0]["text"]) in html
    assert f"samples/{first['id']}/wearer.mp4" in html
    assert '<h2 id="final-cart-title">Final cart' in html
    assert '<ul class="cart-items" data-cart-items>' in html
    assert "Copy BibTeX" in html
    assert '<section class="exhibit cite" id="citation" aria-labelledby="citation-title">' in html
    assert "<details" not in html
    assert content["citation"] in html
    assert " and ".join(author["name"] for author in content["authors"]) in content["citation"]


def test_case_link_preserves_the_synchronized_replay(client, content):
    first = content["data"]["samples"][0]["runs"][0]
    html = client.get(f"/?run={first['id']}").get_data(as_text=True)
    assert 'id="player"' in html
    assert 'id="leaderboard-title"' in html
    for lane in content["player"]["lanes"]:
        assert lane["name"] in html
    for sample in content["data"]["samples"]:
        assert escape(sample["task_id"]) in html
        for run in sample["runs"]:
            assert run["id"] in html


def test_pages_work_without_replay_media(monkeypatch, tmp_path, content):
    monkeypatch.setattr(site_app, "SAMPLE_ROOT", tmp_path / "samples")
    client = create_app(content).test_client()
    for path, html in pages(client).items():
        assert client.get(path).status_code == 200
        assert "EgoUseBench" in html
        assert "/static/samples/" not in html
    assert 'id="player"' not in client.get("/").get_data(as_text=True)
    assert 'id="citation"' in client.get("/").get_data(as_text=True)


def test_clean_preview_has_no_results_or_replay(monkeypatch, tmp_path, public_content):
    monkeypatch.setattr(site_app, "SAMPLE_ROOT", tmp_path)
    assert not public_content["leaderboard"]
    assert not public_content["means"]
    assert not public_content["data"]["samples"]
    rendered = pages(create_app(public_content).test_client())
    assert "Evaluation results are not included" in rendered["/leaderboard"]
    assert "Evaluation results are not included" in rendered["/"]
    for paragraph in public_content["abstract"]:
        assert f"<p>{escape(paragraph)}</p>" in rendered["/"]
    assert 'class="hero-intro"' in rendered["/"]
    assert 'class="optical-floor"' not in rendered["/"]
    assert 'id="player"' not in rendered["/"]
    assert 'id="cases-title"' not in rendered["/"]
    assert "Largest drop" not in rendered["/"]
    assert "Replayed samples" not in rendered["/data"]


def test_incomplete_replay_bundle_reports_missing_asset(monkeypatch, tmp_path, content):
    monkeypatch.setattr(site_app, "SAMPLE_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match="wearer.mp4"):
        create_app(content)


@pytest.mark.parametrize("failure", ["archive", "asset", "publish", None])
def test_replay_export_preserves_the_published_bundle_until_ready(
    monkeypatch, tmp_path, failure
):
    public = tmp_path / "website"
    old_samples = public / "static/samples"
    old_samples.mkdir(parents=True)
    (old_samples / "old.mp4").write_bytes(b"existing replay")
    index = public / "replays.local.json"
    index.write_text('{"samples": []}')

    def build(root, samples):
        assert root == tmp_path
        if failure == "archive":
            raise FileNotFoundError("missing archived run")
        run = samples / "new"
        run.mkdir()
        (run / "wearer.mp4").touch()
        if failure != "asset":
            (run / "poster.jpg").touch()
        return {"samples": [{"runs": [{"id": "new", "steps": []}]}]}

    monkeypatch.setattr(render_website_data, "build_replay_data", build)
    if failure == "publish":
        def cannot_replace(_source, _destination):
            raise PermissionError("cannot publish index")

        monkeypatch.setattr(type(index), "replace", cannot_replace)

    if failure:
        with pytest.raises(OSError):
            render_website_data.export_replays(tmp_path)
        assert (old_samples / "old.mp4").read_bytes() == b"existing replay"
        assert index.read_text() == '{"samples": []}'
    else:
        render_website_data.export_replays(tmp_path)
        assert (old_samples / "new/wearer.mp4").is_file()
        assert not (old_samples / "old.mp4").exists()
        assert '"id": "new"' in index.read_text()
    assert not list(public.glob(".replay-*"))


def test_task_counts_render_on_the_data_page(client, content):
    stats = content["data"]["stats"]
    assert sum(track["total"] for track in stats["tracks"]) == stats["tasks"]
    html = client.get("/data").get_data(as_text=True)
    for track in stats["tracks"]:
        assert f"<b>{track['total']}</b>" in html
    for source in stats["sources"]:
        assert source["name"] in html


def test_data_page_credits_each_sample_and_links_its_runs(client, content):
    html = client.get("/data").get_data(as_text=True)
    for sample in content["data"]["samples"]:
        assert sample["license"] in html
        for run in sample["runs"]:
            assert f"/?run={run['id']}#player" in html


def test_slope_chart_keeps_labels_apart(content):
    chart = slope_chart(content["leaderboard"], content["means"])
    labels = sorted(
        [line["label_y"] for line in chart["lines"]] + [chart["mean"]["label_y"]]
    )
    assert all(round(b - a, 6) >= LABEL_GAP for a, b in pairwise(labels))


def test_data_page_marks_the_placeholder(client):
    html = client.get("/data").get_data(as_text=True)
    assert "This Hugging Face repository is not public yet." in html
    assert "task definitions, frozen website snapshots, and observations cleared for release" in html


def test_docs_omit_an_empty_run_command(client, content):
    html = client.get("/docs").get_data(as_text=True)
    assert content["install_command"] in html
    assert 'class="command"' not in html
    assert "20" in html
    assert "final website state" in html


def test_home_credits_the_authors_and_lab(client, content):
    html = client.get("/").get_data(as_text=True)
    credits = html.split('<ul class="credits"', 1)[1].split("</ul>", 1)[0]
    for author in content["authors"]:
        assert author["name"] in credits
    assert 'href="https://yurunchen.github.io/"' in credits
    resources = html.split('<nav class="hero-links"', 1)[1].split("</nav>", 1)[0]
    for key in ("code_url", "dataset_url", "lab_url"):
        assert f'href="{content[key]}"' in resources
    assert resources.count('class="button"') == 4
    assert "pending" not in resources
    header = html.split("<header", 1)[1].split("</header>", 1)[0]
    assert "ai4gc-lab-logo.png" in header
    assert "zhejiang-university-logo.png" in header
    assert content["affiliation"] in header
    assert "href=" not in header


def test_paper_opens_the_abstract_until_arxiv_url_is_configured(public_content):
    client = create_app(public_content).test_client()
    html = client.get("/").get_data(as_text=True)
    assert '<a class="button" href="#abstract">Paper</a>' in html
    assert '<section class="exhibit" id="abstract"' in html
    assert client.get("/paper.pdf").status_code == 404


def test_paper_links_to_configured_arxiv_entry(public_content):
    public_content["paper_url"] = "https://arxiv.org/abs/test-id"
    app = create_app(public_content)
    html = app.test_client().get("/").get_data(as_text=True)
    assert 'href="https://arxiv.org/abs/test-id" target="_blank" rel="noopener">Paper</a>' in html
    assert "arXiv pending" not in html
    assert 'href="/paper.pdf"' not in html


def test_unknown_path_uses_the_site_frame(client):
    response = client.get("/missing")
    assert response.status_code == 404
    html = response.get_data(as_text=True)
    assert 'href="/"' in html
    assert "EgoUseBench" in html


def test_create_app_rejects_incomplete_content(content):
    missing = {key: value for key, value in content.items() if key != "citation"}
    with pytest.raises(ValueError, match="citation"):
        create_app(missing)

    empty_url = dict(content)
    empty_url["dataset_url"] = "  "
    with pytest.raises(ValueError, match="dataset_url"):
        create_app(empty_url)

    short = dict(content)
    short["leaderboard"] = content["leaderboard"][:-1]
    with pytest.raises(ValueError, match="6 models"):
        create_app(short)


def test_benchmark_states_the_main_evaluation(client):
    html = client.get("/benchmark").get_data(as_text=True)
    assert "142" in html
    assert "93 additional Online tasks" in html
    assert "Product selection" in html
