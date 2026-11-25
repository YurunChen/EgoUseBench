"""Project site copy, with optional local results and replay data."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


PUBLIC_COPY = {
    "title": "EgoUseBench",
    "paper_title": (
        "EgoUseBench: Benchmarking Computer-Use Agents under "
        "Evolving First-Person Contexts"
    ),
    "subtitle": "Benchmarking computer-use agents under evolving first-person contexts",
    "abstract": [
        "EgoUseBench evaluates how computer-use agents handle changing first-person context. "
        "Across 693 task instances, the benchmark compares explicit requests, recordings "
        "available before execution, and observations delivered while an agent acts."
    ],
    "composite_result": (
        "93 additional Online tasks combine multiple change types and use a 30-step budget."
    ),
    "intro": "693 tasks from 74 hours of first-person recordings.",
    "authors": [
        {"name": "Yurun Chen", "url": "https://yurunchen.github.io/"},
        {"name": "Biao Yi"},
        {"name": "Qihang Yu"},
        {"name": "Keting Yin"},
        {"name": "Shengyu Zhang"},
    ],
    "lab": "AI4GC Lab",
    "lab_url": "https://ai4gc.org/",
    "affiliation": "Zhejiang University",
    "code_url": "https://github.com/YurunChen/EgoUseBench",
    "paper_url": "",
    "dataset_url": "https://huggingface.co/datasets/yurun-chen/EgoUseBench",
    "data_contents": (
        "task definitions, a source index, and redistributable observation frames"
    ),
    "source_access_note": (
        "Ego2Web, Ego4D-derived sources, and EgoLife are listed by source. "
        "Obtain restricted recordings through the original dataset providers."
    ),
    "run_command": "",
    "install_command": "uv sync",
    "protocol": (
        "Each task is run once, for at most 20 steps, "
        "and scored from the final website state."
    ),
    "stats": [
        {"value": "693", "label": "tasks"},
        {"value": "74 hours", "label": "first-person recordings"},
        {"value": "3", "label": "interaction settings"},
    ],
    "settings": [
        {
            "name": "Computer-Use",
            "summary": "Explicit instructions, with no first-person recording.",
            "strip": ["say", "act", "act", "act", "act", "act", "act"],
        },
        {
            "name": "Offline Ego Use",
            "summary": "The recording is available before execution.",
            "strip": ["view", "view", "view", "say", "act", "act", "act"],
        },
        {
            "name": "Online Ego Use",
            "summary": "Observations and requests arrive while the agent acts.",
            "strip": ["say", "view", "act", "view", "say", "view", "act"],
        },
    ],
    "strip_legend": [
        {"kind": "view", "label": "first-person frames"},
        {"kind": "say", "label": "the wearer speaks"},
        {"kind": "act", "label": "agent acts on the website"},
    ],
    "task_types": [
        {
            "name": "Graft",
            "summary": "The wearer revises the request.",
            "example": [
                {
                    "kind": "say",
                    "text": (
                        "Add one retail pack of the food I spread "
                        "the filling onto…"
                    ),
                },
                {"kind": "note", "text": "the agent finishes and waits"},
                {
                    "kind": "say",
                    "text": (
                        "Also add one regular 2-kilogram jar of the smooth "
                        "spread I scoop from in the next stage."
                    ),
                },
            ],
        },
        {
            "name": "Refresh",
            "summary": "A later view changes what the earlier scene could show.",
            "example": [
                {
                    "kind": "say",
                    "text": (
                        "Could you add a ready-to-drink coffee like the "
                        "one I'm making to my cart?"
                    ),
                },
                {"kind": "view", "text": "milk goes into the coffee"},
                {"kind": "note", "text": "plain coffee → coffee with milk"},
            ],
        },
        {
            "name": "Clarify",
            "summary": "The agent asks once, and the wearer answers.",
            "example": [
                {
                    "kind": "say",
                    "text": (
                        "Add a bag of the candy right next to the blue "
                        "Airheads on that shelf to my cart."
                    ),
                },
                {"kind": "act", "text": "the agent asks which one"},
                {"kind": "say", "text": "The one on the left."},
            ],
        },
    ],
    "states": [
        {
            "name": "First-person observations",
            "summary": "Frames arrive on the replay clock.",
        },
        {
            "name": "User requirements",
            "summary": "Requests and answers arrive as the wearer speaks.",
        },
        {
            "name": "Website state",
            "summary": "The cart, saves, and fields change through the agent's actions.",
        },
    ],
    "domains": [
        {
            "name": "Product selection",
            "summary": "Add the item the wearer is using.",
        },
        {
            "name": "Tutorial retrieval",
            "summary": "Find a tutorial for the activity underway.",
        },
        {
            "name": "Place bookmarking",
            "summary": "Save a place that fits the request.",
        },
    ],
    "main_eval": {
        "online_tasks": 200,
        "graft": 142,
        "refresh": 30,
        "clarify": 28,
        "steps": 20,
        "paired": 200,
        "composite": (
            "93 additional Online tasks combine multiple change types "
            "and are reported in the paper."
        ),
    },
    "site_names": {"egomart": "EgoMart", "egomedia": "EgoMedia", "egomaps": "EgoMaps"},
    "groups": [
        {"id": "closed", "label": "Closed-weight"},
        {"id": "open", "label": "Open-weight"},
    ],
}


def _bibtex(content: dict) -> str:
    authors = " and ".join(author["name"] for author in content["authors"])
    return (
        "@misc{egousebench2026,\n"
        f"  title = {{{content['paper_title']}}},\n"
        f"  author = {{{authors}}},\n"
        "  year = {2026}\n"
        "}"
    )


def load_content(root: Path = ROOT) -> dict:
    content = {
        **PUBLIC_COPY,
        "leaderboard": [],
        "means": [],
        "source_note": "",
        "player": {
            "labels": {},
            "notes": {},
            "targets": {},
            "lanes": [
                {"kind": "view", "name": "View"},
                {"kind": "say", "name": "Speech"},
                {"kind": "act", "name": "Cart"},
            ],
            "standby_note": "The agent waits for the next request.",
        },
        "data": {**_read_json(root / "data.json"), "samples": []},
    }
    results = root / "results.local.json"
    if results.is_file():
        content.update(_read_json(results))
    replays = root / "replays.local.json"
    if replays.is_file():
        content["data"] = _read_json(replays)
    content["citation"] = _bibtex(content)
    return content


SITE = load_content()
