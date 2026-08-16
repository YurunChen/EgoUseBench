from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OPAQUE_DOCUMENT_ID = re.compile(r"^d_[0-9a-f]{12}$")


def test_resource_screened_seeds_resolve_to_frozen_documents_and_exact_fields() -> None:
    audit = json.loads(
        (ROOT / "data/sources/cross_domain_resource_audit.json").read_text()
    )
    collection = json.loads(
        (ROOT / "data/sources/egodocs_seed_collection.json").read_text()
    )
    documents = {document["document_id"]: document for document in collection["documents"]}

    assert audit["status"] == "resource_screened"
    assert collection["status"] == "frozen_seed_collection"
    assert len(documents) == len(collection["documents"])
    assert all(OPAQUE_DOCUMENT_ID.fullmatch(document_id) for document_id in documents)
    assert len({document["source"]["url"] for document in documents.values()}) == len(documents)

    for seed in audit["seeds"]:
        resources = [seed["target_resource"]]
        resources += [
            seed[key]
            for key in ("later_distractor_resource", "other_candidate")
            if key in seed
        ]
        for resource in resources:
            document = documents[resource["resource_id"]]
            assert document["title"]
            assert document["source"]["url"]
            assert document["source"]["retrieved_at"]
            for field, value in resource.get("exact_fields", {}).items():
                assert document["facts"][field] == value


def test_each_seed_has_a_target_and_multiple_near_neighbors() -> None:
    collection = json.loads(
        (ROOT / "data/sources/egodocs_seed_collection.json").read_text()
    )
    counts: dict[str, int] = {}
    for document in collection["documents"]:
        counts[document["kind"]] = counts.get(document["kind"], 0) + 1

    assert counts["reference_article"] >= 8
    assert counts["poster_record"] >= 4
