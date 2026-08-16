"""Compile reviewed construction roles into executable tasks and site products."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

from .schema import (
    CLARIFICATION_SLOTS,
    TARGET_GRANULARITIES,
    validate_task,
)
from .workspace import resolve_within


class ConstructionError(ValueError):
    """Raised when a construction spec cannot be compiled without guessing."""


PRODUCT_FIELDS = {
    "role", "name", "price", "image_path", "image_provenance", "tier",
    "brand", "rating", "ratings_count", "rating_provenance", "compatibility_provenance",
    "text_provenance", "badge", "unit_price", "attributes",
}
PRODUCT_TIERS = {"target", "hard", "medium", "easy"}
MINIMUM_TIER_COUNTS = {"target": 1, "hard": 3, "medium": 4}
CLARIFICATION_SLOTS_BY_SUBTYPE = {
    "position": {"position"},
    "sequence": {"sequence"},
    "variant": {"appearance", "brand", "variant"},
}


def _acceptable_states(
    options: list[list[str]], fields: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    return [
        {
            "selected_entity_ids": entity_ids,
            "fields": dict(fields or {}),
            "submission_completed": True,
        }
        for entity_ids in options
    ]


def _goal_fields(blueprint: Mapping[str, Any], key: str) -> dict[str, str]:
    value = blueprint.get(key, {})
    if not isinstance(value, dict) or any(
        not isinstance(field, str) or not field
        or not isinstance(answer, str) or not answer
        for field, answer in value.items()
    ):
        raise ConstructionError(f"{key} must map non-empty strings to non-empty strings")
    return dict(value)


def _nonempty_strings(value: Any, keys: set[str]) -> bool:
    return isinstance(value, dict) and set(value) == keys and all(
        isinstance(value[key], str) and value[key].strip() for key in keys
    )


def _valid_claim_provenance(value: Any) -> bool:
    if not isinstance(value, dict) or not str(value.get("access_note", "")).strip():
        return False
    if str(value.get("source_url", "")).strip():
        return bool(str(value.get("retrieved_at", "")).strip())
    return (
        value.get("method") == "benchmark_authored_mock_product"
        and bool(str(value.get("created_at", "")).strip())
    )


def _validate_blueprint(record: dict[str, Any]) -> dict[str, Any]:
    task_id = str(record.get("task_id", ""))
    track = record.get("family")
    subtype = record.get("subtype")
    blueprint = record.get("construct", {}).get("task_blueprint")
    if not isinstance(blueprint, dict):
        raise ConstructionError(f"{task_id}: task_blueprint is missing")
    if blueprint.get("target_granularity") not in TARGET_GRANULARITIES:
        raise ConstructionError(f"{task_id}: target_granularity is invalid")

    if track == "graft":
        if "b_essential_witness" in blueprint:
            raise ConstructionError(
                f"{task_id}: Graft blueprint contains obsolete field 'b_essential_witness'"
            )
        if not _nonempty_strings(blueprint.get("instructions"), {"a", "b", "bundled"}):
            raise ConstructionError(f"{task_id}: Graft instructions must define a, b, and bundled")
        _role_sets(blueprint.get("initial_goal_roles"), f"{task_id}/initial_goal_roles")
        _role_sets(blueprint.get("final_goal_roles"), f"{task_id}/final_goal_roles")
    elif track == "refresh":
        if not isinstance(blueprint.get("instruction"), str) or not blueprint["instruction"].strip():
            raise ConstructionError(f"{task_id}: Refresh instruction must be non-empty")
        for key in ("initial_target_role", "final_target_role"):
            if not isinstance(blueprint.get(key), str) or not blueprint[key].strip():
                raise ConstructionError(f"{task_id}: Refresh {key} must be non-empty")
    elif track == "clarify":
        legacy = {
            "instructions", "clarification", "source_candidate_roles",
            "hidden_target_assignment", "final_goal_roles", "catalog_target_roles",
        } & set(blueprint)
        if legacy:
            raise ConstructionError(
                f"{task_id}: Clarify blueprint contains obsolete fields {sorted(legacy)}"
            )
        if not _nonempty_strings(
            {
                "ambiguous": blueprint.get("ambiguous_instruction"),
                "oracle": blueprint.get("oracle_instruction"),
            },
            {"ambiguous", "oracle"},
        ):
            raise ConstructionError(f"{task_id}: Clarify instructions must be non-empty")
        candidates = blueprint.get("candidate_roles")
        if (
            not isinstance(candidates, list)
            or len(candidates) != 2
            or any(not isinstance(role, str) or not role for role in candidates)
            or len(set(candidates)) != 2
        ):
            raise ConstructionError(f"{task_id}: Clarify requires exactly two candidate roles")
        if blueprint.get("hidden_target_role") not in candidates:
            raise ConstructionError(f"{task_id}: hidden_target_role must be one of the candidates")
        slot = blueprint.get("valid_slot")
        if (
            slot not in CLARIFICATION_SLOTS
            or subtype not in CLARIFICATION_SLOTS_BY_SUBTYPE
            or slot not in CLARIFICATION_SLOTS_BY_SUBTYPE[subtype]
        ):
            raise ConstructionError(
                f"{task_id}: Clarify valid_slot is invalid for subtype {subtype!r}"
            )
        for key in ("canonical_question", "canonical_answer"):
            if not isinstance(blueprint.get(key), str) or not blueprint[key].strip():
                raise ConstructionError(f"{task_id}: Clarify {key} must be non-empty")
    else:
        raise ConstructionError(f"{task_id}: invalid task family")
    return blueprint


def opaque_item_id(task_id: str, role: str) -> str:
    digest = hashlib.sha256(f"egocua-v1\0{task_id}\0{role}".encode()).hexdigest()[:12]
    return f"p_{digest}"


def _ordered_roles(task_id: str, roles: list[str]) -> list[str]:
    return sorted(
        roles,
        key=lambda role: hashlib.sha256(
            f"product-order-v1\0{task_id}\0{role}".encode()
        ).digest(),
    )


def _role_sets(value: Any, context: str) -> list[list[str]]:
    if not isinstance(value, list) or not value or not all(
        isinstance(option, list) and option
        and all(isinstance(role, str) and role for role in option)
        for option in value
    ):
        raise ConstructionError(f"{context}: must be a non-empty list of non-empty role lists")
    return value


def _map_sets(value: Any, role_ids: Mapping[str, str], context: str) -> list[list[str]]:
    options = _role_sets(value, context)
    try:
        return [[role_ids[role] for role in option] for option in options]
    except KeyError as exc:
        raise ConstructionError(f"{context}: product spec has no role {exc.args[0]!r}") from exc


def _products(
    task_id: str, spec: dict[str, Any], *, root: Path
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if spec.get("schema_version") != 1 or spec.get("task_id") != task_id:
        raise ConstructionError(f"{task_id}: product spec identity or schema_version is invalid")
    spec_products = spec.get("products")
    if not isinstance(spec_products, list) or len(spec_products) != 16:
        raise ConstructionError(f"{task_id}: product spec must contain exactly 16 products")

    by_role: dict[str, dict[str, Any]] = {}
    for index, product in enumerate(spec_products):
        context = f"{task_id}/products/{index}"
        if not isinstance(product, dict):
            raise ConstructionError(f"{context}: product must be an object")
        unknown = set(product) - PRODUCT_FIELDS
        if unknown:
            raise ConstructionError(f"{context}: unsupported fields {sorted(unknown)}")
        role = product.get("role")
        if not isinstance(role, str) or not role:
            raise ConstructionError(f"{context}: role must be a non-empty string")
        if role in by_role:
            raise ConstructionError(f"{task_id}: duplicate product role {role!r}")
        for required in ("name", "price", "image_path", "image_provenance", "tier"):
            if required not in product:
                raise ConstructionError(f"{context}: missing {required!r}")
        if product["tier"] not in PRODUCT_TIERS:
            raise ConstructionError(f"{context}: invalid construction tier")
        attributes = product.get("attributes", {})
        if not isinstance(attributes, dict) or {"role", "annotation", "rank"} & set(attributes):
            raise ConstructionError(f"{context}: attributes contain construction-only metadata")
        text_provenance = product.get("text_provenance")
        if text_provenance is not None and not _valid_claim_provenance(text_provenance):
            raise ConstructionError(
                f"{context}: text_provenance needs an external source or an explicit "
                "benchmark-authored mock method"
            )
        rating_fields = {"rating", "ratings_count", "rating_provenance"} & set(product)
        if rating_fields:
            if rating_fields != {"rating", "ratings_count", "rating_provenance"}:
                raise ConstructionError(
                    f"{context}: rating, ratings_count, and rating_provenance must be supplied together"
                )
            rating = product["rating"]
            ratings_count = product["ratings_count"]
            provenance = product["rating_provenance"]
            if (
                not isinstance(rating, (int, float))
                or isinstance(rating, bool)
                or not 0 <= float(rating) <= 5
            ):
                raise ConstructionError(f"{context}: rating must be a number from 0 to 5")
            if (
                not isinstance(ratings_count, int)
                or isinstance(ratings_count, bool)
                or ratings_count < 0
            ):
                raise ConstructionError(f"{context}: ratings_count must be a non-negative integer")
            if not _valid_claim_provenance(provenance):
                raise ConstructionError(
                    f"{context}: rating_provenance needs an external source or an explicit "
                    "benchmark-authored mock method"
                )
        compatibility_claims = {
            "compatible_with", "compatible_parts", "compatible_models"
        } & set(attributes)
        compatibility_provenance = product.get("compatibility_provenance")
        if compatibility_claims:
            if not _valid_claim_provenance(compatibility_provenance):
                raise ConstructionError(
                    f"{context}: compatibility_provenance must use an external source or an "
                    "explicit benchmark-authored mock method"
                )
        elif compatibility_provenance is not None:
            raise ConstructionError(
                f"{context}: compatibility_provenance requires a compatibility attribute"
            )
        by_role[role] = product

    tiers = [product["tier"] for product in by_role.values()]
    for tier, minimum in MINIMUM_TIER_COUNTS.items():
        if tiers.count(tier) < minimum:
            raise ConstructionError(
                f"{task_id}: product spec needs at least {minimum} {tier!r} items"
            )

    role_ids = {role: opaque_item_id(task_id, role) for role in by_role}
    if len(set(role_ids.values())) != 16:
        raise ConstructionError(f"{task_id}: opaque item ID collision")

    products: list[dict[str, Any]] = []
    for role in _ordered_roles(task_id, list(by_role)):
        source = by_role[role]
        image_path = str(source["image_path"])
        try:
            image = resolve_within(root, image_path, context=f"{task_id}/{role} product image")
        except (ValueError, FileNotFoundError) as exc:
            raise ConstructionError(str(exc)) from exc
        provenance = source["image_provenance"]
        if not isinstance(provenance, dict) or "sha256" in provenance:
            raise ConstructionError(
                f"{task_id}/{role}: provenance must be an object without a hand-authored sha256"
            )
        item = {
            key: value for key, value in source.items()
            if key not in {"role", "tier", "image_provenance"}
        }
        item["item_id"] = role_ids[role]
        item["image_provenance"] = {
            **provenance,
            "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
        }
        products.append(item)
    return products, role_ids


def _blueprint_roles(record: dict[str, Any]) -> set[str]:
    blueprint = _validate_blueprint(record)
    track = record["family"]
    roles: set[str] = set()
    if track == "graft":
        for key in ("initial_goal_roles", "final_goal_roles"):
            for option in _role_sets(
                blueprint.get(key), f"{record['task_id']}/{key}"
            ):
                roles.update(option)
    elif track == "refresh":
        roles.update({
            str(blueprint["initial_target_role"]),
            str(blueprint["final_target_role"]),
        })
        later_distractor = blueprint.get("later_distractor_role")
        if isinstance(later_distractor, str) and later_distractor:
            roles.add(later_distractor)
    else:
        roles.update(str(role) for role in blueprint["candidate_roles"])
    return roles


def _compile_task_from_roles(
    record: dict[str, Any], role_ids: Mapping[str, str], *,
    source_identity: dict[str, str], site_data: dict[str, Any], root: str | Path,
) -> dict[str, Any]:
    task_id = str(record["task_id"])
    track = str(record["family"])
    subtype = record.get("subtype")
    blueprint = _validate_blueprint(record)
    granularity = blueprint.get("target_granularity")
    packets = [
        {"id": packet.get("id"), "frame_paths": packet.get("frame_paths")}
        for packet in record.get("packets", [])
    ]
    task: dict[str, Any] = {
        "schema_version": 2,
        "id": task_id,
        "track": track,
        "site": str(site_data["site"]),
        "source": source_identity,
        "instructions": {},
        "visual_packets": packets,
    }

    if track == "graft":
        task["goal"] = {
            "target_granularity": granularity,
            "acceptable_states": _acceptable_states(
                _map_sets(
                    blueprint.get("final_goal_roles"), role_ids,
                    f"{task_id}/final_goal_roles",
                ),
                _goal_fields(blueprint, "final_goal_fields"),
            ),
        }
        task["instructions"] = blueprint.get("instructions")
        task["graft"] = {
            "integration_type": subtype,
            "initial_goal": {
                "target_granularity": granularity,
                "acceptable_states": _acceptable_states(
                    _map_sets(
                        blueprint.get("initial_goal_roles"), role_ids,
                        f"{task_id}/initial_goal_roles",
                    ),
                    _goal_fields(blueprint, "initial_goal_fields"),
                ),
            },
        }
    elif track == "refresh":
        initial_role = str(blueprint["initial_target_role"])
        final_role = str(blueprint["final_target_role"])
        if initial_role not in role_ids:
            raise ConstructionError(
                f"{task_id}: initial_target_role is absent from the product spec"
            )
        if final_role not in role_ids:
            raise ConstructionError(
                f"{task_id}: final_target_role is absent from the product spec"
            )
        task["goal"] = {
            "target_granularity": granularity,
            "acceptable_states": _acceptable_states(
                [[role_ids[final_role]]], _goal_fields(blueprint, "goal_fields")
            ),
        }
        task["instructions"] = {"base": blueprint.get("instruction")}
        task["refresh"] = {
            "policy": subtype,
            "initial_entity_id": role_ids[initial_role],
        }
    else:
        candidate_roles = [str(role) for role in blueprint["candidate_roles"]]
        hidden_role = str(blueprint["hidden_target_role"])
        missing_roles = sorted(set(candidate_roles) - set(role_ids))
        if missing_roles:
            raise ConstructionError(
                f"{task_id}: product spec has no candidate roles {missing_roles}"
            )
        task["goal"] = {
            "target_granularity": granularity,
            "acceptable_states": _acceptable_states(
                [[role_ids[hidden_role]]], _goal_fields(blueprint, "goal_fields")
            ),
        }
        task["instructions"] = {
            "ambiguous": blueprint.get("ambiguous_instruction"),
            "oracle": blueprint.get("oracle_instruction"),
        }
        task["clarification"] = {
            "candidate_ids": [role_ids[role] for role in candidate_roles],
            "valid_slot": blueprint.get("valid_slot"),
            "canonical_answer": blueprint.get("canonical_answer"),
            "max_questions": 1,
        }

    validate_task(task, site_data, root=root, require_assets=True)
    return task


def compile_task_from_site(
    record: dict[str, Any], *, source_identity: dict[str, str],
    site_data: dict[str, Any], root: str | Path,
) -> dict[str, Any]:
    """Compile a screened blueprint against an already frozen site."""

    task_id = str(record.get("task_id", ""))
    if not task_id or record.get("status") != "video_screened":
        raise ConstructionError("only identified video_screened records can be compiled")
    entity_ids = {
        str(entity.get("entity_id"))
        for entity in site_data.get("entities", [])
        if isinstance(entity, dict) and entity.get("entity_id")
    }
    roles = _blueprint_roles(record)
    blueprint = record.get("construct", {}).get("task_blueprint", {})
    explicit_role_ids = blueprint.get("environment_role_ids", {})
    if not isinstance(explicit_role_ids, dict):
        raise ConstructionError(
            f"{task_id}: environment_role_ids must be an object"
        )
    unknown_roles = sorted(set(explicit_role_ids) - set(roles))
    if unknown_roles:
        raise ConstructionError(
            f"{task_id}: environment_role_ids declares unknown roles {unknown_roles}"
        )
    role_ids = {
        role: str(explicit_role_ids.get(role) or opaque_item_id(task_id, role))
        for role in roles
    }
    missing = sorted(set(role_ids.values()) - entity_ids)
    if missing:
        raise ConstructionError(
            f"{task_id}: frozen site is missing referenced entities {missing}"
        )
    return _compile_task_from_roles(
        record, role_ids, source_identity=source_identity,
        site_data=site_data, root=root,
    )


def compile_task(
    record: dict[str, Any], spec: dict[str, Any], *, source_identity: dict[str, str],
    root: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_id = str(record.get("task_id", ""))
    track = record.get("family")
    subtype = record.get("subtype")
    if not task_id or record.get("status") != "video_screened":
        raise ConstructionError("only identified video_screened records can be compiled")
    if track not in {"graft", "refresh", "clarify"}:
        raise ConstructionError(f"{task_id}: invalid task family")
    workspace = Path(root).resolve()
    products, role_ids = _products(task_id, spec, root=workspace)
    task = _compile_task_from_roles(
        record, role_ids, source_identity=source_identity,
        site_data={
            "site": "egomart",
            "entities": [
                {**product, "entity_id": product["item_id"]} for product in products
            ],
        },
        root=workspace,
    )
    return task, products
