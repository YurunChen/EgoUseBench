from __future__ import annotations

import json
import statistics
import unittest
from pathlib import Path

from egocua.schema import load_tasks, referenced_entity_ids
from tests.helpers import ARCHIVED_CORPUS, live_corpus_empty
from tools.catalog.freeze_catalog_order import CATALOGS, order_key

ROOT = Path(__file__).resolve().parents[1]
# A target sitting in the first quarter of the catalog would rank first under the
# default "featured" sort, which is catalog order.
EARLIEST_ACCEPTABLE_MEDIAN_FRACTION = 0.25


class CatalogOrderTest(unittest.TestCase):
    def test_every_catalog_is_stored_in_the_frozen_deterministic_order(self) -> None:
        for relative, bucket, id_key in CATALOGS:
            with self.subTest(catalog=relative):
                entities = json.loads((ROOT / relative).read_text())[bucket]
                keys = [order_key(str(entity[id_key])) for entity in entities]
                self.assertEqual(
                    keys, sorted(keys), f"{relative} is not in frozen order"
                )

    @unittest.skipIf(live_corpus_empty(), ARCHIVED_CORPUS)
    def test_task_targets_do_not_cluster_at_the_front_of_a_catalog(self) -> None:
        tasks = load_tasks(ROOT / "data/tasks")
        for relative, bucket, id_key in CATALOGS:
            site = relative.split("/")[1]
            referenced: set[str] = set()
            for task in tasks:
                if task["site"] == site:
                    referenced.update(referenced_entity_ids(task))
            if not referenced:
                continue
            entities = json.loads((ROOT / relative).read_text())[bucket]
            index = {
                str(entity[id_key]): position
                for position, entity in enumerate(entities)
            }
            positions = [
                index[entity_id] for entity_id in referenced if entity_id in index
            ]
            with self.subTest(catalog=relative):
                self.assertGreater(
                    statistics.median(positions),
                    len(entities) * EARLIEST_ACCEPTABLE_MEDIAN_FRACTION,
                    f"{site} task targets cluster at the front of the catalog; "
                    "position leaks their benchmark role",
                )


if __name__ == "__main__":
    unittest.main()
