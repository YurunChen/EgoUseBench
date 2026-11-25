import unittest

from tools.mine.mine_divergence_events import mine


def _row(identifier: int, instruction: str) -> dict:
    return {
        "id": identifier,
        "vid": f"video-{identifier}",
        "instruction": instruction,
        "visual_clues": "00:00-00:05",
    }


class RefreshMiningTest(unittest.TestCase):
    def test_separates_selection_from_replacement_evidence(self) -> None:
        report = mine(
            [
                _row(1, "Find the second snack picked up by the shopper in the video."),
                _row(
                    2, "The shopper put the item back before choosing another product."
                ),
                _row(3, "Report the final price shown on the product page."),
            ]
        )

        refresh = report["refresh"]
        self.assertEqual(refresh["physical_selection_ids"], [1])
        self.assertEqual(refresh["explicit_replacement_ids"], [2])

    def test_final_web_price_is_not_a_physical_replacement(self) -> None:
        report = mine([_row(1, "Report the final price shown on the product page.")])

        self.assertEqual(report["refresh"]["explicit_replacement_rows"], 0)


if __name__ == "__main__":
    unittest.main()
