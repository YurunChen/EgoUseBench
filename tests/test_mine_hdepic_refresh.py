from tools.mine_hdepic_refresh import has_refresh_cue


def test_detects_direct_replacement_cues() -> None:
    assert has_refresh_cue("Change my mind and grab a different knife.")
    assert has_refresh_cue("With my right hand, pick up different sponge.")
    assert has_refresh_cue("Returning the napkin to make a different selection.")
    assert has_refresh_cue("I use this cloth instead of the other one.")


def test_ignores_plain_repetition() -> None:
    assert not has_refresh_cue("Pick up another spoonful of sugar.")
    assert not has_refresh_cue("Wash both hands under running water.")
