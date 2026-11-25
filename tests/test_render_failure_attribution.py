import json
from copy import deepcopy

import pytest

from tools.render.render_failure_attribution import (
    MAIN_MODELS,
    CHALLENGE_GROUPS,
    PLOT_ORDER,
    PLOT_WIDTH,
    SEGMENTS,
    display_counts,
    main,
    render_counts_table,
    comparison_models,
    render_setting_figure,
)


def summary():
    return {
        "stages": {
            stage: {"count": count}
            for stage, count in zip("PULSE", [1, 2, 3, 4, 0], strict=True)
        },
        "valid_model_failures": 13,
        "unknown_failures": 1,
        "pending_attributions": 2,
        "unresolved_run_validity": 5,
    }


def report():
    groups = {f"{model} / online": summary() for model, _ in MAIN_MODELS}
    for model in ("gpt-6.1-terra", "deepseek-flash", "kimi-k2.6", "qwen3.5-27b", "ui-tars-7b"):
        for setting in ("computer_use", "offline_ego_use", "online"):
            groups[f"{model} / {setting}"] = summary()
    overall = summary()
    for stage in "PULSE":
        overall["stages"][stage]["count"] *= len(groups)
    for key in ("valid_model_failures", "unknown_failures", "pending_attributions", "unresolved_run_validity"):
        overall[key] *= len(groups)
    return {
        "statistics": {"matrices": {"model_setting": groups}, "summary": overall},
        "attributions": [
            {"model": "gpt-6.1-terra", "setting": "online", "attribution_method": "human_adjudication"},
            {"model": "deepseek-flash", "setting": "online", "attribution_method": "human_adjudication"},
            {"model": "deepseek-flash", "setting": "online", "attribution_method": "trajectory_review"},
        ],
    }


def test_denominator_keeps_unknown_and_missing_but_excludes_review():
    assert display_counts(summary()) == ([1, 2, 3, 4, 0, 1, 2], 13, 5)


def test_inconsistent_counts_fail_instead_of_renormalizing():
    data = summary()
    data["valid_model_failures"] = 12
    with pytest.raises(ValueError, match="denominator"):
        display_counts(data)


def test_main_figure_compares_common_models_and_preserves_source():
    data = report()
    before = deepcopy(data)
    figure = render_setting_figure(data)
    assert all(name not in figure for name in ("UI-TARS", "Gemini", "Doubao"))
    assert "(a) Computer-Use" in figure
    assert "(b) Offline Ego Use" in figure and "(c) Online Ego Use" in figure
    for name in ("Terra", "DeepSeek", "Kimi", "Qwen"):
        assert figure.count("{" + name + "}") == 1
    assert r"{$n$}" not in figure
    assert r"\node[text=black!60] at" not in figure
    assert figure.count(r"{100\%}") == 3
    assert "Unattributed" in figure
    assert all(label not in figure for label in ("Review", "No output", "Unknown", "pattern="))
    assert data == before


def test_figure_omits_unattributed_legend_when_all_cases_are_assigned():
    data = report()
    for group in data["statistics"]["matrices"]["model_setting"].values():
        group["valid_model_failures"] -= group["unknown_failures"] + group["pending_attributions"]
        group["unknown_failures"] = group["pending_attributions"] = 0
    figure = render_setting_figure(data)
    assert "Unattributed" not in figure
    assert "Attribution status" not in figure
    assert r"\begin{scope}[xshift=1.90cm]" in figure


def test_appendix_retains_ui_tars_without_inventing_missing_settings():
    table = render_counts_table(report())
    assert "UI-TARS & Online" in table
    assert "Gemini & Offline" not in table
    assert r"\color{blue}" not in table and r"\textcolor{blue}" not in table
    assert r"Terra & Computer-Use & 1 & 2 & 3 & 4 & 0 & 0\\" in table


def test_appendix_total_uses_shared_report_summary():
    assert r"Total & All available & 17 & 34 & 51 & 68 & 0 & 2\\" in render_counts_table(report())


def test_appendix_shows_actual_human_labels_not_task_review_counts():
    table = render_counts_table(report())
    assert all(label not in table for label in ("Unassigned", "Unknown", "No valid", r"Task\\review"))
    assert "Human-labeled" in table
    assert r"Terra & Online & 1 & 2 & 3 & 4 & 0 & 1\\" in table
    assert r"DeepSeek & Online & 1 & 2 & 3 & 4 & 0 & 1\\" in table


def test_category_names_match_the_legend_and_appendix_without_slashes():
    names = [SEGMENTS[stage][0] for stage in "PULSE"]
    assert names == ["Perception", "Intent", "Linking", "State Control", "Verification"]
    figure = render_setting_figure(report())
    table = render_counts_table(report())
    for name in names:
        assert "/" not in name
        assert "{" + name + "}" in figure
        assert name in table


def test_comparison_requires_the_same_setting_coverage():
    data = report()
    del data["statistics"]["matrices"]["model_setting"]["gpt-6.1-terra / offline_ego_use"]
    assert "gpt-6.1-terra" not in dict(comparison_models(data))
    assert "Terra" not in render_setting_figure(data)


def test_no_common_models_fail_instead_of_mixing_cohorts():
    data = report()
    groups = data["statistics"]["matrices"]["model_setting"]
    data["statistics"]["matrices"]["model_setting"] = {
        key: value for key, value in groups.items() if key.endswith(" / online")
    }
    with pytest.raises(ValueError, match="all three settings"):
        render_setting_figure(data)


def test_no_eligible_failures_do_not_render_as_zero_error_rates():
    data = report()
    empty = summary()
    empty.update(valid_model_failures=0, unknown_failures=0, pending_attributions=0)
    empty["stages"] = {stage: {"count": 0} for stage in "PULSE"}
    data["statistics"]["matrices"]["model_setting"]["gpt-6.1-terra / online"] = empty
    with pytest.raises(ValueError, match="no eligible"):
        render_setting_figure(data)


def test_panel_percentages_use_their_own_denominators_and_exclude_review():
    data = report()
    cell = data["statistics"]["matrices"]["model_setting"]["gpt-6.1-terra / computer_use"]
    cell["stages"] = {stage: {"count": count} for stage, count in zip("PULSE", [0, 0, 8, 2, 0], strict=True)}
    cell.update(valid_model_failures=10, unknown_failures=0, pending_attributions=0, unresolved_run_validity=3)
    panels = render_setting_figure(data).split(r"\begin{scope}")[1:]
    assert len(panels) == 3
    assert f"(0.00000,-0.140) rectangle ({PLOT_WIDTH * .8:.5f},0.140)" in panels[0]
    assert "{80}" in panels[0] and "{20}" in panels[0]
    for panel in panels[1:]:
        assert "{31}" in panel
        assert "{80}" not in panel


def test_cli_rejects_a_stale_report_before_writing_figures(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/failure-judge.json").write_text(json.dumps({"studies": ["experiments/current"]}))
    destination = tmp_path / "artifacts/reports"
    destination.mkdir(parents=True)
    data = report()
    data["source_studies"] = {"old-run": ["experiments/old"]}
    (destination / "failure-attribution-formal.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="current registered studies"):
        main(["--root", str(tmp_path)])
    assert not (tmp_path / "paper/figures").exists()


def test_cli_exports_only_the_attribution_figure_and_counts(tmp_path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/failure-judge.json").write_text(json.dumps({"studies": ["experiments/current"]}))
    destination = tmp_path / "artifacts/reports"
    destination.mkdir(parents=True)
    data = report()
    data["source_studies"] = {"run": ["experiments/current"]}
    (destination / "failure-attribution-formal.json").write_text(json.dumps(data))

    main(["--root", str(tmp_path)])

    assert {path.name for path in (tmp_path / "paper/figures").iterdir()} == {
        "failure_attribution.tex", "failure_attribution_counts.tex",
    }


def test_two_level_legend_groups_all_categories_without_reclassification():
    assert CHALLENGE_GROUPS == (
        ("Context Grounding", ("P", "L")),
        ("Intent Continuity", ("U",)),
        ("Execution Adaptation", ("S", "E")),
    )
    assert PLOT_ORDER == ("P", "L", "U", "S", "E", "unattributed")
    figure = render_setting_figure(report())
    for label, _ in CHALLENGE_GROUPS:
        assert "{" + label + "}" in figure
    # Source counts and appendix columns stay in their original PULSE order.
    assert display_counts(summary())[0] == [1, 2, 3, 4, 0, 1, 2]
    first_panel = figure.split(r"\begin{scope}")[1]
    linking_start = PLOT_WIDTH / 13
    linking_end = PLOT_WIDTH * 4 / 13
    update_end = PLOT_WIDTH * 6 / 13
    assert f"({linking_start:.5f},-0.140) rectangle ({linking_end:.5f},0.140)" in first_panel
    assert f"({linking_end:.5f},-0.140) rectangle ({update_end:.5f},0.140)" in first_panel
    # Unknown (1) and missing output (2) share the final segment.
    unattributed_start = PLOT_WIDTH * 10 / 13
    assert f"fill=failureunattributed,draw=white,line width=.5pt] ({unattributed_start:.5f},-0.140)" in first_panel
