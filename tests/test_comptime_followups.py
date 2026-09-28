"""Boolean set names, confidence-level labels, repeated roles, reserved names and all-missing groups."""

import pandas as pd
import pytest

import viz_calc as vc
from viz_calc import datasets

# --- upset ----------------------------------------------------------------------------------------------------------

def test_upset_boolean_set_names_give_correct_degrees():
    two = pd.DataFrame({False: [True, False], True: [True, True]})  # intersections {False, True} and {True}
    table = vc.upset(two).table
    degrees = {tuple(bool(v) for v in row[[0, 1]]): row["degree"] for _, row in table.iterrows()}
    assert degrees == {(True, True): 2, (False, True): 1}


def test_upset_boolean_set_names_with_three_intersections():
    three = pd.DataFrame({False: [True, False, True, True], True: [True, True, False, True]})
    res = vc.upset(three, sort_by="degree")
    assert res.table["degree"].tolist() == [1, 1, 2]
    assert res.table["size"].tolist() == [1, 1, 2]
    # the matrix marks each intersection's members: two dots for {False, True}, one for the others
    members = [len(c.get_offsets()) for c in res.axes[1].collections[1::2]]
    assert members == [1, 1, 2]


# --- percent_grid ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("level, text", [(0.999, "99.9% CI"), (0.975, "97.5% CI"), (0.95, "95% CI"),
                                         (0.8, "80% CI")])
def test_percent_grid_level_label_is_not_rounded(level, text):
    d = pd.DataFrame({"s": [1, 0, 1, 1, 1, 0]})
    title = vc.percent_grid(d, "s", level=level).axes.flat[0].get_title()
    assert title.splitlines()[1].startswith(text + " ")


# --- animated_bubble ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("role", ["x", "y", "size"])
def test_animated_bubble_time_in_another_role_raises(role):
    t = datasets.trial().assign(yr=lambda d: d.index % 2)
    kwargs = {"x": "score", "y": "baseline", "size": "score", role: "yr"}
    with pytest.raises(ValueError, match=f"time and {role} must be different columns; both are 'yr'"):
        vc.animated_bubble(t, time="yr", **kwargs)


def test_animated_bubble_shared_column_appears_once_in_table():
    t = datasets.trial().assign(yr=lambda d: d.index % 2)
    table = vc.animated_bubble(t, time="yr", x="score", y="baseline", size="score").table
    assert table.columns.tolist() == ["yr", "score", "baseline"]


# --- nested_pie -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("renamed, outer, inner", [("arm", "value", "site"), ("site", "arm", "value"),
                                                   ("arm", "percent_of_total", "site")])
def test_nested_pie_category_named_like_output_column_raises(renamed, outer, inner):
    new = outer if renamed == "arm" else inner
    data = datasets.trial().rename(columns={renamed: new})
    with pytest.raises(ValueError, match=rf"rename column\(s\) \['{new}'\]"):
        vc.nested_pie(data, outer, inner)


def test_nested_pie_value_column_named_value_still_works():
    data = datasets.trial().assign(value=1.0)
    table = vc.nested_pie(data, "arm", "site", value="value").table
    assert table["value"].sum() == len(data)


# --- all-missing group column ---------------------------------------------------------------------------------------

def test_circular_bar_all_missing_group_draws_one_missing_cluster():
    w = pd.DataFrame({"lab": list("abcd"), "v": [4.0, 3, 2, 1], "grp": [None] * 4})
    res = vc.circular_bar(w, "lab", "v", group="grp")
    assert res.table["lab"].tolist() == list("abcd")
    texts = [t.get_text() for t in res.axes.texts]
    assert "(missing)" in texts


def test_gantt_all_missing_group_draws_every_task_as_missing():
    projects = datasets.projects().assign(team=None)
    res = vc.gantt(projects, "task", "start", "end", group="team")
    assert len(res.table) == len(projects)
    assert [t.get_text() for t in res.axes.get_legend().get_texts()] == ["(missing)"]
