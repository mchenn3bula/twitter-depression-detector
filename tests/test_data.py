import pandas as pd
import pytest

from keyword_study.data import clean_text, connected_groups, load_split, prepare, remove_keyword


def test_normalization_keeps_negation_and_keyword_boundaries():
    assert clean_text("I am not happy! @name https://example.org") == "I am not happy! [USER] [URL]"
    assert (
        remove_keyword("DEPRESSION, not depressionlike or postdepression")
        == ", not depressionlike or postdepression"
    )


def test_transitive_users_and_ablation_duplicates():
    groups = connected_groups(
        ["hello", "hello depression", "other", "separate"], ["a", "b", "b", "c"], "depression"
    )
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]


def test_deterministic_disjoint_splits_and_tampering(source_csv, tmp_path):
    first = prepare(source_csv, tmp_path / "one")
    second = prepare(source_csv, tmp_path / "two")
    assert first == second
    parts = [load_split(tmp_path / "one", name)[0] for name in ["train", "validation", "test"]]
    assert sum(map(len, parts)) == 120
    for i, left in enumerate(parts):
        assert set(left.label) == {0, 1}
        for right in parts[i + 1 :]:
            assert set(left.row_id).isdisjoint(right.row_id)
            assert set(left.group_id).isdisjoint(right.group_id)
    path = tmp_path / "one" / "train.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        load_split(tmp_path / "one", "train")


@pytest.mark.parametrize("column,value", [("post_text", " "), ("label", 2), ("user_id", None)])
def test_invalid_data_rejected(source_csv, tmp_path, column, value):
    frame = pd.read_csv(source_csv)
    frame.loc[0, column] = value
    frame.to_csv(source_csv, index=False)
    with pytest.raises(ValueError):
        prepare(source_csv, tmp_path / "invalid")


def test_missing_users_require_explicit_opt_out(source_csv, tmp_path):
    pd.read_csv(source_csv).drop(columns="user_id").to_csv(source_csv, index=False)
    with pytest.raises(ValueError, match="Required columns"):
        prepare(source_csv, tmp_path / "bad")
    manifest = prepare(source_csv, tmp_path / "good", group_column=None)
    assert manifest["group_column"] is None
