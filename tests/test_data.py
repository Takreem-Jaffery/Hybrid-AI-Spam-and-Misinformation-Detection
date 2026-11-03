import pytest

from spamdetector import config
from spamdetector.data import load_dataset, split_dataset


def _write(path, n_spam=20, n_ham=80):
    rows = [f"spam\tfree prize number {i} click now" for i in range(n_spam)] + [f"ham\tsee you at {i} ok" for i in range(n_ham)]
    path.write_text("\n".join(rows) + "\n")


def test_missing_file_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_dataset(tmp_path / "nope")


def test_validation_rejects_wrong_dataset(tmp_path):
    f = tmp_path / "d"
    _write(f)
    with pytest.raises(ValueError, match="Unexpected dataset"):
        load_dataset(f)
    assert len(load_dataset(f, validate=False)) == 100


def test_split_is_deterministic_and_stratified(tmp_path):
    f = tmp_path / "d"
    _write(f)
    df = load_dataset(f, validate=False)
    a, b = split_dataset(df), split_dataset(df)
    assert list(a.X_test_raw.index) == list(b.X_test_raw.index)
    assert len(a.X_test) == 20 and a.y_test.sum() == 4  # 20 % of 100, spam share kept
    assert set(a.X_train.index).isdisjoint(a.X_test.index)


@pytest.mark.skipif(not config.data_path().exists(), reason="dataset not downloaded")
def test_real_dataset_split_sizes_match_the_reported_results():
    s = split_dataset(load_dataset())
    assert (len(s.X_train), len(s.X_test)) == (4457, 1115)
    assert int(s.y_test.sum()) == 149
