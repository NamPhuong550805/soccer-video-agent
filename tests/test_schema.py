import numpy as np
import pytest

from sva.constants import ORDINAL_SCALES
from sva.evidence.builder import build_categorical, build_ordinal, build_positional, expected_value
from sva.models.base import mcq_prompt
from sva.models.scoring import (dist_from_logprobs, dist_from_sampling, entropy, margin_confidence,
                                parse_answer)
from sva.schema import Evidence, Sample


def make_sample(texts):
    return Sample("q14", 0, "q?", ["a.mp4"], list(texts), texts, list(texts)[0])


def test_evidence_dist_must_sum_to_one():
    with pytest.raises(ValueError):
        Evidence("t", "categorical", ["O1", "O2"], {"O1": 0.5, "O2": 0.6}, 0.1)


def test_evidence_keys_must_match():
    with pytest.raises(ValueError):
        Evidence("t", "categorical", ["O1", "O2"], {"O1": 0.5, "O3": 0.5}, 0.0)


def test_evidence_noncontiguous_keys():
    ev = Evidence("t", "categorical", ["O1", "O3"], {"O1": 0.85, "O3": 0.15}, 0.70)
    assert ev.pred == "O1"


def test_margin_confidence():
    assert margin_confidence({"O1": 0.85, "O3": 0.15}) == pytest.approx(0.70)
    assert margin_confidence({"O1": 1.0}) == 1.0


def test_failure():
    ev = Evidence.failure("t", ["O1", "O2", "O4"], "loi")
    assert ev.failed and ev.confidence == 0.0
    assert all(v == pytest.approx(1 / 3) for v in ev.dist.values())


def test_dist_from_logprobs_temperature():
    lp = np.array([-0.1, -1.0, -2.0, -3.0])
    keys = ["O1", "O2", "O3", "O4"]
    cold = dist_from_logprobs(lp, keys, 0.5)
    warm = dist_from_logprobs(lp, keys, 2.0)
    assert sum(cold.values()) == pytest.approx(1.0)
    assert cold["O1"] > dist_from_logprobs(lp, keys, 1.0)["O1"] > warm["O1"]
    assert entropy(cold) < entropy(warm)


def test_dist_from_logprobs_length_mismatch():
    with pytest.raises(ValueError):
        dist_from_logprobs([0.0, 1.0], ["O1", "O2", "O3"])


def test_parse_answer():
    texts = {"O1": "No", "O3": "Yes"}
    assert parse_answer("O3", ["O1", "O3"], texts) == "O3"
    assert parse_answer("O3. Yes", ["O1", "O3"], texts) == "O3"
    assert parse_answer("yes.", ["O1", "O3"], texts) == "O3"
    assert parse_answer("O2", ["O1", "O3"], texts) is None


def test_dist_from_sampling():
    texts = {"O1": "No", "O3": "Yes"}
    dist = dist_from_sampling(["O1", "O1", "no", "O3", "???"], ["O1", "O3"], texts, alpha=1.0)
    assert dist == pytest.approx({"O1": 4 / 6, "O3": 2 / 6})


def test_build_categorical():
    s = make_sample({"O1": "No", "O3": "Yes"})
    ev = build_categorical("vlm", s, np.array([-0.2, -1.5]))
    assert ev.option_keys == ["O1", "O3"] and ev.pred == "O1"
    assert ev.material_ref["clips"] == ["a.mp4"]


def test_build_ordinal():
    s = make_sample({"O1": "No card", "O2": "Yellow card", "O4": "Yellow card or red card"})
    ev = build_ordinal("vlm", s, np.log([0.1, 0.55, 0.35]), ORDINAL_SCALES["severity"])
    assert ev.ordinal_scale == {"O1": 0.0, "O2": 1.0, "O4": 1.5}
    assert expected_value(ev) == pytest.approx(0.55 * 1.0 + 0.35 * 1.5)


def test_build_ordinal_unknown_label():
    s = make_sample({"O1": "No", "O3": "Yes"})
    with pytest.raises(ValueError):
        build_ordinal("vlm", s, [0.0, 0.0], ORDINAL_SCALES["severity"])


def test_build_positional():
    s = make_sample({"O1": "The 1st one.", "O2": "The 2nd one.", "O3": "The 3rd one.", "O4": "The 4th one."})
    ev = build_positional("match", s, np.array([0.62, 0.55, 0.54, 0.50]))
    assert ev.claim_type == "positional" and ev.pred == "O1"
    assert ev.dist["O1"] > 0.5


def test_mcq_prompt_keeps_original_keys():
    s = make_sample({"O1": "No", "O3": "Yes"})
    p = mcq_prompt(s)
    assert "O1. No" in p and "O3. Yes" in p and "O2" not in p

    p = mcq_prompt(s, ["A", "B"])
    assert "A. No" in p and "B. Yes" in p and "option letter" in p
