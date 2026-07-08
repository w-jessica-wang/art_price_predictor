# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Docs-don't-drift test — the headline numbers quoted in README.md and model_card.md
must match the source-of-truth artifacts (models/meta.json `display` block and
models_eval/walk_forward.json). This is the guard the reviewer asked for: it fails
CI the moment a metric in prose drifts from the trained artifact.

Auto-skips when artifacts aren't present (fresh clone without a trained model).
"""
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_META = os.path.join(ROOT, "models", "meta.json")
_WF = os.path.join(ROOT, "models_eval", "walk_forward.json")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(_META) and os.path.exists(_WF)),
    reason="trained artifacts not present — run train + walk_forward_eval first",
)


def _read(name):
    with open(os.path.join(ROOT, name)) as f:
        return f.read()


def test_readme_matches_artifacts():
    d = json.load(open(_META))["display"]
    wf = json.load(open(_WF))["pooled"]
    readme = _read("README.md")
    card = _read("model_card.md")

    # dataset counts (deployed model's training set)
    assert f"{d['n_total']:,}" in readme, f"n_total {d['n_total']:,} missing from README"
    assert f"{d['n_priced']:,}" in readme, f"n_priced {d['n_priced']:,} missing from README"
    # secondary holdout n
    assert f"{d['n_holdout']:,}" in readme and f"{d['n_holdout']:,}" in card

    # walk-forward headline (primary) — model & estimate MedAPE appear in both docs
    for doc in (readme, card):
        assert f"{wf['medape_model']}%" in doc, f"WF MedAPE {wf['medape_model']}% missing"
        assert f"{wf['medape_est']}%" in doc, f"WF est MedAPE {wf['medape_est']}% missing"
    # walk-forward OOS size + Christie's OOS count
    assert f"{wf['n']:,}" in readme
    assert f"{json.load(open(_WF))['christies_test_lots']:,}" in readme


def test_data_card_matches_artifacts():
    d = json.load(open(_META))["display"]
    wf_all = json.load(open(_WF))
    data_card = _read("DATA.md")
    # dataset counts + holdout n + Christie's OOS count quoted in the data card
    assert f"{d['n_total']:,}" in data_card, "n_total missing from DATA.md"
    assert f"{d['n_priced']:,}" in data_card, "n_priced missing from DATA.md"
    assert f"{d['n_holdout']:,}" in data_card, "holdout n missing from DATA.md"
    assert f"{wf_all['christies_test_lots']:,}" in data_card, "Christie's OOS count missing from DATA.md"


def test_no_known_stale_tokens():
    # numbers from earlier (pre-wine / pre-fix) drafts that must never reappear
    stale = ["155,776", "133,752", "29.7%", "0.884", "6,464", "19,481", "4,894", "82.5%"]
    for doc in ("README.md", "model_card.md", "TECHNICAL_REPORT.md", "DATA.md"):
        s = _read(doc)
        for tok in stale:
            assert tok not in s, f"stale token {tok!r} reappeared in {doc}"


if __name__ == "__main__":
    test_readme_matches_artifacts()
    test_data_card_matches_artifacts()
    test_no_known_stale_tokens()
    print("✅ docs match artifacts; no stale tokens")
