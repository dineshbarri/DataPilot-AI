import pandas as pd
import pytest

from datapilot.analyst import (
    ai_context,
    build_data_dictionary,
    gemini_dataset_summary,
    inspect_dataset,
)


def test_immediate_profile_and_target_ranking():
    frame = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4],
            "age": [24, None, 39, 41],
            "churn": ["no", "no", "yes", "yes"],
        }
    )
    profile = inspect_dataset(frame)
    assert profile["brief"].rows == 4
    assert profile["brief"].missing_cells == 1
    assert profile["targets"][0]["column"] == "churn"


def test_dictionary_and_ai_context_redact_pii():
    frame = pd.DataFrame(
        {"email": ["a@x.com", "b@x.com"], "revenue": [99.0, 101.0], "secret": ["x", "y"]}
    )
    profile = inspect_dataset(frame)
    dictionary = build_data_dictionary(frame).set_index("column")
    context = ai_context(frame, profile, ["secret"])
    assert "Potential PII" in dictionary.loc["email", "issues"]
    assert context["sample"] == [{"revenue": 99.0}, {"revenue": 101.0}]
    assert set(context["excluded_columns"]) == {"email", "secret"}


def test_gemini_requires_a_key():
    frame = pd.DataFrame({"x": [1, 2], "target": [0, 1]})
    with pytest.raises(ValueError, match="API key"):
        gemini_dataset_summary(frame, inspect_dataset(frame), "", "gemini", [])
