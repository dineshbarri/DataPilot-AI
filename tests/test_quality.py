import pandas as pd

from datapilot.quality import audit_quality, build_profile, drift_report
from datapilot.schemas import TaskType


def test_profile_and_quality_flags():
    frame = pd.DataFrame(
        {
            "customer_id": [1, 2, 3, 4, 5, 6],
            "feature": [1.0, None, 3.0, 4.0, 5.0, 100.0],
            "target": ["yes", "yes", "yes", "yes", "yes", "no"],
        }
    )
    profile = build_profile(frame, "target")
    issues, evidence = audit_quality(frame, profile)
    assert profile.task_type == TaskType.classification
    assert profile.missing_cells == 1
    assert evidence
    assert {"MISSING_VALUES", "LEAKAGE_RISK"}.issubset({item.code for item in issues})


def test_numeric_drift_report():
    reference = pd.DataFrame({"x": list(range(100))})
    current = pd.DataFrame({"x": list(range(100, 200))})
    report = drift_report(reference, current)
    assert report[0]["column"] == "x"
    assert report[0]["status"] == "high"
