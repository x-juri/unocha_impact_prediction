from __future__ import annotations

import math

from data_model import (
    AMOUNT_NUMERIC_COLUMN,
    CBPF_ALLOCATION_SOURCE_COLUMN,
    CBPF_BUDGET_NUMERIC_COLUMN,
    CBPF_COUNTRY_COLUMN,
    CBPF_DATA_PATH,
    CBPF_DURATION_NUMERIC_COLUMN,
    CBPF_ORGANIZATION_TYPE_COLUMN,
    CBPF_PROJECT_SECTOR_LIST_COLUMN,
    CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN,
    CERF_DATA_PATH,
    CATEGORICAL_FEATURES,
    CIRV_PREV_NUMERIC_COLUMN,
    TARGET_NUMERIC_COLUMN,
    budget_sensecheck,
    cbpf_sector_feature_columns,
    load_and_clean_data,
    load_and_clean_cbpf_projects,
    make_cbpf_prediction_frame,
    make_cerf_prediction_frame,
    predict_with_uncertainty,
    train_cbpf_model,
    train_cerf_model,
)


def main() -> None:
    df = load_and_clean_data(CERF_DATA_PATH)
    assert len(df) == 1024, f"Expected 1,024 CERF rows, found {len(df):,}"
    assert df[TARGET_NUMERIC_COLUMN].notna().all(), "CERF CIRV - Inc has missing numeric values"
    assert df[AMOUNT_NUMERIC_COLUMN].notna().all(), "CERF totalAmountApproved has missing numeric values"

    bundle = train_cerf_model(df)
    sample = df.iloc[0]
    cerf_frame = make_cerf_prediction_frame(
        total_amount_approved=sample[AMOUNT_NUMERIC_COLUMN],
        emergency_type=sample[CATEGORICAL_FEATURES[0]],
        country=sample[CATEGORICAL_FEATURES[1]],
        project_sectors=sample[CATEGORICAL_FEATURES[2]],
    )
    prediction = predict_with_uncertainty(bundle, cerf_frame)
    assert math.isfinite(prediction.prediction), "CERF prediction is not finite"
    assert prediction.lower < prediction.upper, "CERF interval is invalid"

    cbpf = load_and_clean_cbpf_projects(CBPF_DATA_PATH)
    assert len(cbpf) == 11206, f"Expected 11,206 CBPF rows, found {len(cbpf):,}"
    assert cbpf[TARGET_NUMERIC_COLUMN].notna().all(), "CBPF CIRV - Inc has missing numeric values"
    assert cbpf[CBPF_BUDGET_NUMERIC_COLUMN].notna().all(), "CBPF Budget has missing numeric values"
    assert cbpf[CBPF_PROJECT_SECTOR_LIST_COLUMN].map(len).gt(0).all(), "CBPF project sectors did not parse"
    assert "Water, Sanitation and Hygiene (WASH)" in {
        sector for values in cbpf[CBPF_PROJECT_SECTOR_LIST_COLUMN] for sector in values
    }, "WASH sector with comma was not preserved"
    budget_profile = budget_sensecheck(cbpf)
    assert budget_profile["parsed_missing"] == 0, "CBPF budget parse introduced missing values"

    cbpf_bundle = train_cbpf_model(cbpf)
    cbpf_sample = cbpf.iloc[0]
    sector_columns = cbpf_sector_feature_columns(cbpf)
    cbpf_frame = make_cbpf_prediction_frame(
        sector_columns,
        allocation_source=cbpf_sample[CBPF_ALLOCATION_SOURCE_COLUMN],
        organization_type=cbpf_sample[CBPF_ORGANIZATION_TYPE_COLUMN],
        project_duration_months=cbpf_sample[CBPF_DURATION_NUMERIC_COLUMN],
        budget=cbpf_sample[CBPF_BUDGET_NUMERIC_COLUMN],
        total_people=cbpf_sample[CBPF_TOTAL_PEOPLE_NUMERIC_COLUMN],
        cirv_prev=cbpf_sample[CIRV_PREV_NUMERIC_COLUMN],
        project_sectors=cbpf_sample[CBPF_PROJECT_SECTOR_LIST_COLUMN],
    )
    cbpf_prediction = predict_with_uncertainty(cbpf_bundle, cbpf_frame)
    assert math.isfinite(cbpf_prediction.prediction), "CBPF prediction is not finite"
    assert cbpf_prediction.lower < cbpf_prediction.upper, "CBPF interval is invalid"

    print("CERF rows:", len(df))
    print("CERF target mean:", round(df[TARGET_NUMERIC_COLUMN].mean(), 4))
    print("CERF prediction sample:", round(prediction.prediction, 4))
    print("CERF interval:", round(prediction.lower, 4), round(prediction.upper, 4))
    print("CERF metrics:", {key: round(value, 4) for key, value in bundle.metrics.items()})
    print("CBPF rows:", len(cbpf))
    print("CBPF target mean:", round(cbpf[TARGET_NUMERIC_COLUMN].mean(), 4))
    print("CBPF budget sensecheck:", budget_profile)
    print("CBPF prediction sample:", round(cbpf_prediction.prediction, 4))
    print("CBPF interval:", round(cbpf_prediction.lower, 4), round(cbpf_prediction.upper, 4))
    print("CBPF metrics:", {key: round(value, 4) for key, value in cbpf_bundle.metrics.items()})


if __name__ == "__main__":
    main()
