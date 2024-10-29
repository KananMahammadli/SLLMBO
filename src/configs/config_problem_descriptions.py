from configs.config_enum import DATASET_NAMES


GAS_DRIFT_PROBLEM_DESC = """
We are working with a tabular classification dataset for gas drift and there are 6 classes.
there is only numerical features.
"""

COVER_TYPE_PROBLEM_DESC = """
We are working with a tabular classification dataset for
forest cover type prediction based on numeric features.
"""

ADULT_CENSUS_PROBLEM_DESC = """
We are working with a tabular classification dataset,
goal is to income group of person based on age, education and working hours.
"""

BIKE_SHARING_PROBLEM_DESC = """
We are working with a tabular regression dataset
for hourly demand based on numeric datetime and weather features.
"""

CONCRETE_STRENGTH_PROBLEM_DESC = """
We are working with a tabular regression dataset for cement strength based on numeric features.
"""

ENERGY_PROBLEM_DESC = """
We are working with a tabular regression dataset for energy consumption based on numeric features.
"""

M5_PROBLEM_DESC = """
We are working with a tabular regression dataset for daily demand forecasting for
multiple items at different stores. We have categorical columns such as item, department,
store ids, date related features, lag and rolling features of sales and selling prices of items.
In total we have 90 unique time series data, which we train together with global model.
"""


PROBLEM_DESCRIPTIONS = {
    DATASET_NAMES.gas_drift.name: GAS_DRIFT_PROBLEM_DESC,
    DATASET_NAMES.cover_type.name: COVER_TYPE_PROBLEM_DESC,
    DATASET_NAMES.adult_census.name: ADULT_CENSUS_PROBLEM_DESC,
    DATASET_NAMES.bike_sharing.name: BIKE_SHARING_PROBLEM_DESC,
    DATASET_NAMES.concrete_strength.name: CONCRETE_STRENGTH_PROBLEM_DESC,
    DATASET_NAMES.energy.name: ENERGY_PROBLEM_DESC,
    DATASET_NAMES.m5.name: M5_PROBLEM_DESC,
}
