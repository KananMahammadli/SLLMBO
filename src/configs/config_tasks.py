from configs.config_enum import DATASET_NAMES
from configs.datasets.adult_census import ADULT_CENSUS_METADATA
from configs.datasets.bike_sharing import BIKE_SHARING_METAADATA
from configs.datasets.concrete_strength import CONCRETE_STRENGTH_METADATA
from configs.datasets.cover_type import COVER_TYPE_METADATA
from configs.datasets.energy import ENERGY_METADATA
from configs.datasets.gas_drift import GAS_DRIFT_METADATA
from configs.datasets.m5 import M5_METADATA


TASKS_METADATA = {
    DATASET_NAMES.gas_drift.name: GAS_DRIFT_METADATA,
    DATASET_NAMES.cover_type.name: COVER_TYPE_METADATA,
    DATASET_NAMES.adult_census.name: ADULT_CENSUS_METADATA,
    DATASET_NAMES.bike_sharing.name: BIKE_SHARING_METAADATA,
    DATASET_NAMES.concrete_strength.name: CONCRETE_STRENGTH_METADATA,
    DATASET_NAMES.energy.name: ENERGY_METADATA,
    DATASET_NAMES.m5.name: M5_METADATA,
}
