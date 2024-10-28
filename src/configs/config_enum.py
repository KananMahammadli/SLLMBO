import enum

OPTIMIZATION_DIRECTIONS = enum.Enum("optimization_directions", ["minimize", "maximize"])
ML_MODEL_NAMES = enum.Enum("model_names", ["lightgbm", "xgboost"])

LLM_NAMES = enum.Enum(
    "llm_names",
    {
        "gpt_3_5_turbo": "gpt-3.5-turbo",
        "gpt_4o": "gpt-4o",
        "gemini_1_5_flash": "gemini-1.5-flash",
        "claude_3_5_sonnet_20240620": "claude-3-5-sonnet-20240620",
    },
)
PROBLEM_TYPES = enum.Enum("problem_types", ["classification", "regression"])
DATASET_NAMES = enum.Enum(
    "dataset_names",
    [
        "gas_drift",
        "cover_type",
        "adult_census",
        "bike_sharing",
        "concrete_strength",
        "energy",
        "m5",
    ],
)
METRICS = enum.Enum(
    "metrics",
    [
        "f1_weighted",
        "neg_mean_absolute_error",
    ],
)
SLLMBO_METHODS = enum.Enum(
    "sllmbo_methods",
    [
        "sllmbo_fully_llm_with_intelligent_summary",
        "sllmbo_fully_llm_with_langchain",
        "sllmbo_llm_tpe",
    ],
)
LLM_TPE_INIT_METHODS = enum.Enum(
    "llm_tpe_init_methods",
    [
        "llm",
        "random",
    ],
)
OPTIMIZATION_METHODS = enum.Enum(
    "optimization_methods",
    [
        "sllmbo",
        "optuna",
        "hyperopt",
    ],
)
