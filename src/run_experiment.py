from configs.config_dicts import DEFAULT_PARAMS_DICT, ESTIMATORS_DICT, OPTIMIZERS_DICT
from configs.config_enum import (
    DATASET_NAMES,
    LLM_NAMES,
    LLM_TPE_INIT_METHODS,
    ML_MODEL_NAMES,
    OPTIMIZATION_DIRECTIONS,
    OPTIMIZATION_METHODS,
    SLLMBO_METHODS,
)
from configs.config_params import (
    CV_VAR,
    DATASET_VAR,
    DIRECTION_VAR,
    MAX_N_ITERS_WITHOUT_IMPROVEMENT,
    METRIC_FUNC_VAR,
    METRIC_VAR,
    N_SUMMARIZE_ITER,
    NEPTUNE_PROJECT_NAME,
    OPTIMIZATION_CHECK_FUNC_VAR,
    N_TRIALS,
    PROBLEM_DESCRIPTION_VAR,
    PROBLEM_TYPE_VAR,
    RANDOM_STATE,
)
from configs.config_tasks import TASKS_METADATA
from optimizers.optuna_sampler import LLM_TPE_SAMPLER, LLMSampler
from optimizers.util import get_param_space
from utils import (
    run_single_experiment,
    report_to_neptune,
    visualize_hyperopt_history,
    visualize_llm_history,
)
import optuna
import argparse
import os


if __name__ == "__main__":
    # get dataset_name, ml_model_name, llm_model_name from user
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAMES.gas_drift.name,
        help="Name of the dataset to be used for the experiment",
        choices=[
            DATASET_NAMES.gas_drift.name,
            DATASET_NAMES.cover_type.name,
            DATASET_NAMES.adult_census.name,
            DATASET_NAMES.bike_sharing.name,
            DATASET_NAMES.concrete_strength.name,
            DATASET_NAMES.energy.name,
            DATASET_NAMES.m5.name,
        ],
    )
    parser.add_argument(
        "--ml_model",
        type=str,
        default=ML_MODEL_NAMES.lightgbm.name,
        help="Name of the Machine Learning model to be used for the experiment",
        choices=[
            ML_MODEL_NAMES.lightgbm.name,
            ML_MODEL_NAMES.xgboost.name,
        ],
    )
    parser.add_argument(
        "--llm",
        type=str,
        default=LLM_NAMES.gpt_3_5_turbo.name,
        help="Name of the LLM to be used for the experiment",
        choices=[
            LLM_NAMES.gpt_3_5_turbo.value,
            LLM_NAMES.gpt_4o.value,
            LLM_NAMES.gemini_1_5_flash.value,
            LLM_NAMES.claude_3_5_sonnet_20240620.value,
        ],
    )
    parser.add_argument(
        "--optimization_method",
        type=str,
        default=OPTIMIZATION_METHODS.sllmbo.name,
        help="Name of the optimization method to be used for the experiment",
        choices=[
            OPTIMIZATION_METHODS.sllmbo.name,
            OPTIMIZATION_METHODS.optuna.name,
            OPTIMIZATION_METHODS.hyperopt.name,
        ],
    )
    parser.add_argument(
        "--sllmbo_method",
        type=str,
        default=SLLMBO_METHODS.sllmbo_llm_tpe.name,
        help="Name of the SLLMBO method to be used for the experiment",
        choices=[
            SLLMBO_METHODS.sllmbo_fully_llm_with_intelligent_summary.name,
            SLLMBO_METHODS.sllmbo_fully_llm_with_langchain.name,
            SLLMBO_METHODS.sllmbo_llm_tpe.name,
        ],
    )
    parser.add_argument(
        "--llm_tpe_init_method",
        type=str,
        default=LLM_TPE_INIT_METHODS.llm.name,
        help="Name of the LLM TPE initialization method to be used for the experiment",
        choices=[
            LLM_TPE_INIT_METHODS.llm.name,
            LLM_TPE_INIT_METHODS.random.name,
        ],
    )
    parser.add_argument(
        "--log_to_neptune",
        action="store_true",
        help="Whether to log the experiment to Neptune",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to print the optimization results",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Number of iterations to wait for improvement before stopping the optimization",
    )

    args = parser.parse_args()
    dataset_name = args.dataset
    ml_model_name = args.ml_model
    llm_name = args.llm
    optimization_method_name = args.optimization_method
    sllmbo_method_name = args.sllmbo_method
    llm_tpe_init_method_name = args.llm_tpe_init_method
    log_to_neptune = args.log_to_neptune
    verbose = args.verbose
    patience = args.patience

    # get task metadata
    problem_metadata = TASKS_METADATA[dataset_name]
    X_train, X_test, y_train, y_test = problem_metadata[DATASET_VAR]
    cat_cols = X_train.select_dtypes(include=["category"]).columns
    problem_description = problem_metadata[PROBLEM_DESCRIPTION_VAR]
    metric = problem_metadata[METRIC_VAR]
    metric_func = problem_metadata[METRIC_FUNC_VAR]
    direction = problem_metadata[DIRECTION_VAR]
    optmization_check_func = problem_metadata[OPTIMIZATION_CHECK_FUNC_VAR]
    problem_type = problem_metadata[PROBLEM_TYPE_VAR]
    cv = problem_metadata[CV_VAR]

    default_params = DEFAULT_PARAMS_DICT[ml_model_name]
    if len(cat_cols) > 1 and ml_model_name == ML_MODEL_NAMES.xgboost.name:
        default_params["enable_categorical"] = True
    estimator = ESTIMATORS_DICT[ml_model_name][problem_type]
    optimizer_func = OPTIMIZERS_DICT[optimization_method_name]

    optimization_args = {
        "X": X_train,
        "y": y_train,
        "model_name": ml_model_name,
        "estimator": estimator,
        "cv": cv,
        "default_params": default_params,
        "n_trials": N_TRIALS,
        "metric": metric,
        "direction": direction,
    }

    # start the experiment
    if optimization_method_name == OPTIMIZATION_METHODS.optuna.name:
        # tuning with optuna optimizer
        experiment_name = f"{optimization_method_name}_{dataset_name}_{ml_model_name}"
        if patience is not None and patience > 0:
            experiment_name += f"_patience_{patience}"

        storage = (
            f"sqlite:///{os.path.abspath('src/results')}/{experiment_name}.db"
        )
        optimization_args |= {
            "study_name": experiment_name,
            "storage": storage,
            "random_state": RANDOM_STATE,
            "patience": patience,
        }

        best_params, best_score, best_study, runtime, best_test_score = (
            run_single_experiment(
                optimizer_func=optimizer_func,
                optimization_args=optimization_args,
                X_test=X_test,
                y_test=y_test,
                metric_func=metric_func,
            )
        )
        fig_history = optuna.visualization.plot_optimization_history(best_study)
        fig_history.update_layout(title=f"Optimization History for {experiment_name}")

    elif optimization_method_name == OPTIMIZATION_METHODS.hyperopt.name:
        # tuning with hyperopt optimizer
        experiment_name = f"{optimization_method_name}_{dataset_name}_{ml_model_name}"
        optimization_args |= {
            "random_state": RANDOM_STATE,
        }
        best_params, best_score, best_study, runtime, best_test_score = (
            run_single_experiment(
                optimizer_func=optimizer_func,
                optimization_args=optimization_args,
                X_test=X_test,
                y_test=y_test,
                metric_func=metric_func,
            )
        )

        fig_history = visualize_hyperopt_history(best_study)
        fig_history.update_layout(title=f"Optimization History for {experiment_name}")

    elif optimization_method_name == OPTIMIZATION_METHODS.sllmbo.name:
        if (
            sllmbo_method_name
            == SLLMBO_METHODS.sllmbo_fully_llm_with_intelligent_summary.name
        ):
            experiment_name = f"{sllmbo_method_name}_{dataset_name}_{ml_model_name}"
            optimization_args |= {
                "optimization_directions_enum": OPTIMIZATION_DIRECTIONS,
                "problem_description": problem_description,
                "n_summarize_iter": N_SUMMARIZE_ITER,
                "max_n_iters_without_improvement": MAX_N_ITERS_WITHOUT_IMPROVEMENT,
                "optmization_check_func": optmization_check_func,
            }

            best_params, best_score, best_study, runtime, best_test_score = (
                run_single_experiment(
                    optimizer_func=optimizer_func[sllmbo_method_name],
                    optimization_args=optimization_args,
                    X_test=X_test,
                    y_test=y_test,
                    metric_func=metric_func,
                )
            )

            fig_history = visualize_llm_history(
                all_iterations_scores=best_study, best_score=best_score
            )
            fig_history.update_layout(
                title=f"Optimization History for {experiment_name}"
            )

        elif sllmbo_method_name == SLLMBO_METHODS.sllmbo_fully_llm_with_langchain.name:
            experiment_name = (
                f"{sllmbo_method_name}_{llm_name}_{dataset_name}_{ml_model_name}"
            )
            optimization_args |= {
                "langchain_model_name": llm_name,
                "problem_description": problem_description,
                "max_n_iters_without_improvement": MAX_N_ITERS_WITHOUT_IMPROVEMENT,
                "optimization_directions_enum": OPTIMIZATION_DIRECTIONS,
                "optmization_check_func": optmization_check_func,
            }

            best_params, best_score, best_study, runtime, best_test_score = (
                run_single_experiment(
                    optimizer_func=optimizer_func[sllmbo_method_name],
                    optimization_args=optimization_args,
                    X_test=X_test,
                    y_test=y_test,
                    metric_func=metric_func,
                )
            )
            fig_history = visualize_llm_history(
                all_iterations_scores=best_study,
                best_score=best_score,
            )
            fig_history.update_layout(
                title=f"Optimization History for {experiment_name}"
            )

        elif sllmbo_method_name == SLLMBO_METHODS.sllmbo_llm_tpe.name:
            langchain_sampler = LLMSampler(
                llm_name=llm_name,
                model_name=ml_model_name,
                metric=metric,
                direction=direction,
                problem_description=problem_description,
                search_space_dict=get_param_space(ml_model_name),
            )
            hybrid_sampler = LLM_TPE_SAMPLER(
                langchain_sampler=langchain_sampler,
                seed=RANDOM_STATE,
                init_method=llm_tpe_init_method_name,
            )

            experiment_name = f"{sllmbo_method_name}_{llm_name}_{dataset_name}_{ml_model_name}_{llm_tpe_init_method_name}_init"  # noqa: E501
            if patience is not None and patience > 0:
                experiment_name += f"_patience_{patience}"

            storage = (
                f"sqlite:///{os.path.abspath('src/results')}/{experiment_name}.db"
            )
            optimization_args |= {
                "study_name": experiment_name,
                "storage": storage,
                "random_state": RANDOM_STATE,
                "sampler": hybrid_sampler,
                "patience": patience,
            }

            best_params, best_score, best_study, runtime, best_test_score = (
                run_single_experiment(
                    optimizer_func=optimizer_func[sllmbo_method_name],
                    optimization_args=optimization_args,
                    X_test=X_test,
                    y_test=y_test,
                    metric_func=metric_func,
                )
            )

            fig_history = optuna.visualization.plot_optimization_history(best_study)
            fig_history.update_layout(
                title=f"Optimization History for {experiment_name}"
            )
        else:
            raise ValueError(f"Unknown SLLMBO method: {sllmbo_method_name}")

    else:
        raise ValueError(f"Unknown Optimization method: {optimization_method_name}")

    if verbose:
        print(
            f"best params: {best_params}, best score: {best_score}, best test score: {best_test_score}"  # noqa: E501
        )

    if log_to_neptune:
        report_to_neptune(
            project=NEPTUNE_PROJECT_NAME,
            dataset_name=dataset_name,
            model_name=ml_model_name,
            best_params=best_params,
            best_cv_score=best_score,
            best_test_score=best_test_score,
            runtime=runtime,
            fig_history=fig_history,
            experiment_name=experiment_name,
        )
