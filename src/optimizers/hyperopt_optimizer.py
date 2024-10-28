from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
from optimizers.util import get_param_space
import numpy as np


def optimize_hyperopt(
    X,
    y,
    model_name,
    estimator,
    cv,
    default_params={},
    n_trials=50,
    metric="f1_weighted",
    direction="maximize",
    random_state=42,
):
    param_space = get_param_space(model_name)
    hyperopt_space = {}

    for param, (low, high) in param_space.items():
        if param in ["num_leaves", "max_depth", "n_estimators", "min_child_samples"]:
            hyperopt_space[param] = hp.quniform(param, low, high, 1)
        else:
            hyperopt_space[param] = hp.uniform(param, low, high)

    def objective(params):

        params = {
            key: (
                int(value)
                if key
                in ["num_leaves", "max_depth", "n_estimators", "min_child_samples"]
                else float(value)
            )
            for key, value in params.items()
        }
        all_params = {**params, **default_params}
        model = estimator(**all_params)

        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)

        if scores.mean() < 0:
            scores = -scores

        return {
            "loss": -scores.mean() if direction == "maximize" else scores.mean(),
            "status": STATUS_OK,
        }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=hyperopt_space,
        algo=tpe.suggest,
        max_evals=n_trials,
        trials=trials,
        rstate=np.random.default_rng(random_state),
    )

    best_params = {
        key: (
            int(value)
            if key
            in [
                "num_leaves",
                "max_depth",
                "n_estimators",
                "min_child_samples",
            ]
            else float(value)
        )
        for key, value in best.items()
    }
    best_score = (
        -trials.best_trial["result"]["loss"]
        if direction == "maximize"
        else trials.best_trial["result"]["loss"]
    )

    return best_params, best_score, trials
