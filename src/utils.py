import plotly.graph_objects as go
import neptune
import time
from evaluation import evaluate_predictions


def visualize_hyperopt_history(trials, status_colors=None, title="Loss History"):
    if status_colors is None:
        status_colors = {
            "new": "black",
            "running": "green",
            "ok": "blue",
            "fail": "red",
        }

    Ys, colors = zip(
        *[
            (y, status_colors[s])
            for y, s in zip(trials.losses(), trials.statuses())
            if y is not None
        ]
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(Ys))),
            y=Ys,
            mode="markers",
            marker=dict(color=colors),
            name="Trials",
        )
    )

    best_err = trials.average_best_error()
    print("avg best error:", best_err)

    fig.add_trace(
        go.Scatter(
            x=[0, len(Ys) - 1],
            y=[best_err, best_err],
            mode="lines",
            line=dict(dash="dash", color="green"),
            name="Best Error",
        )
    )

    fig.update_layout(
        title=title, xaxis_title="Time", yaxis_title="Loss", showlegend=True
    )

    return fig


def visualize_llm_history(all_iterations_scores, best_score):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=list(range(len(all_iterations_scores))),
            y=all_iterations_scores,
            mode="lines",
            name="All Iterations Scores",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, len(all_iterations_scores) - 1],
            y=[best_score, best_score],
            mode="lines",
            line=dict(dash="dash", color="red"),
            name="Best Score",
        )
    )

    fig.update_layout(
        title="LLM History Visualization",
        xaxis_title="Iterations",
        yaxis_title="Scores",
        showlegend=True,
    )

    return fig


def report_to_neptune(
    project,
    experiment_name,
    dataset_name,
    model_name,
    best_params,
    best_cv_score,
    best_test_score,
    runtime,
    fig_history,
):
    run = neptune.init_run(
        project=project,
        name=experiment_name,
    )

    run["dataset"] = dataset_name
    run["model"] = model_name
    run["best_params"] = best_params
    run["best_cv_score"] = best_cv_score
    run["best_test_score"] = best_test_score
    run["runtime"] = runtime
    run["fig_history"] = fig_history


def run_optimization(optimize_func, **kwargs):
    start_time = time.time()
    results = optimize_func(**kwargs)
    end_time = time.time()
    runtime = end_time - start_time
    return results, runtime


def run_single_experiment(
    optimizer_func,
    optimization_args,
    X_test,
    y_test,
    metric_func,
):
    (best_params, best_score, best_study), runtime = run_optimization(
        optimizer_func, **optimization_args
    )

    best_test_score = evaluate_predictions(
        X_train=optimization_args["X"],
        X_test=X_test,
        y_train=optimization_args["y"],
        y_test=y_test,
        estimator=optimization_args["estimator"],
        best_params=best_params,
        default_params=optimization_args["default_params"],
        metric=metric_func,
    )
    return best_params, best_score, best_study, best_test_score, runtime
