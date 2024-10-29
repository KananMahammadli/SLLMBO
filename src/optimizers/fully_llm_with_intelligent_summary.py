import openai
import json
from sklearn.model_selection import cross_val_score
import os
from dotenv import load_dotenv

load_dotenv()


def check_improvement_condition(
    last_score, best_score, direction, optimization_directions
):
    if direction == optimization_directions.minimize.name:
        return last_score <= best_score
    elif direction == optimization_directions.maximize.name:

        return last_score >= best_score
    else:
        raise ValueError(f"Invalid direction: {direction}")


def optimize_fully_llm_with_intelligent_summary(
    X,
    y,
    model_name,
    estimator,
    cv,
    optimization_directions_enum,
    problem_description,
    direction,
    metric,
    n_summarize_iter,
    max_n_iters_without_improvement,
    optmization_check_func,
    default_params={},
    n_trials=50,
):
    # Initialize the LLM Optimizer
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set")
    llm_optimizer = LLMOptimizer(api_key=api_key, model="gpt-3.5-turbo")

    # Initialize the optimization process
    initialization_try_ccount = 0
    try:
        initialization = llm_optimizer.initialize_optimization(
            model_name=model_name,
            problem_description=problem_description,
            metric=metric,
            direction=direction,
            n_summarize_iter=n_summarize_iter,
        )
    except Exception as e:
        if initialization_try_ccount > 4:
            raise e
        initialization_try_ccount += 1
        initialization = llm_optimizer.initialize_optimization(
            model_name=model_name,
            problem_description=problem_description,
            metric=metric,
            direction=direction,
            n_summarize_iter=n_summarize_iter,
        )
    initial_params = initialization["initial_params"]
    current_ranges = initialization["param_ranges"]

    # Run the optimization loop
    all_results = []
    iteration = 1  # since we already have one iteration from initialization
    n_iters_without_improvement = 0

    # define initial best score and params
    best_cv_score = (
        1e6 if direction == optimization_directions_enum.minimize.name else -1e6
    )
    best_params = None
    try:
        while iteration < n_trials + 1:
            # Create and evaluate the model
            model = estimator(**initial_params, **default_params)

            scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
            # if scores is negative, convert it to positive
            if scores.mean() < 0:
                scores = -scores
            cv_score = scores.mean()
            cv_std = scores.std()

            # check if there is no improvement in cv_score for
            # max_n_iters_without_improvement iterations
            if check_improvement_condition(
                last_score=cv_score,
                best_score=best_cv_score,
                direction=direction,
                optimization_directions=optimization_directions_enum,
            ):
                best_cv_score = cv_score
                best_params = initial_params
                n_iters_without_improvement = 0

            else:
                n_iters_without_improvement += 1

            if n_iters_without_improvement >= max_n_iters_without_improvement:
                print(
                    f"LLM, Stopping optimization. Reason: No improvement in cv_score for "
                    f"{max_n_iters_without_improvement} iterations"
                )
                break

            # Save the results
            result = {
                "iteration": iteration,
                "params": initial_params,
                "param_ranges": current_ranges,
                "cv_score": cv_score,
                "cv_std": cv_std,
            }
            all_results.append(result)

            # print optuna like results, which is current iteration, params, cv_score, cv_std,
            # and best it, first find best cv_score and cv_std
            best_result = optmization_check_func(
                all_results, key=lambda x: x["cv_score"]
            )
            print(
                f"Iteration {iteration - 1}: Params: {result['params']}, CV Score: {result['cv_score']}, CV Std: {result['cv_std']}"  # noqa: E501
            )
            print(
                f"Best Iteration: {best_result['iteration']}, Params: {best_result['params']}, CV Score: {best_result['cv_score']}, CV Std: {best_result['cv_std']}"  # noqa: E501
            )
            print()

            # Get the next set of parameters and possibly update ranges
            next_step = llm_optimizer.get_next_parameters_and_ranges(
                current_ranges=current_ranges,
                model_name=model_name,
                metric=metric,
                direction=direction,
                best_score=best_cv_score,
                best_params=best_params,
                curr_iter=iteration,
                n_summarize_iter=n_summarize_iter,
            )

            if next_step["update_param_ranges"]:
                current_ranges = next_step["new_param_ranges"]

            initial_params = next_step["next_params"]
            iteration += 1

    except Exception as e:
        history = [result["cv_score"] for result in all_results]
        print(f"LLM, Error: {e}")
        return best_params, best_cv_score, history

    history = [result["cv_score"] for result in all_results]
    return best_params, best_cv_score, history


class LLMOptimizer:
    def __init__(self, api_key, model="gpt-4"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history = []
        self.system_message = ""
        self.initialization_iter = 0

    def summarize_history(self, max_entries=5):
        if len(self.conversation_history) > max_entries * 2:
            summary = []
            for i in range(0, max_entries * 2, 2):
                user_message = self.conversation_history[-max_entries * 2 + i][
                    "content"
                ]
                assistant_response = self.conversation_history[
                    -max_entries * 2 + i + 1
                ]["content"]
                summary.append(
                    {"role": "user", "content": f"Iteration {i//2}: {user_message}"}
                )
                summary.append(
                    {"role": "assistant", "content": f"Response: {assistant_response}"}
                )
            self.conversation_history = summary

    def _call_llm(self, user_message, curr_iter, n_summarize_iter):
        """
        Call the LLM with the given prompt and return the response.
        """
        messages = [
            {"role": "system", "content": self.system_message},
            *self.conversation_history,
            {"role": "user", "content": user_message},
        ]
        if curr_iter % n_summarize_iter != 0 or curr_iter == self.initialization_iter:
            try:
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages
                )
            except openai.BadRequestError as e:
                if "maximum context length" in str(e):
                    self.intelligent_summarize()
                    messages = [
                        {"role": "system", "content": self.system_message},
                        *self.conversation_history,
                        {"role": "user", "content": user_message},
                    ]
                    response = self.client.chat.completions.create(
                        model=self.model, messages=messages
                    )
                else:
                    raise e
        else:
            self.intelligent_summarize()
            messages = [
                {"role": "system", "content": self.system_message},
                *self.conversation_history,
                {"role": "user", "content": user_message},
            ]
            response = self.client.chat.completions.create(
                model=self.model, messages=messages
            )

        assistant_response = response.choices[0].message.content

        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_response}
        )

        return assistant_response

    def initialize_optimization(
        self,
        model_name,
        problem_description,
        metric,
        direction,
        n_summarize_iter,
    ):
        """
        Initialize the optimization process by explaining the problem to the LLM.

        """

        with open(
            os.path.abspath(
                "src/configs/prompts/fully_llm_with_intelligent_summary_sys_prompt.txt"
            ),
            "r",
        ) as f:
            self.system_message = f.read().format(
                model_name=model_name,
                problem_description=problem_description,
                metric=metric,
                direction=direction,
            )

        with open(os.path.abspath("src/configs/prompts/init_prompt.txt"), "r") as f:
            init_prompt = f.read()

        response = self._call_llm(
            user_message=init_prompt,
            curr_iter=self.initialization_iter,
            n_summarize_iter=n_summarize_iter,
        )
        return self._parse_llm_response(
            response=response,
            default_params={},
            curr_iter=self.initialization_iter,
        )

    def get_next_parameters_and_ranges(
        self,
        current_ranges,
        model_name,
        metric,
        direction,
        best_score,
        best_params,
        curr_iter,
        n_summarize_iter,
    ):
        """
        Get the next set of parameters to try and decide whether to update ranges.
        """
        with open(
            os.path.abspath(
                "src/configs/prompts/fully_llm_with_intelligent_summary_opt_prompt.txt"
            ),
            "r",
        ) as f:
            prompt = f.read().format(
                model_name=model_name,
                direction=direction,
                metric=metric,
                best_score=best_score,
                best_params=json.dumps(best_params),
                current_ranges=json.dumps(current_ranges),
            )

        response = self._call_llm(
            user_message=prompt,
            curr_iter=curr_iter,
            n_summarize_iter=n_summarize_iter,
        )
        return self._parse_llm_response(
            response=response,
            default_params=best_params,
            curr_iter=curr_iter,
        )

    def intelligent_summarize(self):
        with open(
            os.path.abspath("src/configs/prompts/summarization_prompt.txt"), "r"
        ) as f:
            prompt = f.read().format(
                conversation_history=json.dumps(self.conversation_history)
            )

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages  # type: ignore
            )
            summary_response = response.choices[0].message.content
            self.conversation_history = json.loads(summary_response)  # type: ignore
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Error: Invalid JSON response from LLM during summarization. Keeping original history. {e}"  # noqa: E501
            )

        except Exception as e:
            raise ValueError(
                f"Error during summarization: {e}. Keeping original history."
            )

    def _parse_llm_response(self, response, default_params, curr_iter):
        try:
            parsed_response = json.loads(response)
            if not isinstance(parsed_response, dict):
                raise ValueError("Response is not a dictionary")
            if curr_iter == self.initialization_iter:
                if (
                    "param_ranges" not in parsed_response
                    or "initial_params" not in parsed_response
                ):
                    raise ValueError("Response is missing required keys")

            else:
                if (
                    "update_param_ranges" not in parsed_response
                    or "next_params" not in parsed_response
                ):
                    raise ValueError("Response is missing required keys")

            return parsed_response
        except (json.JSONDecodeError, ValueError) as e:
            if curr_iter == self.initialization_iter:
                raise ValueError(f"Error parsing LLM response: {e}")
            else:
                return {"update_param_ranges": False, "next_params": default_params}
