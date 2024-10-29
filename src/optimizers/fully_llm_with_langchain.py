import json
from typing import Dict, Any, List, Tuple
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_anthropic import ChatAnthropic
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from sklearn.model_selection import cross_val_score
import os
from huggingface_hub import login
import re
from dotenv import load_dotenv

load_dotenv()


class LangchainOptimizer:
    def __init__(self, model_name):
        self.model_name = model_name
        self.llm = self._initialize_llm()
        self.memory = ConversationBufferMemory(return_messages=True)
        self.system_message = ""

    def _initialize_llm(self):
        if self.model_name.startswith("gpt"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            return ChatOpenAI(
                model_name=self.model_name,  # type: ignore
                openai_api_key=api_key,
            )

        elif self.model_name.startswith("claude"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            return ChatAnthropic(
                model=self.model_name,  # type: ignore
                anthropic_api_key=api_key,  # type: ignore
            )

        elif self.model_name.startswith("gemini"):
            print(f"model_name: {self.model_name}")
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set")
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                api_key=api_key,  # type: ignore
            )
        else:
            # Hugging Face models
            api_key = os.getenv("HUGGINGFACE_API_KEY")
            if not api_key:
                raise ValueError("HUGGINGFACE_API_KEY is not set")
            login(token=api_key)
            pipe = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                max_length=2000,
                truncation=True,
            )
            return HuggingFacePipeline(pipeline=pipe)

    def _create_chain(self, system_message: str):
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(system_message),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )
        return LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)

    def initialize_optimization(
        self,
        model_name: str,
        problem_description: str,
        metric: str,
        direction: str,
    ) -> Dict[str, Any]:
        with open(
            os.path.abspath(
                "src/configs/prompts/fully_llm_with_langchain_sys_prompt.txt"
            ),
            "r",
        ) as f:
            self.system_message = f.read().format(
                model_name=model_name,
                problem_description=problem_description,
                metric=metric,
                direction=direction,
            )

        chain = self._create_chain(self.system_message)

        with open(os.path.abspath("src/configs/prompts/init_prompt.txt"), "r") as f:
            init_prompt = f.read()

        response = chain.run(init_prompt)
        return self._parse_llm_response(response, {}, is_initialization=True)

    def get_next_parameters_and_ranges(
        self,
        current_ranges: Dict[str, Any],
        model_name: str,
        metric: str,
        direction: str,
        best_score: float,
        best_params: Dict[str, Any],
    ) -> Dict[str, Any]:
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

        chain = self._create_chain(self.system_message)
        response = chain.run(prompt)
        return self._parse_llm_response(response, best_params, is_initialization=False)

    def _parse_llm_response(
        self, response: str, default_params: Dict[str, Any], is_initialization: bool
    ) -> Dict[str, Any]:
        try:
            # First, try to parse the entire response as JSON
            parsed_response = json.loads(response)
        except json.JSONDecodeError:
            # If that fails, try to extract JSON-like content from the response
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                try:
                    parsed_response = json.loads(json_match.group())
                except json.JSONDecodeError:
                    parsed_response = self._extract_key_value_pairs(response)
            else:
                parsed_response = self._extract_key_value_pairs(response)

        if not isinstance(parsed_response, dict):
            raise ValueError("Response is not a dictionary")

        if is_initialization:
            if (
                "param_ranges" not in parsed_response
                or "initial_params" not in parsed_response
            ):
                # Try to infer param_ranges and initial_params from the response
                param_ranges, initial_params = self._infer_params(parsed_response)
                return {
                    "param_ranges": param_ranges,
                    "initial_params": initial_params,
                }
        else:
            if (
                "update_param_ranges" not in parsed_response
                or "next_params" not in parsed_response
            ):
                # Try to infer update_param_ranges and next_params from the response
                update_param_ranges, next_params, reason = self._infer_next_params(
                    parsed_response, default_params
                )
                return {
                    "update_param_ranges": update_param_ranges,
                    "next_params": next_params,
                }

        return parsed_response

    def _extract_key_value_pairs(self, response: str) -> Dict[str, Any]:
        pairs = {}
        lines = response.split("\n")
        for line in lines:
            match = re.match(r'^\s*(["\']?)(\w+)\1\s*:\s*(.+)$', line)
            if match:
                key, value = match.group(2), match.group(3)
                try:
                    pairs[key] = json.loads(value)
                except json.JSONDecodeError:
                    pairs[key] = value.strip()
        return pairs

    def _infer_params(
        self, parsed_response: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        param_ranges = {}
        initial_params = {}
        for key, value in parsed_response.items():
            if isinstance(value, list):
                param_ranges[key] = value
                initial_params[key] = value[0] if len(value) > 0 else None
            elif isinstance(value, dict) and "min" in value and "max" in value:
                param_ranges[key] = [value["min"], value["max"]]
                initial_params[key] = (value["min"] + value["max"]) / 2
            else:
                param_ranges[key] = value
                initial_params[key] = value
        return param_ranges, initial_params

    def _infer_next_params(
        self, parsed_response: Dict[str, Any], default_params: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any], str]:
        update_param_ranges = False
        next_params = default_params.copy()
        reason = ""
        for key, value in parsed_response.items():
            if key.lower() == "update_param_ranges":
                update_param_ranges = bool(value)
            elif isinstance(value, (int, float, str, bool)):
                next_params[key] = value
        return update_param_ranges, next_params, reason


def optimize_fully_llm_with_langchain(
    X,
    y,
    model_name: str,
    estimator,
    cv,
    langchain_model_name: str,
    problem_description: str,
    direction: str,
    metric: str,
    max_n_iters_without_improvement: int,
    optimization_directions_enum,
    optmization_check_func,
    default_params: Dict[str, Any] = {},
    n_trials: int = 50,
) -> Tuple[Dict[str, Any], float, List[float]]:
    langchain_optimizer = LangchainOptimizer(langchain_model_name)

    # Initialization with retry logic
    initialization_try_count = 0
    max_initialization_tries = 3
    while initialization_try_count < max_initialization_tries:
        try:
            initialization = langchain_optimizer.initialize_optimization(
                model_name=model_name,
                problem_description=problem_description,
                metric=metric,
                direction=direction,
            )
            break
        except Exception as e:
            initialization_try_count += 1
            if initialization_try_count == max_initialization_tries:
                raise ValueError(
                    f"Failed to initialize after {max_initialization_tries} attempts: {str(e)}"
                )

    initial_params = initialization["initial_params"]
    current_ranges = initialization["param_ranges"]

    all_results = []
    best_cv_score = (
        1e6 if direction == optimization_directions_enum.minimize.name else -1e6
    )
    best_params = None
    n_iters_without_improvement = 0

    try:
        for iteration in range(1, n_trials + 1):
            try:
                model = estimator(**initial_params, **default_params)
                scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
                # if scores is negative, convert it to positive
                if scores.mean() < 0:
                    scores = -scores
                cv_score = scores.mean()
                cv_std = scores.std()

                if check_improvement_condition(
                    cv_score, best_cv_score, direction, optimization_directions_enum
                ):
                    best_cv_score = cv_score
                    best_params = initial_params.copy()
                    n_iters_without_improvement = 0
                else:
                    n_iters_without_improvement += 1

                if n_iters_without_improvement >= max_n_iters_without_improvement:
                    print(
                        f"Langchain, Stopping optimization. Reason: No improvement in cv_score for "
                        f"{max_n_iters_without_improvement} iterations"
                    )
                    break

                result = {
                    "iteration": iteration,
                    "params": initial_params,
                    "param_ranges": current_ranges,
                    "cv_score": cv_score,
                    "cv_std": cv_std,
                }
                all_results.append(result)

                best_result = optmization_check_func(
                    all_results, key=lambda x: x["cv_score"]
                )
                print(
                    f"Iteration {iteration}: Params: {result['params']}, CV Score: {result['cv_score']}, CV Std: {result['cv_std']}"  # noqa: E501
                )
                print(
                    f"Best Iteration: {best_result['iteration']}, Params: {best_result['params']}, CV Score: {best_result['cv_score']}, CV Std: {best_result['cv_std']}"  # noqa: E501
                )
                print()

                next_step = langchain_optimizer.get_next_parameters_and_ranges(
                    current_ranges=current_ranges,
                    model_name=model_name,
                    metric=metric,
                    direction=direction,
                    best_score=best_cv_score,
                    best_params=best_params,  # type: ignore
                )

                if next_step["update_param_ranges"]:
                    current_ranges = next_step["new_param_ranges"]

                initial_params = next_step["next_params"]

            except Exception as e:
                print(f"Error in iteration {iteration}: {str(e)}")
                continue

    except Exception as e:
        print(f"Optimization process interrupted: {str(e)}")

    history = [result["cv_score"] for result in all_results]

    return best_params, best_cv_score, history  # type: ignore


def check_improvement_condition(
    last_score: float, best_score: float, direction: str, optimization_directions
) -> bool:
    if direction == optimization_directions.minimize.name:
        return last_score <= best_score
    elif direction == optimization_directions.maximize.name:
        return last_score >= best_score
    else:
        raise ValueError(f"Invalid direction: {direction}")
