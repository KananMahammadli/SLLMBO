import optuna
from optuna.samplers import TPESampler, BaseSampler
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
import random
import json
from langchain_anthropic import ChatAnthropic
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
import re
from typing import Dict, Any
import time

from optimizers.util import api_rate_limiter

load_dotenv()


class LLMSampler(BaseSampler):
    def __init__(
        self,
        llm_name,
        model_name,
        metric,
        direction,
        problem_description,
        search_space_dict,
    ):
        self.llm_name = llm_name
        self.model_name = model_name
        self.metric = metric
        self.direction = direction
        self.problem_description = problem_description
        self.search_space_dict = search_space_dict
        self.memory = ConversationBufferMemory(return_messages=True)
        self.system_prompt = self._generate_system_prompt()
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        if self.llm_name.startswith("gpt"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is not set")
            return ChatOpenAI(
                model_name=self.llm_name,
                openai_api_key=api_key,  # type: ignore
            )

        elif self.llm_name.startswith("claude"):
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY is not set")
            return ChatAnthropic(
                model=self.llm_name,  # type: ignore
                anthropic_api_key=api_key,  # type: ignore
            )

        elif self.llm_name.startswith("gemini"):
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY is not set")
            return ChatGoogleGenerativeAI(
                model=self.llm_name,
                api_key=api_key,  # type: ignore
            )
        else:
            raise ValueError("Invalid LLM name")

    def _generate_system_prompt(self):

        with open(
            os.path.abspath("src/configs/prompts/llm_tpe_sys_prompt.txt"), "r"
        ) as f:
            return f.read().format(
                model_name=self.model_name,
                problem_description=self.problem_description,
                metric=self.metric,
                direction=self.direction,
            )

    @api_rate_limiter()
    def _create_chain(self):
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                HumanMessagePromptTemplate.from_template("{human_input}"),
            ]
        )
        return LLMChain(llm=self.llm, prompt=prompt, memory=self.memory)

    @api_rate_limiter()
    def sample_relative(self, study, trial, search_space):
        if search_space == {}:
            return {}
        chain = self._create_chain()
        if chain is None:
            return self._fallback_sample_relative(search_space)

        current_ranges = self._get_param_ranges(search_space)

        if trial.number == 0:
            with open(
                os.path.abspath(
                    "src/configs/prompts/llm_tpe_sample_relative_init_prompt.txt"
                ),
                "r",
            ) as f:
                human_input = f.read().format(
                    search_space_dict=json.dumps(self.search_space_dict)
                )
        else:
            best_trial = study.best_trial if study.best_trial else None
            best_score = best_trial.value if best_trial else None
            best_params = best_trial.params if best_trial else {}

            with open(
                os.path.abspath(
                    "src/configs/prompts/llm_tpe_sample_relative_opt_prompt.txt"
                ),
                "r",
            ) as f:
                human_input = f.read().format(
                    best_score=best_score,
                    best_params=json.dumps(best_params),
                    current_ranges=json.dumps(current_ranges),
                    search_space_dict=json.dumps(self.search_space_dict),
                )

        raw_response = chain.run(human_input=human_input)
        if raw_response is None:
            return self._fallback_sample_relative(search_space)
        response = self._parse_llm_response(raw_response)

        params = {}
        if len(search_space) == 0:
            raise ValueError("Search space is empty")
        for param, param_distribution in search_space.items():
            if not isinstance(
                param_distribution,
                (
                    optuna.distributions.FloatDistribution,
                    optuna.distributions.IntDistribution,
                ),
            ):
                msg = f"Only supports float and int distributions, got {param_distribution}"
                raise NotImplementedError(msg)

            if param in response:
                params[param] = response[param]
        time.sleep(1)
        return params

    @api_rate_limiter()
    def sample_independent(self, study, trial, param_name, param_distribution):
        chain = self._create_chain()
        if chain is None:
            return self._fallback_sample_independent(param_distribution)

        param_range = self._get_single_param_range(param_name, param_distribution)

        if trial.number == 0:
            with open(
                os.path.abspath(
                    "src/configs/prompts/llm_tpe_sample_independent_init_prompt.txt"
                ),
                "r",
            ) as f:
                human_input = f.read().format(
                    param_name=param_name,
                    param_range=json.dumps(param_range),
                )
        else:
            best_trial = study.best_trial if study.best_trial else None
            best_score = best_trial.value if best_trial else None
            best_params = best_trial.params if best_trial else {}

            with open(
                os.path.abspath(
                    "src/configs/prompts/llm_tpe_sample_independent_opt_prompt.txt"
                ),
                "r",
            ) as f:
                human_input = f.read().format(
                    best_score=best_score,
                    best_params=json.dumps(best_params),
                    param_name=param_name,
                    param_range=json.dumps(param_range),
                )

        raw_response = chain.run(human_input=human_input)
        if raw_response is None:
            return self._fallback_sample_independent(param_distribution)
        response = self._parse_llm_response(raw_response)

        try:
            if isinstance(response, dict):
                return list(response.values())[0]
            else:
                return response
        except Exception as e:
            print(f"Raw response: {raw_response}")
            print(f"Parsed response: {response}")
            raise e

    def _fallback_sample_relative(self, search_space):
        # Implement a fallback sampling strategy
        return {
            param: self._fallback_sample_independent(dist)
            for param, dist in search_space.items()
        }

    def _fallback_sample_independent(self, distribution):
        if isinstance(distribution, optuna.distributions.IntDistribution):
            return distribution.low + (distribution.high - distribution.low) // 2
        elif isinstance(distribution, optuna.distributions.FloatDistribution):
            return (distribution.low + distribution.high) / 2
        else:
            raise NotImplementedError(
                f"Unsupported distribution type: {type(distribution)}"
            )

    def _get_param_ranges(self, search_space):
        return {
            name: self._get_single_param_range(name, dist)
            for name, dist in search_space.items()
        }

    def _get_single_param_range(self, name, distribution):
        if isinstance(distribution, optuna.distributions.IntDistribution):
            return {"type": "int", "low": distribution.low, "high": distribution.high}
        elif isinstance(distribution, optuna.distributions.FloatDistribution):
            return {"type": "float", "low": distribution.low, "high": distribution.high}
        elif isinstance(distribution, optuna.distributions.CategoricalDistribution):
            return {"type": "categorical", "choices": distribution.choices}
        else:
            raise ValueError(f"Unsupported distribution type for parameter {name}")

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

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
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

        return parsed_response

    def infer_relative_search_space(self, study, trial):
        return optuna.search_space.intersection_search_space(
            study.get_trials(deepcopy=False)
        )


class LLM_TPE_SAMPLER(BaseSampler):
    def __init__(self, langchain_sampler, seed=None, init_method="llm"):
        self.langchain_sampler = langchain_sampler
        self.tpe_sampler = TPESampler(seed=seed)
        self.rng = random.Random(seed)
        self.last_called_sampler = None
        self.init_method = init_method

    def sample_relative(self, study, trial, search_space):
        if trial.number == 0:
            if self.init_method == "llm":
                self.last_called_sampler = "langchain"
                time.sleep(1.5)
                return self.langchain_sampler.sample_relative(
                    study, trial, search_space
                )
            elif self.init_method == "random":
                params = self.tpe_sampler.sample_relative(study, trial, search_space)
                self._add_suggestion_to_memory(trial, params, "TPESampler")
                self.last_called_sampler = "tpe"
                return params
            else:
                raise ValueError(
                    f"Invalid init_method. Expected 'llm' or 'random', got {self.init_method}"
                )

        elif self.rng.random() < 0.5:
            self.last_called_sampler = "langchain"
            time.sleep(1.5)
            return self.langchain_sampler.sample_relative(study, trial, search_space)
        else:
            params = self.tpe_sampler.sample_relative(study, trial, search_space)
            self._add_suggestion_to_memory(trial, params, "TPESampler")
            self.last_called_sampler = "tpe"
            return params

    def sample_independent(self, study, trial, param_name, param_distribution):
        if self.rng.random() < 0.5 or trial.number == 0:
            self.last_called_sampler = "langchain"
            time.sleep(1.5)
            return self.langchain_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )

        else:
            value = self.tpe_sampler.sample_independent(
                study, trial, param_name, param_distribution
            )
            self._add_suggestion_to_memory(trial, {param_name: value}, "TPESampler")
            self.last_called_sampler = "tpe"
            return value

    def _add_suggestion_to_memory(self, trial, params, source):
        human_message = f"{source}, trial {trial.number}"
        ai_message = f"Parameters suggested by {source}: {json.dumps(params)}"
        self.langchain_sampler.memory.chat_memory.add_user_message(human_message)
        self.langchain_sampler.memory.chat_memory.add_ai_message(ai_message)

    def tell(self, study, trial):
        result = f"Trial {trial.number} finished with value: {trial.value} and parameters: {json.dumps(trial.params)}"  # noqa: E501
        self.langchain_sampler.memory.chat_memory.add_user_message(
            f"Trial {trial.number} completed"
        )
        self.langchain_sampler.memory.chat_memory.add_ai_message(result)
        self.tpe_sampler.tell(study, trial)  # type: ignore

    def infer_relative_search_space(self, study, trial):
        if self.last_called_sampler == "tpe":
            search_space = self.tpe_sampler.infer_relative_search_space(study, trial)
        else:
            search_space = self.langchain_sampler.infer_relative_search_space(
                study, trial
            )
        return search_space
