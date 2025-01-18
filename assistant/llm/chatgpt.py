import os
import time
import random
import backoff

from typing import List, Dict, Union, Any
from openai import OpenAI, OpenAIError
from openai.types.chat.chat_completion import ChatCompletion

from .abstract_language_model import AbstractLanguageModel


class ChatGPT(AbstractLanguageModel):
    def __init__(
            self,
            config_path: str = '',
            model_name: str = 'chatgpt',
            cache: bool = False
    ) -> None:
        super().__init__(config_path, model_name, cache)
        self.config: Dict = self.config[model_name]

        self.model_id: str = self.config['model_id']

        self.prompt_token_cost: float = self.config['prompt_token_cost']
        self.response_token_cost: float = self.config['response_token_cost']

        self.temperature: float = self.config['temperature']

        self.max_tokens: int = self.config['max_tokens']

        self.stop: Union[str, List[str]] = self.config['stop']

        self.organization: str = self.config['organization']
        if self.organization == '':
            self.logger.warning('OPENAI_ORGANIZATION is not set')
        self.api_key: str = os.getenv('OPENAI_API_KEY', self.config['api_key'])
        if self.api_key == '':
            raise ValueError('OPENAI_API_KEY is not set.')

        self.client = OpenAI(
            api_key=self.api_key,
            organization=self.organization
        )

    def query(
            self,
            query: str,
            num_responses: int = 1
    ) -> Union[List[ChatCompletion], ChatCompletion]:
        if self.cache and query in self.response_cache:
            return self.response_cache[query]

        if num_responses == 1:
            response = self.chat([{'role': 'user', 'content': query}], num_responses)
        else:
            response = []
            next_try = num_responses
            total_num_attempts = num_responses
            while num_responses > 0 and total_num_attempts > 0:
                try:
                    assert next_try > 0
                    res = self.chat([{'role': 'user', 'content': query}], next_try)
                    response.append(res)
                    num_responses -= next_try
                    next_try = min(num_responses, next_try)
                except Exception as e:
                    next_try = (next_try + 1) // 2
                    self.logger.warning(
                        f"Error in chatgpt: {e}, trying again with {next_try} samples"
                    )
                    time.sleep(random.randint(1, 3))
                    total_num_attempts -= 1

        if self.cache:
            self.response_cache[query] = response
        return response

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def chat(
            self,
            messages: List[Dict],
            num_responses: int = 1
    ) -> ChatCompletion:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            n=num_responses,
            stop=self.stop,
        )

        self.prompt_tokens += response.usage.prompt_tokens
        self.completion_tokens += response.usage.completion_tokens
        prompt_tokens_k = float(self.prompt_token_cost) / 1000.0
        completion_tokens_k = float(self.completion_tokens) / 1000.0
        self.cost = (
            self.prompt_token_cost * prompt_tokens_k
            + self.response_token_cost * completion_tokens_k
        )
        self.logger.info(
            f"This is the response from chatgpt: {response}"
            f"\nThis is the the cost of the response: {self.cost}"
        )
        return response

    def get_response_texts(
            self,
            query_response: Union[List[ChatCompletion], ChatCompletion]
    ) -> List[str]:
        if not isinstance(query_response, List):
            query_response = [query_response]

        return [
            choice.message.content
            for response in query_response
            for choice in response.choices
        ]

