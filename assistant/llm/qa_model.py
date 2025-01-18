import os
import torch
import logging
import getpass

from openai import OpenAI
from abc import ABC, abstractmethod
from transformers import T5ForConditionalGeneration, T5Tokenizer
from tenacity import retry, stop_after_attempt, wait_random_exponential


class BaseQAModel(ABC):
    @abstractmethod
    def answer_question(self, context: str, question: str):
        pass


class GPT3QAModel(BaseQAModel):
    def __init__(self, model="text-davinci-003"):
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context: str, question: str, max_tokens: int = 150, stop_sequence=None):
        try:
            response = self.client.completions.create(
                prompt=f"using the following information {context}. Answer the following question in less than 5-7 words, if possible: {question}",
                temperature=0,
                max_tokens=max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=stop_sequence,
                model=self.model,
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(e)
            return ""


class GPT3TurboQAModel(BaseQAModel):
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(self, context: str, question: str, max_tokens: int = 150, stop_sequence=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answer Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Give the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context: str, question: str, max_tokens: int = 150, stop_sequence=None):
        try:
            return self._attempt_answer_question(
                context, question, max_tokens, stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class GPT4QAModel(BaseQAModel):
    def __init__(self, model="gpt-4"):
        self.model = model
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def _attempt_answer_question(self, context: str, question: str, max_tokens: int = 150, stop_sequence=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are Question Answering Portal"},
                {
                    "role": "user",
                    "content": f"Given Context: {context} Given the best full answer amongst the option to question {question}",
                },
            ],
            temperature=0,
        )

        return response.choices[0].message.content.strip()

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def answer_question(self, context: str, question: str, max_tokens: int = 150, stop_sequence=None):
        try:
            return self._attempt_answer_question(
                context, question, max_tokens, stop_sequence
            )
        except Exception as e:
            print(e)
            return e


class UnifiedQAModel(BaseQAModel):
    def __init__(self, model_name: str = "allenai/unifiedqa-v2-t5-3b-1363200"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def run_model(self, input_string: str, **generator_args):
        input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)
        res = self.model.generate(input_ids, **generator_args)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

    def answer_question(self, context: str, question: str):
        input_string = question + " \\n" + context
        output = self.run_model(input_string)
        return output[0]
