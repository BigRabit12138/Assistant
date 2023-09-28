import asyncio

from hugchat import hugchat
from hugchat.login import Login
from langchain.llms.base import LLM
from typing import List, Optional, Any
from llm.base_gpt_api import BaseGPTAPI
from langchain.callbacks.manager import CallbackManagerForLLMRun

import global_var


class HuggingChatForLangchain(LLM):

    email = None
    passwd = None

    if email is None:
        email = global_var.email
    if passwd is None:
        passwd = global_var.passwd

    sign = Login(email, passwd)
    cookies = sign.login()
    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())

    def generate_response(self,
                          text,
                          temperature=.9,
                          top_p=.95,
                          repetition_penalty=1.2,
                          top_k=50,
                          truncate=1024,
                          watermark=False,
                          max_new_tokens=1024,
                          stop=None,
                          return_full_text=False,
                          stream=True,
                          use_cache=False,
                          is_retry=False,
                          retry_count=5) -> str:
        if stop is None:
            stop = ['</s>']
        resp = self.chatbot.chat(text,
                                 temperature,
                                 top_p,
                                 repetition_penalty,
                                 top_k,
                                 truncate,
                                 watermark,
                                 max_new_tokens,
                                 stop,
                                 return_full_text,
                                 stream,
                                 use_cache,
                                 is_retry,
                                 retry_count)
        return resp

    @property
    def _llm_type(self) -> str:
        return "huggingchat"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        return self.generate_response(text=prompt)


class HuggingChatForMetaGPT(BaseGPTAPI):
    def __init__(self, email: str = None, passwd: str = None):
        if email is None:
            email = global_var.email
        if passwd is None:
            passwd = global_var.passwd

        self.sign = Login(email, passwd)
        self.cookies = self.sign.login()
        self.chatbot = hugchat.ChatBot(cookies=self.cookies.get_dict())
        self.chatbot.switch_llm(1)

    def completion(self, messages: list[dict]):
        msg = '\n'.join([message['role'] + ':' + message['content'] for message in messages])
        resp = self.chatbot.chat(text=msg,
                                 temperature=0.9,
                                 top_p=0.95,
                                 repetition_penalty=1.2,
                                 top_k=50,
                                 truncate=1024,
                                 watermark=False,
                                 max_new_tokens=1024,
                                 stop=['</s>'],
                                 return_full_text=False,
                                 stream=True,
                                 use_cache=False,
                                 is_retry=False,
                                 retry_count=3)
        resp = {
            'choices': [
                {
                    'message': {
                        'content': resp
                    }
                }
            ]
        }

        return resp

    async def acompletion(self, messages: list[dict]):
        loop = asyncio.get_event_loop()
        msg = '\n'.join([message['role'] + ':' + message['content'] for message in messages])
        future = loop.run_in_executor(None, self.chatbot.chat, msg, 0.9, 0.95,
                                      1.2, 50, 2048, False, 2048, ['</s>'], False, True, False,
                                      False, 3)
        resp = await future
        resp = {
            'choices': [
                {
                    'message': {
                        'content': resp
                    }
                }
            ]
        }

        return resp

    async def acompletion_text(self, messages: list[dict], stream=False) -> str:
        if stream:
            # hugchat包目前不支持流式输出
            pass
        loop = asyncio.get_event_loop()
        msg = '\n'.join([message['role'] + ':' + message['content'] for message in messages])
        future = loop.run_in_executor(None, self.chatbot.chat, msg, 0.9, 0.95,
                                      1.2, 50, 2048, False, 2048, ['</s>'], False, True, False,
                                      False, 3)
        resp = await future
        return resp

