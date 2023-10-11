#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/7/21 11:15
@Author  : Leo Xiao
@File    : anthropic_api.py
"""
import asyncio
import anthropic


from anthropic import Anthropic
from claude2_api.client import ClaudeAPIClient
from claude2_api.client import get_session_data

import global_var

from llm.base_gpt_api import BaseGPTAPI
from utils import socket_no_proxy


class Claude2(BaseGPTAPI):
    def __init__(self):
        self.client = Anthropic(api_key=global_var.claude_api_key)

    def completion(self, messages: list[dict]):
        msg = '\n'.join([message['role'] + ':' + message['content'] for message in messages])

        resp = self.client.completions.create(
            model="claude-2",
            prompt=f"{anthropic.HUMAN_PROMPT} {msg} {anthropic.AI_PROMPT}",
            max_tokens_to_sample=1000
        )
        resp = {
            'choices': [
                {
                    'message': {
                        'content': resp.completion
                    }
                }
            ]
        }
        return resp

    async def acompletion(self, messages: list[dict]):
        loop = asyncio.get_event_loop()
        msg = '\n'.join([message['role'] + ':' + message['content'] for message in messages])
        future = loop.run_in_executor(None, self.client.completions.create, "claude-2",
                                      f"{anthropic.HUMAN_PROMPT} {msg} {anthropic.AI_PROMPT}", 1000
                                      )
        resp = await future
        resp = {
            'choices': [
                {
                    'message': {
                        'content': resp.completion
                    }
                }
            ]
        }
        return resp

    async def acompletion_text(self, messages: list[dict], stream=False) -> str:
        if stream:
            # 目前暂时无流式输出
            pass
        loop = asyncio.get_event_loop()
        msg = '\n'.join([message['role'] + ':' + message['content'] for message in messages])
        future = loop.run_in_executor(None, self.client.completions.create, "claude-2",
                                      f"{anthropic.HUMAN_PROMPT} {msg} {anthropic.AI_PROMPT}", 1000
                                      )
        resp = await future
        resp = resp.completion
        return resp


class Claude2V2(BaseGPTAPI):
    def __init__(self):
        with socket_no_proxy():
            self.data = get_session_data()

        self.client = ClaudeAPIClient(self.data)
        self.chat_id = self.client.create_chat()

    def completion(self, messages: list[dict]):
        # msg = '\n'.join([message['role'] + ':' + message['content'] for message in messages])
        msg = '\n'.join([message['content'] for message in messages if message['role'] == 'user'])
        resp = self.client.send_message(chat_id=self.chat_id, prompt=msg)
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
        # msg = '\n'.join([message['role'] + ':' + message['content'] for message in messages])
        msg = '\n'.join([message['content'] for message in messages if message['role'] == 'user'])
        future = loop.run_in_executor(None, self.client.send_message,
                                      self.chat_id, msg)
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
            # claude-api-py包目前不支持流式输出
            pass
        loop = asyncio.get_event_loop()
        # msg = '\n'.join([message['role'] + ':' + message['content'] for message in messages])
        msg = '\n'.join([message['content'] for message in messages if message['role'] == 'user'])
        future = loop.run_in_executor(None, self.client.send_message,
                                      self.chat_id, msg)
        resp = await future
        return resp
