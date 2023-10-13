import json
import base64
import asyncio
import websockets

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import AsyncCallbackManagerForChainRun
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.prompts.base import BasePromptTemplate
from pydantic import Extra
from typing import List
from typing import Dict
from typing import Any
from typing import Optional

import global_var

from utils import socket_no_proxy


class ChatBotChain(Chain):
    prompt: BasePromptTemplate
    llm: BaseLanguageModel
    output_key: str = 'json_text'

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        return 'Custom chat bot chain'

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        tasks = asyncio.gather(audios_to_text(inputs['audios']),
                               images_to_text(inputs['images'])
                               )
        audios, images = loop.run_until_complete(tasks)
        inputs['images'] = images
        inputs['audios'] = audios
        prompt_value = self.prompt.format_prompt(**inputs)
        response = self.llm.generate_prompt(prompts=[prompt_value],
                                            callbacks=run_manager.get_child() if run_manager else None
                                            )

        if run_manager:
            run_manager.on_text(response.generations[0][0].text)
        return {self.output_key: response.generations[0][0].text}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:

        images, audios = await asyncio.gather(
            audios_to_text(inputs['audios']),
            images_to_text(inputs['images'])
        )
        inputs['images'] = images
        inputs['audios'] = audios

        prompt_value = self.prompt.format_prompt(**inputs)
        response = self.llm.generate_prompt(prompts=[prompt_value],
                                            callbacks=run_manager.get_child() if run_manager else None
                                            )

        if run_manager:
            await run_manager.on_text(response.generations[0][0].text)
        return {self.output_key: response.generations[0][0].text}


class ServerException(Exception):
    pass


async def audios_to_text(audio_list: list):
    # 处理音频
    text_list_from_audios = []

    async def send_and_recv():
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            for audio in audio_list:
                audio = audio.read()
                encoded_audio = base64.b64encode(audio).decode()
                message = {
                    'from': 'CLIENT.STT',
                    'to': 'STT',
                    'content': encoded_audio
                }
                message = json.dumps(message)
                await websocket.send(message)

                recv = await websocket.recv()
                recv = json.loads(recv)
                if recv.get('status') == 404:
                    print(recv)
                    global_var.logger.error(f'服务器后端出错：\n{recv}')

                    raise ServerException()
                else:
                    text_list_from_audios.append(recv['content'])

    if global_var.run_local_mode:
        with socket_no_proxy():
            await send_and_recv()
    else:
        await send_and_recv()

    return text_list_from_audios


async def images_to_text(images_list: list):
    # 处理图片
    text_list_from_images = []

    async def send_and_recv():
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            for image in images_list:
                image = image.read()
                encoded_image = base64.b64encode(image).decode()
                message = {
                    'from': 'CLIENT.ITT',
                    'to': 'ITT',
                    'content': encoded_image
                }
                message = json.dumps(message)
                await websocket.send(message)

                recv = await websocket.recv()
                recv = json.loads(recv)
                if recv.get('status') == 404:
                    print(recv)
                    global_var.logger.error(f'服务器后端出错：\n{recv}')

                    raise ServerException()
                else:
                    for i in recv['content']:
                        text_list_from_images.append(i)

    if global_var.run_local_mode:
        with socket_no_proxy():
            await send_and_recv()
    else:
        await send_and_recv()
    return text_list_from_images


async def text_to_audio(text: str):
    # 处理图片
    recv_list = []

    async def send_and_recv():
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            message = {
                'from': 'CLIENT.TTS',
                'to': 'TTS',
                'content': text
            }
            message = json.dumps(message)
            await websocket.send(message)

            recv = await websocket.recv()
            recv = json.loads(recv)
            if recv.get('status') == 404:
                print(recv)
                global_var.logger.error(f'服务器后端出错：\n{recv}')

                raise ServerException()
            else:
                recv['content']['speech_value'] = base64.b64decode(recv['content']['speech_value'].encode())
                recv_list.append(recv)

    if global_var.run_local_mode:
        with socket_no_proxy():
            await send_and_recv()
    else:
        await send_and_recv()
    return recv_list[0]['content']['speech_value'], recv_list[0]['content']['sampling_rate']


async def text_to_image(text: str):
    # 处理图片
    recv_list = []

    async def send_and_recv():
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            message = {
                'from': 'CLIENT.TTI',
                'to': 'TTI',
                'content': text
            }
            message = json.dumps(message)
            await websocket.send(message)

            recv = await websocket.recv()
            recv = json.loads(recv)
            if recv.get('status') == 404:
                print(recv)
                global_var.logger.error(f'服务器后端出错：\n{recv}')

                raise ServerException()
            else:
                recv['content'] = base64.b64decode(recv['content'].encode())
                recv_list.append(recv)

    if global_var.run_local_mode:
        with socket_no_proxy():
            await send_and_recv()
    else:
        await send_and_recv()
    return recv_list[0]['content']


async def text_to_video(text: str):
    # 处理图片
    recv_list = []

    async def send_and_recv():
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            message = {
                'from': 'CLIENT.TTV',
                'to': 'TTV',
                'content': text
            }
            message = json.dumps(message)
            await websocket.send(message)

            recv = await websocket.recv()
            recv = json.loads(recv)
            if recv.get('status') == 404:
                print(recv)
                global_var.logger.error(f'服务器后端出错：\n{recv}')

                raise ServerException()
            else:
                recv['content'] = base64.b64decode(recv['content'].encode())
                recv_list.append(recv)

    if global_var.run_local_mode:
        with socket_no_proxy():
            await send_and_recv()
    else:
        await send_and_recv()
    return recv_list[0]['content']


async def image_to_image(prompt: str, image: bytes):
    # 处理图片
    recv_list = []

    async def send_and_recv():
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            image_ = base64.b64encode(image).decode()
            message = {
                'from': 'CLIENT.ITI',
                'to': 'ITI',
                'content': {
                    'prompt': prompt,
                    'image': image_
                }
            }
            message = json.dumps(message)
            await websocket.send(message)

            recv = await websocket.recv()
            recv = json.loads(recv)
            if recv.get('status') == 404:
                print(recv)
                global_var.logger.error(f'服务器后端出错：\n{recv}')

                raise ServerException()
            else:
                recv['content'] = base64.b64decode(recv['content'].encode())
                recv_list.append(recv)

    if global_var.run_local_mode:
        with socket_no_proxy():
            await send_and_recv()
    else:
        await send_and_recv()
    return recv_list[0]['content']
