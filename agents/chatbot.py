import json
import copy
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
        input_transformed = copy.deepcopy(inputs)
        del(input_transformed['audios_list'])
        del (input_transformed['images_list'])

        loop = asyncio.get_event_loop()
        tasks = asyncio.gather(audios_to_text(inputs['audios_list']),
                               images_to_text(inputs['images_list'])
                               )
        text_list_from_audios, text_list_from_images = loop.run_until_complete(tasks)
        input_transformed['text_list_from_images'] = text_list_from_images
        input_transformed['text_list_from_audios'] = text_list_from_audios

        prompt_value = self.prompt.format_prompt(**input_transformed)
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
        input_transformed = copy.deepcopy(inputs)
        del (input_transformed['audios_list'])
        del (input_transformed['images_list'])

        text_list_from_images, text_list_from_audios = await asyncio.gather(
            audios_to_text(inputs['audios_list']),
            images_to_text(inputs['images_list'])
        )
        input_transformed['text_list_from_images'] = text_list_from_images
        input_transformed['text_list_from_audios'] = text_list_from_audios

        prompt_value = self.prompt.format_prompt(**input_transformed)
        response = self.llm.generate_prompt(prompts=[prompt_value],
                                            callbacks=run_manager.get_child() if run_manager else None
                                            )

        response = json.loads(response.generations[0][0].text)

        if run_manager:
            await run_manager.on_text(response.generations[0][0].text)
        return {self.output_key: response.generations[0][0].text}


async def audios_to_text(audio_list: list):
    # 处理音频
    with socket_no_proxy():
        async with websockets.connect('ws://127.0.0.1:9999/', max_size=10 * 1024 * 1024) as websocket:
            text_list_from_audios = []
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

                text_list_from_audios.append(recv['content'])
    return text_list_from_audios


async def images_to_text(images_list: list):
    # 处理图片
    with socket_no_proxy():
        async with websockets.connect('ws://127.0.0.1:9999/', max_size=10 * 1024 * 1024) as websocket:
            text_list_from_images = []
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

                [text_list_from_images.append(i) for i in recv['content']]

    return text_list_from_images


async def text_to_audio(text: str):
    # 处理图片
    with socket_no_proxy():
        async with websockets.connect('ws://127.0.0.1:9999/', max_size=10 * 1024 * 1024) as websocket:
            message = {
                'from': 'CLIENT.TTS',
                'to': 'TTS',
                'content': text
            }
            message = json.dumps(message)
            await websocket.send(message)

            recv = await websocket.recv()
            recv = json.loads(recv)
            recv['content']['speech_value'] = base64.b64decode(recv['content']['speech_value'].encode())
    return recv['content']['speech_value'], recv['content']['sampling_rate']


async def text_to_image(text: str):
    # 处理图片
    with socket_no_proxy():
        async with websockets.connect('ws://127.0.0.1:9999/', max_size=10 * 1024 * 1024) as websocket:
            message = {
                'from': 'CLIENT.TTI',
                'to': 'TTI',
                'content': text
            }
            message = json.dumps(message)
            await websocket.send(message)

            recv = await websocket.recv()
            recv = json.loads(recv)
            recv['content'] = base64.b64decode(recv['content'].encode())
    return recv['content']


async def text_to_video(text: str):
    # 处理图片
    with socket_no_proxy():
        async with websockets.connect('ws://127.0.0.1:9999/', max_size=10 * 1024 * 1024) as websocket:
            message = {
                'from': 'CLIENT.TTV',
                'to': 'TTV',
                'content': text
            }
            message = json.dumps(message)
            await websocket.send(message)

            recv = await websocket.recv()
            recv = json.loads(recv)
            recv['content'] = base64.b64decode(recv['content'].encode())
    return recv['content']


async def image_to_image(prompt: str, image: bytes):
    # 处理图片
    with socket_no_proxy():
        async with websockets.connect('ws://127.0.0.1:9999/', max_size=10 * 1024 * 1024) as websocket:
            image = base64.b64encode(image).decode()
            message = {
                'from': 'CLIENT.TTV',
                'to': 'TTV',
                'content': {
                    'prompt': prompt,
                    'image': image
                }
            }
            message = json.dumps(message)
            await websocket.send(message)

            recv = await websocket.recv()
            recv = json.loads(recv)
            recv['content'] = base64.b64decode(recv['content'].encode())
    return recv['content']
