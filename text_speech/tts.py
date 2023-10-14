import os
import io
import json
import base64
import asyncio
import websockets

from scipy.io.wavfile import write
from transformers import AutoProcessor, AutoModel

import global_var

from utils import socket_no_proxy

IS_INITIALIZED = False


class TTS:
    def __init__(self):
        model_name = "suno/bark-small"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def transcribe(self, txt: str = ''):
        assert txt != '', 'txt should not been empty'
        inputs = self.processor(
            text=[txt],
            return_tensors="pt",
        )

        speech_values = self.model.generate(**inputs, do_sample=True)
        speech_values = speech_values.cpu().numpy().squeeze()

        sampling_rate = self.model.generation_config.sample_rate

        return speech_values, sampling_rate


async def tts_client(tts):
    global IS_INITIALIZED

    async def send_and_recv():
        global IS_INITIALIZED
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            global_var.logger.info("TTS启动")
            while True:
                if not IS_INITIALIZED:
                    message = {
                        'from': 'TTS',
                        'to': 'SERVER',
                        'content': 'hello'
                    }
                    message = json.dumps(message)
                    await websocket.send(message)

                    response = await websocket.recv()
                    global_var.logger.info(response)

                    response = json.loads(response)

                    if response['content'] == 'ok':
                        IS_INITIALIZED = True
                else:
                    recv = await websocket.recv()
                    global_var.logger.info(recv)

                    recv = json.loads(recv)

                    speech_value, sampling_rate = tts.transcribe(recv['content'])
                    speech_value = base64.b64encode(speech_value).decode()

                    message = {
                        'from': 'TTS',
                        'to': 'CLIENT.TTS',
                        'content': {
                            'speech_value': speech_value,
                            'sampling_rate': sampling_rate,
                        }
                    }
                    message = json.dumps(message)
                    await websocket.send(message)

    if global_var.run_local_mode:
        with socket_no_proxy():
            await send_and_recv()
    else:
        await send_and_recv()