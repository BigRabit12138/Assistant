import os
import json
import torch
import base64
import websockets

from io import BytesIO
from faster_whisper import WhisperModel

import global_var

from assistant.utils import socket_no_proxy

IS_INITIALIZED = False


class STT:
    def __init__(self):
        model_size = 'medium'
        self.device = global_var.device if torch.cuda.is_available() else 'cpu'
        self.model = WhisperModel(model_size)

    def transcribe(self, audio: bytes):
        segments, info = self.model.transcribe(BytesIO(audio))
        txt = ''.join([segment.text for segment in segments])
        return txt

    def transcribe_from_file(self, file_name=''):
        assert file_name != '', ('transcribe_from_file needs file_name, '
                                 'which should be stored in text_speech/new_audios/')
        path_to_file = os.path.join(global_var.project_dir, 'text_speech/new_audios', file_name)
        assert os.path.exists(path_to_file), f'{file_name} does not exist!'
        segments, info = self.model.transcribe(path_to_file)
        txt = ''.join([segment.text for segment in segments])
        return txt

    def stream_transcribe(self):
        pass


async def stt_client(stt):
    global IS_INITIALIZED

    async def send_and_recv():
        global IS_INITIALIZED
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            global_var.logger.info("STT启动")
            while True:
                if not IS_INITIALIZED:
                    message = {
                        'from': 'STT',
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

                    audio_bytes_base64_decoded = base64.b64decode(recv['content'].encode())
                    text = stt.transcribe(audio_bytes_base64_decoded)

                    message = {
                        'from': 'STT',
                        'to': 'CLIENT.STT',
                        'content': text
                    }
                    message = json.dumps(message)
                    await websocket.send(message)

    if global_var.run_local_mode:
        with socket_no_proxy():
            await send_and_recv()
    else:
        await send_and_recv()
