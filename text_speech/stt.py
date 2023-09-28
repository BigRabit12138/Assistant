import os
import io
import json
import torch
import base64
import websockets

from faster_whisper import WhisperModel

import global_var

from utils import socket_no_proxy

IS_INITIALIZED = False


class STT:
    def __init__(self):
        model_size = 'medium'
        self.device = global_var.device if torch.cuda.is_available() else 'cpu'
        self.model = WhisperModel(model_size)

    def transcribe(self, audio_binary_io):
        segments, info = self.model.transcribe(audio_binary_io)
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
    if global_var.run_local_mode:
        with socket_no_proxy():
            async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                          max_size=10 * 1024 * 1024) as websocket:
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
                        response = json.loads(response)

                        if response['content'] == 'ok':
                            IS_INITIALIZED = True
                    else:
                        recv = await websocket.recv()
                        recv = json.loads(recv)

                        audio_bytes_base64_decoded = base64.b64decode(recv['content'].encode())
                        audio_binary_io = io.BytesIO(audio_bytes_base64_decoded)
                        text = stt.transcribe(audio_binary_io)

                        message = {
                            'from': 'STT',
                            'to': 'CLIENT.STT',
                            'content': text
                        }
                        message = json.dumps(message)
                        await websocket.send(message)
    else:
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
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
                    response = json.loads(response)

                    if response['content'] == 'ok':
                        IS_INITIALIZED = True
                else:
                    recv = await websocket.recv()
                    recv = json.loads(recv)

                    audio_bytes_base64_decoded = base64.b64decode(recv['content'].encode())
                    audio_binary_io = io.BytesIO(audio_bytes_base64_decoded)
                    text = stt.transcribe(audio_binary_io)

                    message = {
                        'from': 'STT',
                        'to': 'CLIENT.STT',
                        'content': text
                    }
                    message = json.dumps(message)
                    await websocket.send(message)
