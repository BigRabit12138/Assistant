import torch
import json
import base64
import websockets

from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

import global_var

from utils import socket_no_proxy

IS_INITIALIZED = False


class TTV:
    def __init__(self):
        self.device = global_var.device if torch.cuda.is_available() else 'cpu'
        self.pipe = DiffusionPipeline.from_pretrained(
            "damo-vilab/text-to-video-ms-1.7b",
            torch_dtype=global_var.torch_type,
            variant="fp16"
        )
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.num_inference_steps = 25
        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_attention_slicing()

    def text2video(self, prompt: str):
        video_frames = self.pipe(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            num_images_per_prompt=200
        )
        return video_frames


async def ttv_client(ttv):
    global IS_INITIALIZED
    if global_var.run_local_mode:
        with socket_no_proxy():
            async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                          max_size=10 * 1024 * 1024) as websocket:
                while True:
                    if not IS_INITIALIZED:
                        message = {
                            'from': 'TTV',
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

                        video_frame = ttv.text2video(recv['content'])

                        video_frame = base64.b64encode(video_frame).decode()

                        message = {
                            'from': 'ITI',
                            'to': 'CLIENT',
                            'content': video_frame
                        }

                        message = json.dumps(message)
                        await websocket.send(message)
    else:
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            while True:
                if not IS_INITIALIZED:
                    message = {
                        'from': 'TTV',
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

                    video_frame = ttv.text2video(recv['content'])

                    video_frame = base64.b64encode(video_frame).decode()

                    message = {
                        'from': 'ITI',
                        'to': 'CLIENT',
                        'content': video_frame
                    }

                    message = json.dumps(message)
                    await websocket.send(message)
