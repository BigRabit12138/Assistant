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

    async def send_and_recv():
        global IS_INITIALIZED
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            global_var.logger.info("TTV启动")
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
                    global_var.logger.info(response)

                    response = json.loads(response)

                    if response['content'] == 'ok':
                        IS_INITIALIZED = True
                else:
                    recv = await websocket.recv()
                    global_var.logger.info(recv)

                    recv = json.loads(recv)

                    video_frame = ttv.text2video(recv['content'])
                    video_frame = base64.b64encode(video_frame).decode()

                    message = {
                        'from': 'TTV',
                        'to': 'CLIENT.TTV',
                        'content': video_frame
                    }

                    message = json.dumps(message)
                    await websocket.send(message)

    if global_var.run_local_mode:
        with socket_no_proxy():
            await send_and_recv()
    else:
        await send_and_recv()
