import json
import torch
import websockets
import os.path
import base64

from io import BytesIO
from PIL import Image
from PIL import ImageOps
from diffusers import StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler

import global_var

from utils import socket_no_proxy

IS_INITIALIZED = False


class ITI:
    def __init__(self):
        self.device = global_var.device if torch.cuda.is_available() else 'cpu'
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=global_var.torch_type,
            safety_checker=None
        )
        self.pipe.to(self.device)
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )
        self.num_steps = 10

    def image2image_from_file(self, image_name: str, prompt: str):
        image_path = os.path.join(global_var.project_dir, 'text_image_video/new_images', image_name)
        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')
        image = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=self.num_steps,
            image_guidance_scale=1
        ).images[0]

        return image

    def image2image(self, image: bytes, prompt: str):
        image = Image.open(BytesIO(image))
        image = ImageOps.exif_transpose(image)
        image = image.convert('RGB')
        image = self.pipe(
            prompt=prompt,
            image=image,
            num_inference_steps=self.num_steps,
            image_guidance_scale=1
        ).images[0]

        return image


async def iti_client(iti):
    global IS_INITIALIZED

    async def send_and_recv():
        global IS_INITIALIZED
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            global_var.logger.info("ITI启动")
            while True:
                if not IS_INITIALIZED:
                    message = {
                        'from': 'ITI',
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

                    image_bytes_base64_decoded = base64.b64decode(recv['content']['image'].encode())
                    image = iti.image2image(image_bytes_base64_decoded, recv['content']['prompt'])

                    image = base64.b64encode(image).decode()
                    message = {
                        'from': 'ITI',
                        'to': 'CLIENT.ITI',
                        'content': image
                    }

                    message = json.dumps(message)
                    await websocket.send(message)

    if global_var.run_local_mode:
        with socket_no_proxy():
            await send_and_recv()
    else:
        await send_and_recv()

