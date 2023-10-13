import json
import torch
import base64
import websockets

from diffusers import DiffusionPipeline

import global_var

from utils import socket_no_proxy

IS_INITIALIZED = False


class TTI:
    def __init__(self):
        self.device = global_var.device if torch.cuda.is_available() else 'cpu'
        self.base_model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=global_var.torch_type,
            variant="fp16",
            use_safetensors=True
        )
        self.base_model.to(self.device)
        self.refiner_model = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0",
            text_encoder_2=self.base_model.text_encoder_2,
            vae=self.base_model.vae,
            torch_dtype=global_var.torch_type,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner_model.to(self.device)
        self.n_steps = 40
        self.high_noise_frac = 0.8

    def text2image(self, prompt: str):
        image = self.base_model(
            prompt=prompt,
            num_inference_steps=self.n_steps,
            denoising_end=self.high_noise_frac,
            output_type='latent'
        ).images
        image = self.refiner_model(
            prompt=prompt,
            num_inference_steps=self.n_steps,
            denoising_start=self.high_noise_frac,
            image=image
        ).images[0]
        return image


async def tti_client(tti):
    global IS_INITIALIZED

    async def send_and_recv():
        global IS_INITIALIZED
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            while True:
                if not IS_INITIALIZED:
                    message = {
                        'from': 'TTI',
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

                    image = tti.text2image(recv['content'])

                    image = base64.b64encode(image).decode()

                    message = {
                        'from': 'TTI',
                        'to': 'CLIENT.TTI',
                        'content': image
                    }

                    message = json.dumps(message)
                    await websocket.send(message)

    if global_var.run_local_mode:
        with socket_no_proxy():
            await send_and_recv()
    else:
        await send_and_recv()
