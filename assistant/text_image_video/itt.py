import json
import torch
import os.path
import base64
import websockets

from io import BytesIO
from PIL import Image
from fastsam import FastSAM, FastSAMPrompt
from lavis.models import load_model_and_preprocess

import global_var

from assistant.utils import socket_no_proxy

IS_INITIALIZED = False


class ITT:
    def __init__(self):
        self.device = global_var.device if torch.cuda.is_available() else 'cpu'

        self.captioning_model, self.vis_processor, _ = load_model_and_preprocess(
            name='blip_caption', model_type='large_coco', is_eval=True, device=self.device
        )

        sam_checkpoint = os.path.join(global_var.project_dir, 'model_weight/FastSAM-x.pt')
        self.sam_model = FastSAM(sam_checkpoint)

    def captioning_image_form_file(self, image_name: str) -> list[str]:
        image_path = os.path.join(global_var.project_dir, 'text_image_video/new_images', image_name)
        raw_image = Image.open(image_path).convert('RGB')
        image = self.vis_processor['eval'](raw_image).unsqueeze(0).to(self.device)
        caption = self.captioning_model.generate({'image': image})
        return caption

    def captioning_image(self, image: bytes) -> list[str]:
        raw_image = Image.open(BytesIO(image)).convert('RGB')
        image = self.vis_processor['eval'](raw_image).unsqueeze(0).to(self.device)
        caption = self.captioning_model.generate({'image': image})
        return caption

    def segment_anything_all(self, image_name: str):
        image_path = os.path.join(global_var.project_dir, 'text_image_video/new_images', image_name)
        everything_results = self.sam_model(image_path, device=self.device, retina_masks=True,
                                            imgsz=1024, conf=0.4, iou=0.9)
        prompt_process = FastSAMPrompt(image_path, everything_results, device=self.device)
        masks = prompt_process.everything_prompt()
        pass


async def itt_client(itt):
    global IS_INITIALIZED

    async def send_and_recv():
        global IS_INITIALIZED
        async with websockets.connect(f'ws://{global_var.ip}:{global_var.port}/',
                                      max_size=10 * 1024 * 1024) as websocket:
            global_var.logger.info("ITT启动")
            while True:
                if not IS_INITIALIZED:
                    message = {
                        'from': 'ITT',
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

                    image_bytes_base64_decoded = base64.b64decode(recv['content'].encode())
                    text_list = itt.captioning_image(image_bytes_base64_decoded)

                    message = {
                        'from': 'ITT',
                        'to': 'CLIENT.ITT',
                        'content': text_list
                    }

                    message = json.dumps(message)
                    await websocket.send(message)

    if global_var.run_local_mode:
        with socket_no_proxy():
            await send_and_recv()
    else:
        await send_and_recv()
