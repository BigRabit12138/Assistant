import global_var
if global_var.run_local_mode:
    import copy
    import socks
    import socket

    default_socket = copy.deepcopy(socket.socket)
    global_var.default_socket = default_socket

    socks.set_default_proxy(socks.SOCKS5, **global_var.proxy_setting)
    socket.socket = socks.socksocket


import asyncio
import argparse

from server import switcher_server
from text_speech import STT
from text_speech import stt_client
from text_speech import TTS
from text_speech import tts_client
from text_image_video import ITT
from text_image_video import itt_client
from text_image_video import ITI
from text_image_video import iti_client
from text_image_video import TTI
from text_image_video import tti_client
from text_image_video import TTV
from text_image_video import ttv_client


async def main():
    parser = argparse.ArgumentParser(
        prog='Assistant',
        description='Assistant的后端启动程序')

    parser.add_argument('--TTS', action='store_true', help='启动TTS(text to speech)服务')
    parser.add_argument('--STT', action='store_true', help='启动STT(speech to text)服务')
    parser.add_argument('--ITI', action='store_true', help='启动ITI(image to image)服务')
    parser.add_argument('--ITT', action='store_true', help='启动ITT(image to text)服务')
    parser.add_argument('--TTI', action='store_true', help='启动TTI(text to image)服务')
    parser.add_argument('--TTV', action='store_true', help='启动TTV(text to video)服务')
    parser.add_argument('--Server', action='store_true', help='启动Websockets服务器')
    args = parser.parse_args()

    tasks = []
    if args.TTS:
        tasks.append(tts_client(TTS()))
    if args.STT:
        tasks.append(stt_client(STT()))
    if args.ITI:
        tasks.append(iti_client(ITI()))
    if args.ITT:
        tasks.append(itt_client(ITT()))
    if args.TTI:
        tasks.append(tti_client(TTI()))
    if args.TTV:
        tasks.append(ttv_client(TTV()))
    if args.Server:
        tasks.append(switcher_server())

    await asyncio.gather(*tasks)


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
