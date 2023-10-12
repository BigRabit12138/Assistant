# 注意这个测试文件需要在主目录下运行

import global_var
import copy
import socks
import socket

default_socket = copy.deepcopy(socket.socket)
global_var.default_socket = default_socket

socks.set_default_proxy(socks.SOCKS5, **global_var.proxy_setting)
socket.socket = socks.socksocket

import asyncio

from agents import ChatBotChain
from llm import HuggingChatForLangchain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.stdout import StdOutCallbackHandler

from agents.prompts import CHATBOT_PROMPT

chat_chain = ChatBotChain(
    prompt=PromptTemplate.from_template(CHATBOT_PROMPT),
    llm=HuggingChatForLangchain()
)
audios_list = [open('/resource/audio.wav', 'rb')]
images_list = [open('/resource/photo.png', 'rb')]
gg = chat_chain.run({'audios_list': audios_list,
                     'images_list': images_list,
                     'text': '你看见的图片是啥子内容？',
                     'text_list_from_audios': '',
                     'text_list_from_images': ''},
                    callbacks=[StdOutCallbackHandler()])
bb = asyncio.run(chat_chain.arun({'audios_list': audios_list,
                                  'images_list': images_list,
                                  'text': '你看见的图片是啥子内容？',
                                  'text_list_from_audios': '',
                                  'text_list_from_images': ''},
                                 callbacks=[StdOutCallbackHandler()]))
pass
