# 注意这个测试文件需要在主目录下运行

import global_var
import copy
import socks
import socket

default_socket = copy.deepcopy(socket.socket)
global_var.default_socket = default_socket

socks.set_default_proxy(socks.SOCKS5, **global_var.proxy_setting)
socket.socket = socks.socksocket


from agents import ChatBotChain
from llm import HuggingChatForLangchain
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.stdout import StdOutCallbackHandler

from agents.prompts import CHATBOT_PROMPT

chat_chain = ChatBotChain(
    prompt=PromptTemplate.from_template(CHATBOT_PROMPT),
    llm=HuggingChatForLangchain()
)

audios_list = [open('./resource/audio.wav', 'rb')]
images_list = [open('./resource/photo.png', 'rb')]
gg = chat_chain.run({'audios': audios_list,
                     'images': images_list,
                     'text': '你看见的图片是啥子内容？'},
                    callbacks=[StdOutCallbackHandler()])
pass
