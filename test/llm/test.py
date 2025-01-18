import socks
import socket

socks.set_default_proxy(socks.SOCKS5, **{
    'addr': '127.0.0.1',
    'port': 10808
})
socket.socket = socks.socksocket

# from llm import HuggingChatForLangchain
#
# chat = HuggingChatForLangchain()
# res = chat.chatbot.query('Hi!who are yoy?')
# print(res)
