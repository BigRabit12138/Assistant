import global_var

import copy
import socks
import socket

default_socket = copy.deepcopy(socket.socket)
global_var.default_socket = default_socket

socks.set_default_proxy(socks.SOCKS5, **global_var.proxy_setting)
socket.socket = socks.socksocket

import asyncio

from agents.multi_agents.roles import Director
from agents.multi_agents.roles.director import DirectorContext

dc = DirectorContext()

director = Director('在一个都市公寓，写一个几个年轻人之间的故事、纠葛。')
asyncio.run(director.run())
