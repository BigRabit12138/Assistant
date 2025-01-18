import global_var

import copy
import socks
import socket

default_socket = copy.deepcopy(socket.socket)
global_var.default_socket = default_socket

socks.set_default_proxy(socks.SOCKS5, **global_var.proxy_setting)
socket.socket = socks.socksocket


import asyncio

from assistant.agents import Scene
from assistant.agents import Actor
from assistant.agents import SceneMessage
from assistant.agents.multi_agents.roles.actor import ActorContext
from assistant.agents.multi_agents.roles.actor import ActorSetting

ac = ActorContext()
as_ = ActorSetting()

actor1 = Actor(name='张三',
               gender='男',
               age=25,
               figure='五短身材。',
               appearance='方脸，鹰勾鼻，小眼睛，大嘴巴。',
               character='老实，懦弱，逆来顺受。',
               principles_of_conduct='第一优先级：活下去',
               occupation='卖炊饼')
scene = Scene()
actor1.set_scene(scene)
actor1.set_goal('活下去')
scene.memory.append(SceneMessage(role='李三',
                                 action='张三拿起一米长的大刀。',
                                 conversation='我的大刀早已饥渴难耐了.'))
asyncio.run(actor1.run())
actor1.recv(SceneMessage(role='李四',
                         action='李四拿起一米长的大刀。',
                         conversation='我的大刀早已饥渴难耐了.'))
pass
