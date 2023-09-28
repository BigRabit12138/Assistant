import global_var

import copy
import socks
import socket

default_socket = copy.deepcopy(socket.socket)
global_var.default_socket = default_socket

socks.set_default_proxy(socks.SOCKS5, **global_var.proxy_setting)
socket.socket = socks.socksocket


import asyncio

from agents.multi_agents.envs import Scene
from agents.multi_agents.roles import Actor
from agents.messages import SceneMessage

actor1 = Actor(name='张三',
               gender='男',
               age=25,
               figure='五短身材。',
               appearance='方脸，鹰勾鼻，小眼睛，大嘴巴。',
               character='老实，懦弱，逆来顺受。',
               principles_of_conduct='第一优先级：活下去',
               occupation='卖炊饼')
scene1 = Scene()
scene2 = Scene(actors=[actor1], memory=[], history='', scene_desc='这个大房间，有一个大桌子，和椅子。')
scene1.add_actor(actor1)
scene1.add_actors([actor1])
scene1.set_desc(desc='这个大房间，有一个大桌子，和椅子。')
scene1.publish_message(SceneMessage(role='李三',
                                    action='张三拿起一米长的大刀。',
                                    conversation='我的大刀早已饥渴难耐了.'))
actors = scene1.get_actors()
asyncio.run(scene1.run())
pass
