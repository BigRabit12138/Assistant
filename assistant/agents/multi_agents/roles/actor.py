#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/5/11 14:42
@Author  : alexanderwu
@File    : role.py
"""
import re
import json

from pydantic import Field
from pydantic import BaseModel
from typing import TYPE_CHECKING

import global_var

from assistant.llm import Claude2V2 as LLM
from assistant.agents.messages import SceneMessage
from assistant.agents.prompts import ACTOR_TEMPLATE

if TYPE_CHECKING:
    from assistant.agents.multi_agents.envs import Scene


class ActorSetting(BaseModel):
    """人物设定"""
    # 外部属性
    gender: str = Field(default='')
    age: int = Field(default=0)
    figure: str = Field(default='')
    appearance: str = Field(default='')

    # 内部属性
    name: str = Field(default='')
    character: str = Field(default='')
    principles_of_conduct: str = Field(default='')
    occupation: str = Field(default='')

    @property
    def _get_actor_desc_by_other_actors(self):
        return f"{self.gender}，大约{self.age}，{self.appearance}，{self.figure}。"

    @property
    def _get_actor_desc_by_director(self):
        return (f"{self.name}，{self.gender}，大约{self.age}，{self.appearance}，{self.figure}，"
                f"{self.character}, {self.principles_of_conduct}，{self.occupation}。")

    def __str__(self):
        return f"{self.name}_{self.gender}_{self.age}"

    def __repr__(self):
        return self.__str__()


class ActorContext(BaseModel):
    """演员运行时上下文"""
    scene: 'Scene' = Field(default=None)
    # TODO: 使用简单列表，后续修改
    memory: list[SceneMessage] = Field(default=[])
    news: list[SceneMessage] = Field(default=[])
    goal: str = Field(default='')

    @property
    def history(self) -> list[SceneMessage]:
        return self.memory


class Actor:
    """演员"""

    def __init__(self,
                 name="",
                 gender="",
                 age=0,
                 figure="",
                 appearance="",
                 character="",
                 principles_of_conduct="",
                 occupation=""
                 ):
        self._llm = LLM()
        self._setting = ActorSetting(name=name,
                                     gender=gender,
                                     age=age,
                                     figure=figure,
                                     appearance=appearance,
                                     character=character,
                                     principles_of_conduct=principles_of_conduct,
                                     occupation=occupation
                                     )
        self._actor_id = str(self._setting)
        self._ac = ActorContext()

    def flush(self):
        for msg in self._ac.news:
            self.recv(msg)
        self._ac.news = []

    def set_scene(self, scene: 'Scene'):
        """设置演员当前的场景"""
        self._ac.scene = scene

    def set_goal(self, goal: str):
        """设置演员在当前场景的目标"""
        self._ac.goal = goal

    def _get_prompt(self):
        """获取prompt"""
        gender = self._setting.gender
        name = self._setting.name
        age = self._setting.age
        appearance = self._setting.appearance
        figure = self._setting.figure
        character = self._setting.character
        principles_of_conduct = self._setting.principles_of_conduct
        scene = self._ac.scene.scene_desc
        actors = '\n'.join([actor._setting._get_actor_desc_by_other_actors for actor in self._ac.scene.get_actors()])
        goal = self._ac.goal
        memory = '\n'.join([str(msg) for msg in self._ac.memory])
        memory += '\n'.join([str(msg) for msg in self._ac.news])
        occupation = self._setting.occupation
        injection_json = {
            'gender': gender,
            'name': name,
            'age': age,
            'appearance': appearance,
            'figure': figure,
            'character': character,
            'principles_of_conduct': principles_of_conduct,
            'scene': scene,
            'actors': actors,
            'goal': goal,
            'memory': memory,
            'occupation': occupation
        }
        return ACTOR_TEMPLATE.format(**injection_json)

    async def _act(self) -> SceneMessage:
        prompt = self._get_prompt()

        response = await self._llm.aask(prompt)
        if response.startswith('{'):
            json_resp = json.loads(response)
        else:
            code_regex = r'`{3}json([\s\S]*?)`{3}'
            response_json = re.findall(code_regex, response, re.MULTILINE | re.DOTALL)
            if len(response_json) == 0:
                code_regex = r'`({[\s\S]*?})'
                response_json = re.findall(code_regex, response, re.MULTILINE | re.DOTALL)
                assert response_json == 1
            response = response_json[0]
            json_resp = json.loads(response)
        json_resp['role'] = self._setting.name
        scene_msg = SceneMessage(**json_resp)
        self._ac.news.append(scene_msg)
        return scene_msg

    def _observe(self):
        """从场景中观察新消息，并加入新信息列表"""
        if not self._ac.scene:
            return

        # 获取新信息并添加到新信息列表
        news_text = ''
        for msg in self._ac.scene.news:
            if msg in self._ac.news:
                continue
            else:
                self._ac.news.append(msg)
                news_text += str(msg)

        if news_text != '':
            global_var.logger.debug(f'{self._setting} observed: {news_text}')

    def _publish_message(self, msg: SceneMessage):
        """如果actor归属于scene，那么actor的消息会向scene广播"""
        if self._ac.scene:
            self._ac.scene.publish_message(msg)

    def recv(self, message: SceneMessage) -> None:
        """往记忆中添加新信息"""
        if message in self._ac.memory:
            return
        else:
            self._ac.memory.append(message)

    async def run(self):
        """观察，并基于观察的结果思考、行动"""

        self._observe()

        scene_message = await self._act()
        # 将回复发布到环境，等待下一个订阅者处理
        self._publish_message(scene_message)
