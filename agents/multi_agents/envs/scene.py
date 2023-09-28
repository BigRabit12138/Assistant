import asyncio

from pydantic import Field
from pydantic import BaseModel
from typing import Iterable
from agents.multi_agents.roles import Actor
from agents.messages import SceneMessage


class Scene(BaseModel):
    """场景，承载一批演员，演员在场景中互动消息，可以被其他演员观察到
    """

    actors: list[Actor] = Field(default=[])
    # TODO: 使用简单列表，后续修改
    memory: list[SceneMessage] = Field(default=[])
    news: list[SceneMessage] = Field(default=[])
    history: str = Field(default='')
    scene_desc: str = Field(default='')
    scene_name: str = Field(default='')

    class Config:
        arbitrary_types_allowed = True

    def flush(self):
        for new_msg in self.news:
            self.memory.append(new_msg)
        self.news = []

    def get_scene_desc(self):
        """获取场景描述"""
        pass

    def set_desc_and_name(self, desc: str, name: str):
        """设置场景的描述"""
        self.scene_desc = desc
        self.scene_name = name

    def add_actor(self, actor: 'Actor'):
        """在当前场景增加一个演员
        """
        actor.set_scene(self)
        self.actors.append(actor)

    def add_actors(self, actors: Iterable['Actor']):
        """在当前场景增加一批演员
        """
        for actor in actors:
            self.add_actor(actor)

    def publish_message(self, message: 'SceneMessage'):
        """向当前场景发布信息
        """
        self.news.append(message)

        self.history += f"\n{message}"

    async def run(self, k=50):
        """处理一次所有信息的运行
        """
        for _ in range(k):
            futures = []
            for actor in self.actors:
                future = actor.run()
                futures.append(future)

            await asyncio.gather(*futures)

    def get_actors(self) -> list['Actor']:
        """获得场景内的所有演员
        """
        return self.actors
