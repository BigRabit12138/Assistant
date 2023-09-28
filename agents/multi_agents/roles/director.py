import re
import asyncio
import json

from pydantic import Field
from pydantic import BaseModel

from agents.multi_agents.envs import Scene
from agents.prompts import INIT_TEMPLATE
from agents.prompts import ADD_SCENE_TEMPLATE
from agents.prompts import ADD_ACTOR_TEMPLATE
from agents.prompts import WRITE_DOWN_TEMPLATE
from agents.prompts import EVALUATE_PLOT_TEMPLATE
from agents.multi_agents.roles import Actor
from llm import Claude2V2 as LLM


class DirectorContext(BaseModel):
    # TODO: 使用简单列表，后续修改
    # memory: list[SceneMessage] = Field(default=[])
    memory: list[str] = Field(default=[])
    scenes: list[Scene] = Field(default=[])


class Director:
    def __init__(self, creation_demand: str):
        self._dc: DirectorContext = DirectorContext()
        self._llm = LLM()
        self._creation_demand = creation_demand
        # 初始化场景和一批演员
        self._init_scene_actors()

    def _init_scene_actors(self):
        """初始化一个场景和一批演员"""
        prompt = INIT_TEMPLATE.format(**{'Creation_Demand': self._creation_demand})
        response = self._llm.ask(prompt)
        if response.startswith('{'):
            json_resp = json.loads(response)
        else:
            code_regex = r'`{3}json([\s\S]*?)`{3}'
            response = re.findall(code_regex, response, re.MULTILINE | re.DOTALL)
            assert len(response) == 1
            response = response[0]
            json_resp = json.loads(response)

        scene = Scene()
        scene.set_desc_and_name(desc=json_resp['scene']['scene_desc'],
                                name=json_resp['scene']['scene_name'])
        self._dc.scenes.append(scene)

        for idx, actor in enumerate(json_resp['actors']):
            goal = actor['goal']
            del actor['goal']
            actor = Actor(**actor)
            actor.set_goal(goal)
            self._dc.scenes[0].add_actor(actor=actor)

    def _add_scene(self):
        """添加一个场景"""
        plot = self._dc.memory
        scenes = '\n'.join([scene.get_scene_desc() for scene in self._dc.scenes])

        prompt = ADD_SCENE_TEMPLATE.format(**{'plot': plot, 'scenes': scenes})
        response = self._llm.ask(prompt)
        json_resp = json.loads(response)

        scene = Scene()
        scene.set_desc_and_name(desc=json_resp['scene']['scene_desc'],
                                name=json_resp['scene']['scene_name'])
        self._dc.scenes.append(scene)

    def _add_actor(self):
        """添加一个演员"""
        plot = self._dc.memory
        scenes = '\n'.join([scene.get_scene_desc() for scene in self._dc.scenes])

        prompt = ADD_ACTOR_TEMPLATE.format(**{'plot': plot, 'scenes': scenes})
        response = self._llm.ask(prompt)
        json_resp = json.loads(response)
        selected_scene = None
        for scene in self._dc.scenes:
            if scene.scene_name == json_resp['scene_name']:
                selected_scene = scene

        assert selected_scene is not None

        goal = json_resp['actor']['goal']
        del json_resp['actor']['goal']

        new_actor = Actor(**json_resp['actor'])
        new_actor.set_goal(goal)
        selected_scene.add_actor(new_actor)

    def _add_scenes(self, scene_count: int = 1):
        """添加一批场景"""
        for _ in range(scene_count):
            self._add_scene()

    def _add_actors(self, actor_count: int = 1):
        """初始化一批演员"""
        for _ in range(actor_count):
            self._add_actor()

    def evaluate(self, new_story: str) -> bool:
        """评估剧情时候合格,合格返回True"""
        prompt = EVALUATE_PLOT_TEMPLATE.format(**{'story': new_story})
        response = self._llm.ask(prompt)
        json_resp = json.loads(response)
        return json_resp['evaluate_result']

    def write_story_down(self) -> str:
        """获取所有场景的新消息，将新消息整合为小说"""
        all_scenes_text = ''
        for scene in self._dc.scenes:
            all_scenes_text += scene.scene_name + '，' + scene.scene_desc + '\n'
            all_scenes_text += scene.history + '\n'

        prompt = WRITE_DOWN_TEMPLATE.format(**{'raw_story': all_scenes_text})
        response = self._llm.ask(prompt)
        json_resp = json.loads(response)
        return json_resp['new_story']

    async def run(self):
        while True:
            future = []
            for scene in self._dc.scenes:
                future.append(scene.run())
            await asyncio.gather(*future)

            new_story = self.write_story_down()
            if self.evaluate(new_story):
                self._dc.memory.append(new_story)
                for scene in self._dc.scenes:
                    scene.flush()

                    for actor in scene.actors:
                        actor.flush()
