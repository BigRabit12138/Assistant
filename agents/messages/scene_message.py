from pydantic import Field
from pydantic import BaseModel


class SceneMessage(BaseModel):
    role: str = Field(default='')
    action: str = Field(default='')
    conversation: str = Field(default='')

    def __str__(self):
        if self.role.lower() == 'director':
            return f"{self.action}"
        else:
            return f"{self.action}\n{self.role}：”{self.conversation}“"

    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "action": self.action,
            "conversation": self.conversation
        }

