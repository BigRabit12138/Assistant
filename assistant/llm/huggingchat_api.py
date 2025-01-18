from hugchat import hugchat
from hugchat.login import Login
from langchain.llms.base import LLM
from typing import List, Optional, Any
from langchain.callbacks.manager import CallbackManagerForLLMRun

import global_var


class HuggingChatForLangchain(LLM):

    if global_var.login_with_passwd:
        sign = Login(global_var.email, global_var.passwd)
        cookies = sign.login()
        sign.saveCookiesToDir(str(global_var.cookie_path))
    else:
        sign = Login(global_var.email)
        cookies = sign.loadCookiesFromDir(str(global_var.cookie_path))

    chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
    chatbot.switch_llm(2)

    @property
    def _llm_type(self) -> str:
        return "huggingchat"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        return self.chatbot.query(text=prompt).text
