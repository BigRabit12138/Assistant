# import socks
# import socket
# socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 10080)
# socket.socket = socks.socksocket
# 记得把断点处改为多线程模式
import os
os.environ["OPENAI_API_KEY"] = "sk-hGf4MyFwU75NaMl5wecBzaExMmAjPSjcFSU2FVRkkOcdh1DJ"
os.environ["OPENAI_BASE_URL"] = "https://api.chatanywhere.tech/v1"
with open("sample.txt", 'r') as file:
    text = file.read()

print(text[: 100])

from assistant.memory.raptor import RetrievalAugmentation
ra = RetrievalAugmentation()
ra.add_documents(text)

question = "How did Cinderella reach her happy ending?"
answer = ra.answer_question(question=question)
print("Answer: ", answer)

save_path = 'cinderella'
ra.save(save_path)

ra = RetrievalAugmentation(tree=save_path)
answer = ra.answer_question(question=question)
print("Answer: ", answer)

