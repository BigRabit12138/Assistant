import base64
import asyncio
import json
import websockets
import streamlit as st

from PIL import Image
from langchain.callbacks.stdout import StdOutCallbackHandler

from agents import ChatBotChain
from agents.prompts import CHATBOT_PROMPT
from llm import HuggingChatForLangchain


async def main():
    st.set_page_config(layout='wide')
    col1, col2, col3 = st.columns([0.2, 0.3, 0.5])
    if st.session_state.get('messages') is None:
        st.session_state['messages'] = []

    with col1:
        st.title('输入')
        st.header('输入')
        text = st.text_area('输入文本',
                            placeholder='你好！')
        audios_list = st.file_uploader('上传音频',
                                       type=['wav', 'mp3'],
                                       accept_multiple_files=True
                                       )
        images_list = st.file_uploader('上传图片',
                                       type=['jpeg', 'png'],
                                       accept_multiple_files=True
                                       )
        submit = st.button('发送',
                           use_container_width=True)

        if submit:

            chat = ChatBotChain(prompt=CHATBOT_PROMPT, llm=HuggingChatForLangchain())
            chat.run({'text': text, 'audios_list': audios_list, 'images_list': images_list},
                     callbacks=[StdOutCallbackHandler()])

    with col2:
        st.title('输出')
        st.header('输出')
        st.video('/home/student/projects/Assistant/resource/big_buck_bunny.mp4', start_time=0)
        st.audio('/home/student/projects/Assistant/resource/audio.wav', start_time=0)
        st.image('/home/student/projects/Assistant/resource/photo.png')
        st.text('说话输出')

    with col3:
        st.title('输出')
        st.header('输出')
        for message in st.session_state['messages']:
            if message['role'] == 'user':
                with st.chat_message('user'):
                    st.write(text)
                    st.write(audios_list)
                    st.write(images_list)
            else:
                with st.chat_message('ai'):
                    st.write('text')
                    st.write('/home/student/projects/Assistant/resource/big_buck_bunny.mp4')
                    st.write('/home/student/projects/Assistant/resource/audio.wav')


if __name__ == '__main__':
    asyncio.run(main())
