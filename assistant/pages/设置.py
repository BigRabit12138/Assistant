import streamlit as st

st.set_page_config(page_title='设置', layout='wide')
st.title('设置')
st.text_input('key', value='', max_chars=None, key=None, type='default')

saved = st.button('保存')
