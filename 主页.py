import socks
import socket

socks.set_default_proxy(socks.SOCKS5, **{
    'addr': '127.0.0.1',
    'port': 10808
})
socket.socket = socks.socksocket
import streamlit as st
import time

print('GG')
