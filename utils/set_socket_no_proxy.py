import socks
import socket
import contextlib

import global_var


@contextlib.contextmanager
def socket_no_proxy():
    """临时取消全局代理的上下文管理器"""

    # 取消全局代理设置
    socks.set_default_proxy()
    socket.socket = global_var.default_socket
    try:
        yield
    finally:
        # 恢复全局设置
        socks.set_default_proxy(socks.SOCKS5, **global_var.proxy_setting)
        socket.socket = socks.socksocket
