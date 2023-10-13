import json
import asyncio
import websockets

from websockets.exceptions import ConnectionClosedError

import global_var

connections = {}


async def handler(websocket):
    message_json = {}
    try:
        async for message in websocket:
            print(message[:200])
            global_var.logger.info(message[:200])

            message_json = json.loads(message)
            connections[message_json['from']] = websocket
            if message_json['to'] == 'SERVER' and message_json['content'] == 'hello':
                connection_resp = {
                    'from': 'SERVER',
                    'to': message_json['from'],
                    'content': 'ok'
                }
                connection_resp = json.dumps(connection_resp)
                await connections[message_json['from']].send(connection_resp)
            else:
                assert connections[message_json['to']] is not None
                await connections[message_json['to']].send(message)
    except ConnectionClosedError as e:
        print(f"服务器与{message_json['to']}连接断开，详情查看日志：\n{e[-200:]}")
        global_var.logger.error(f"服务器与{message_json['to']}连接断开:\n{e}")
        connection_close_error_resp = {
            'from': 'SERVER',
            'to': message_json['from'],
            'status': 404,
            'content': f"服务器后端{message_json['to']}出错"
        }
        connection_close_error_resp = json.dumps(connection_close_error_resp)
        await connections[message_json['from']].send(connection_close_error_resp)


async def switcher_server():

    async with websockets.serve(handler,
                                global_var.ip,
                                global_var.port,
                                max_size=10 * 1024 * 1024,
                                ping_timeout=60 * 3):
        print('Server启动')
        await asyncio.Future()
