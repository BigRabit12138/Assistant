import json
import asyncio
import websockets

import global_var

connections = {}


async def handler(websocket):

    async for message in websocket:
        print(message[:200])

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


async def switcher_server():

    async with websockets.serve(handler,
                                global_var.ip,
                                global_var.port,
                                max_size=10 * 1024 * 1024,
                                ping_timeout=60 * 3):
        print('Server启动')
        await asyncio.Future()
