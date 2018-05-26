import asyncio
import websockets
import parking.shared.ws_models as wsmodels
from parking.shared.util import serialize_model


async def hello(websocket, path):

    # receive a request
    await websocket.recv()
    # request = json.loads(request)
    # message = deserialize_ws_message(request)

    # send an allocation
    message = serialize_model(wsmodels.ParkingAllocationMessage(
        wsmodels.ParkingLot(1, "x", 1.0, wsmodels.Location(500.0, 500.0))))
    await websocket.send(message)

    while True:
        await websocket.recv()

start_server = websockets.serve(hello, 'localhost', 8765)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
