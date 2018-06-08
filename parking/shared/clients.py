import asyncio
import collections
import json
from six import string_types
import logging

from tornado import httpclient, websocket
from tornado.concurrent import future_set_result_unless_cancelled

import parking.shared.rest_models as rest_models
import parking.shared.ws_models as ws_models
from parking.shared.util import serialize_model

HEADERS = {'Content-Type': 'application/json; charset=UTF-8'}

logger = logging.getLogger('shared client')


class ParkingLotRest(object):
    """
    An async client for the parking lot REST API
    """

    def __init__(self, base_url, http_client):
        self.client = http_client
        self.rest_url = f"{base_url}/spaces"

    async def create_lot(self, lot: rest_models.ParkingLot):
        request = httpclient.HTTPRequest(self.rest_url, body=serialize_model(lot), headers=HEADERS, method='POST')
        response = await self.client.fetch(request)
        return rest_models.ParkingLotCreationResponse(**json.loads(response.body)).id

    async def update_available(self, lot_id: int, available: int):
        msgbody = serialize_model(rest_models.ParkingLotAvailableMessage(available))
        request = httpclient.HTTPRequest(f"{self.rest_url}/{lot_id}/available", body=msgbody, headers=HEADERS,
                                         method='POST')
        try:
            await self.client.fetch(request)
        except httpclient.HTTPError:
            logger.info("server error while updating lot availability for lot number " + str(lot_id))
            raise

    async def update_price(self, lot_id: int, price: float):
        msgbody = serialize_model(rest_models.ParkingLotPriceMessage(price))
        request = httpclient.HTTPRequest(f"{self.rest_url}/{lot_id}/price", body=msgbody, headers=HEADERS,
                                         method='POST')
        await self.client.fetch(request)

    async def delete_lot(self, lot_id: int):
        request = httpclient.HTTPRequest(f"{self.rest_url}/{lot_id}", method='DELETE')
        await self.client.fetch(request)


class CarWebsocket(object):
    def __init__(self, ws, receive_callbacks=None):
        self._ws = ws
        self._ws._on_message_callback = self._on_message

        self._waiting = {}
        self._message_queue = collections.defaultdict(collections.deque)
        self._receive_callbacks = receive_callbacks if receive_callbacks else {}

    @classmethod
    async def create(cls, base_url, receive_callbacks=None, user_id=''):
        return cls(await websocket.websocket_connect(base_url + "/" + user_id), receive_callbacks)

    async def _send(self, message):
        if not isinstance(message, string_types):
            message = serialize_model(message)
        await self._ws.write_message(message)
        logger.info("message sent: '{}'")  # .format(message._type))

    async def send_location(self, location: rest_models.Location):
        await self._send(ws_models.LocationUpdateMessage(location))

    async def send_parking_request(self, location: rest_models.Location, preferences: dict):
        await self._send(ws_models.ParkingRequestMessage(location, preferences))

    async def send_parking_acceptance(self, lot_id: int):
        await self._send(ws_models.ParkingAcceptanceMessage(lot_id))

    async def send_parking_rejection(self, lot_id: int):
        await self._send(ws_models.ParkingRejectionMessage(lot_id))

    async def send_parking_cancellation(self, lot_id: int, reason: int):
        await self._send(ws_models.ParkingCancellationMessage(lot_id, reason))

    def _on_message(self, message: str):
        message = ws_models.deserialize_ws_message(message)
        message_type = message.__class__

        if message_type in self._receive_callbacks:
            self._receive_callbacks[message_type](message)
        elif message_type in self._waiting:
            future_set_result_unless_cancelled(self._waiting[message_type], message)
            del self._waiting[message_type]
        else:
            self._message_queue[message_type].append(message)

    def receive(self, message_type: type) -> asyncio.Future:
        future = asyncio.Future()
        if self._message_queue[message_type]:
            future_set_result_unless_cancelled(future, self._message_queue[message_type].popleft())
            logger.debug("message recieved as expected: '{}'".format(message_type))
        else:
            self._waiting[message_type] = future
            logger.debug("unexpected message received; expected: " + str(message_type))
        return future
