import json
from enum import IntEnum, Enum

import attr

from parking.shared.location import Location
from parking.shared.rest_models import ParkingLot
from parking.shared.util import ensure, validate_non_neg, enforce_type


class WebSocketMessageType(IntEnum):
    LOCATION_UPDATE = 1
    PARKING_REQUEST = 2
    PARKING_ALLOCATION = 3
    PARKING_ACCEPTANCE = 4
    PARKING_REJECTION = 5
    PARKING_DEALLOC = 6
    PARKING_CANCEL = 7
    ERROR = 8
    CONFIRMATION = 9


@attr.s
class WsError:
    err_type: int = attr.ib(validator=enforce_type)
    msg: str = attr.ib(validator=enforce_type)


class WebSocketErrorType(Enum):
    INVALID_MESSAGE = WsError(1, "Could not parse the message.")
    NO_AVAILABLE_PARKING_LOT = WsError(2, "No parking lot available.")
    NOT_IMPLEMENTED = WsError(3, "The handling for this message has not been implemented yet.")
    CORRUPTED_SESSION = WsError(4, "Something went wrong with this session. Please create a new one.")
    ALLOCATION_COMMIT_FAIL = WsError(5, "Failed to commit parking allocation. Please request a new parking space.")
    ANOTHER_CONNECTION_OPEN = WsError(6, "WebSocket connection already open for this user.")
    DATABASE = WsError(7, "Unexpected exception with the database. Please try again.")
    INTERNAL = WsError(8, "Unexpected internal error.")


@attr.s
class ParkingLotAllocation(ParkingLot):
    distance: float = attr.ib(validator=enforce_type, default=0.0)  # in meters
    availability: int = attr.ib(validator=enforce_type, default=0)


@attr.s
class ErrorMessage:
    error: WsError = attr.ib(converter=ensure(WsError))
    _type: int = attr.ib(default=WebSocketMessageType.ERROR.value, init=False)


@attr.s
class ConfirmationMessage:
    _type: int = attr.ib(default=WebSocketMessageType.CONFIRMATION.value, init=False)
    msg: str = attr.ib(default="Success.")


@attr.s
class LocationUpdateMessage:
    location: Location = attr.ib(converter=ensure(Location))
    _type: int = attr.ib(default=WebSocketMessageType.LOCATION_UPDATE.value, init=False)


@attr.s
class ParkingRequestMessage:
    location: Location = attr.ib(converter=ensure(Location))
    # TODO: Maybe make a preferences class so that we can validate the content
    # of preferences.
    preferences: dict = attr.ib(validator=enforce_type, factory=dict)
    _type: int = attr.ib(default=WebSocketMessageType.PARKING_REQUEST.value, init=False)


@attr.s
class ParkingAllocationMessage:
    # TODO: Maybe have an error class to validate the error.
    lot: ParkingLotAllocation = attr.ib(converter=ensure(ParkingLotAllocation))
    _type: int = attr.ib(default=WebSocketMessageType.PARKING_ALLOCATION.value, init=False)


@attr.s
class ParkingAcceptanceMessage:
    id: int = attr.ib(validator=[enforce_type, validate_non_neg])
    _type: int = attr.ib(default=WebSocketMessageType.PARKING_ACCEPTANCE.value, init=False)


@attr.s
class ParkingRejectionMessage:
    id: int = attr.ib(validator=[enforce_type, validate_non_neg])
    _type: int = attr.ib(default=WebSocketMessageType.PARKING_REJECTION.value, init=False)


@attr.s
class ParkingDeallocationMessage:
    _type: int = attr.ib(default=WebSocketMessageType.PARKING_DEALLOC.value, init=False)


@attr.s
class ParkingCancellationMessage:
    # TODO: Add a reason enum somewhere, with 0 being REASON_UNKNOWN or similar.
    id: int = attr.ib(validator=[enforce_type, validate_non_neg])
    reason: int = attr.ib(validator=enforce_type, default=0)
    _type: int = attr.ib(default=WebSocketMessageType.PARKING_CANCEL.value, init=False)


def deserialize_ws_message(data: str):
    json_data = json.loads(data)
    if '_type' not in json_data:
        raise ValueError('Missing _type')

    _type = json_data.pop('_type')

    if _type not in [e.value for e in WebSocketMessageType]:
        raise ValueError('Invalid _type: {}'.format(_type))

    return message_types[_type](**json_data)


message_types = {
    WebSocketMessageType.LOCATION_UPDATE: LocationUpdateMessage,
    WebSocketMessageType.PARKING_REQUEST: ParkingRequestMessage,
    WebSocketMessageType.PARKING_ALLOCATION: ParkingAllocationMessage,
    WebSocketMessageType.PARKING_ACCEPTANCE: ParkingAcceptanceMessage,
    WebSocketMessageType.PARKING_REJECTION: ParkingRejectionMessage,
    WebSocketMessageType.PARKING_DEALLOC: ParkingDeallocationMessage,
    WebSocketMessageType.PARKING_CANCEL: ParkingCancellationMessage,
    WebSocketMessageType.ERROR: ErrorMessage,
    WebSocketMessageType.CONFIRMATION: ConfirmationMessage
}
