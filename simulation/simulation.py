import asyncio
import websockets
import json
import random
import time
import tkinter as tk
import math
from tornado import httpclient, gen

import parking.shared.ws_models as wsmodels
import parking.shared.rest_models as restmodels
from parking.shared.util import serialize_model

resturl = 'http://127.0.0.1:5000/spaces'

class SimManager:
    def __init__(self, no_spaces, min_spaces_per_lot, max_spaces_per_lot, no_cars,
                 x, y, parking_lot_seed, car_seed, max_time):
        self.random_lot = random.Random(parking_lot_seed)
        self.random_car = random.Random(car_seed)
        self.x = x
        self.y = y
        self.no_cars = no_cars
        self.car_tasks = []
        self.space_tasks = []
        self.max_time = max_time
        self.cars = []
        self.lots = []

        count = 0
        name = 0
        while count < no_spaces:

            px = self.random_lot.randint(0, self.x-1)
            py = self.random_lot.randint(0, self.y-1)

            max_al = min(max_spaces_per_lot, (no_spaces - count))
            if max_al < min_spaces_per_lot:
                n = max_al  # could potentially be smaller than min spaces per lot
            else:
                n = self.random_lot.randint(min_spaces_per_lot, max_al)
            price = round(self.random_lot.uniform(0, 10), 2)
            self.space_tasks.append(asyncio.ensure_future(space_routine(0,
                                                                        px,
                                                                        py,
                                                                        n,
                                                                        str(name),
                                                                        price,
                                                                        n,
                                                                        self)))
            count += n
            name += 1

        for i in range(self.no_cars):
            self.car_tasks.append(asyncio.ensure_future(car_routine(round(self.random_car.uniform(0, self.max_time), 1),
                                                                    self.random_car.randint(0, self.x-1),
                                                                    self.random_car.randint(0, self.y-1),
                                                                    self)))

        self.tasks = self.car_tasks + self.space_tasks

        framerate = 1/60
        root = tk.Tk()
        self.tasks.append(asyncio.ensure_future(self.run_tk(root, framerate)))
        asyncio.get_event_loop().run_until_complete(asyncio.gather(*self.tasks))

    @asyncio.coroutine
    def run_tk(self, root, interval):
        w = tk.Canvas(root, width=self.x, height=self.y)
        w.pack()
        try:
            while True:
                w.delete("all")
                now = time.time()
                for lot in self.lots:
                    # TODO no need to redraw these every time
                    lotlat, lotlong = lot.get_location()
                    w.create_rectangle(lotlat, lotlong, lotlat + 20, lotlong + 20, width=0, fill="green")

                for car in self.cars:
                    if car.state == "Roaming":
                        colour = "blue"
                    else:
                        colour = "red"
                    if car.drawing:
                        dotlat, dotlong = car.get_position(now)
                        w.create_oval(dotlat, dotlong, dotlat + 5, dotlong + 5, width=0, fill=colour)
                root.update()
                yield from asyncio.sleep(interval)
        except tk.TclError as e:
            if "application has been destroyed" not in e.args[0]:
                raise


class Waypoint:
    def __init__(self, timestamp, lat, long):
        self.time = timestamp
        self.lat = lat
        self.long = long


class Car:
    def __init__(self, lat, long):
        self.lat = lat
        self.long = long
        self.destX = 200.0
        self.destY = 200.0
        self.aDestX = 0
        self.aDestY = 0
        self.drawing = True
        self.speed = 60.0
        self.waypoints = []
        self.waypoints.append(Waypoint(time.time(), self.lat, self.long))
        newtime = time.time() + (self.distance_to(self.destX, self.destY) / self.speed)
        self.waypoints.append(Waypoint(newtime, self.destX, self.destY))
        self.state = "Roaming"

    def get_location(self):
        # TODO update to use waypoints
        return self.lat, self.long

    def distance_to(self, x, y):
        return math.sqrt((self.lat - x)**2 + (self.long - y)**2)

    def get_position(self, now):
        latdiff = self.waypoints[-1].lat - self.waypoints[-2].lat
        longdiff = self.waypoints[-1].long - self.waypoints[-2].long
        timediff = self.waypoints[-1].time - self.waypoints[-2].time
        progress = (now - self.waypoints[-2].time) / timediff
        progress = min(progress, 1)
        poslat = self.waypoints[-2].lat + (latdiff * progress)
        poslong = self.waypoints[-2].long + (longdiff * progress)
        self.lat = poslat
        self.long = poslong
        return poslat, poslong


    def set_initial_destination(self, x, y):
        self.destX = x
        self.destY = y
        # for the moment, assuming no congestion and constant speed, we can calculate the arrival time
        # however might need to pass it in when things get more complicated
        newtime = time.time() + (self.distance_to(x, y) / self.speed)
        self.waypoints.append(Waypoint(newtime, x, y))

    def get_initial_destination(self):
        return self.destX, self.destY

    def set_allocated_destination(self, x, y):
        self.aDestX = x
        self.aDestY = y
        lat, long = self.get_position(time.time())
        self.waypoints.append(Waypoint(time.time(), lat, long))
        newtime = time.time() + (self.distance_to(x, y) / self.speed)
        self.waypoints.append(Waypoint(newtime, x, y))

    def get_allocated_destination(self):
        return self.aDestX, self.aDestY


class ParkingLot:
    def __init__(self, lat: float, long: float,
                 capacity: int, name: str, price: float, newid: int, client, available: int = 0):
        if (capacity < 1):
            raise ValueError("Parking capacity must be positive")

        if available > capacity:
            raise ValueError("Capacity has to be greater than available spaces")

        self.lat: float = lat
        self.long: float = long
        self.capacity: int = capacity
        self.available: int = available
        self.cars = []
        self.name = name
        self.price = price
        self.id = newid
        self.client = client

    def get_location(self):
        return self.lat, self.long

    def get_capacity(self):
        return self.capacity

    def get_available(self):
        return self.available

    def allocate_space(self, car_id: int) -> bool:
        # TODO should there be less available as soon as an allocation is made?
        # TODO I think this should only be when a car parks
        if (car_id not in self.cars) & self.available > 0:
            self.cars.append(car_id)
            self.available -= 1
            return True
        else:
            return False

    @gen.coroutine
    def fill_space(self) -> bool:
        if self.available > 0:
            msgbody = serialize_model(restmodels.SpaceAvailableMessage(self.available - 1))
            request = httpclient.HTTPRequest(resturl + "/:" + str(self.id) + "/available", body=msgbody, method='POST')
            response = yield self.client.fetch(request)
            if response.error:
                pass
            self.available -= 1
            return True
        else:
            return False

    def free_space(self) -> bool:
        if self.available < self.capacity:
            self.available += 1
            return True
        else:
            return False

    @gen.coroutine
    def change_price(self, new_price):
        self.price = new_price
        msgbody = serialize_model(restmodels.SpacePriceMessage(new_price))
        request = httpclient.HTTPRequest(resturl + "/:" + str(self.id) + "/price", body=msgbody, method='POST')
        response = yield self.client.fetch(request)

        if response.error:
            pass
        return True

    @gen.coroutine
    def delete(self):
        request = httpclient.HTTPRequest(resturl + "/:" + str(self.id), method='DELETE')
        response = yield self.client.fetch(request)
        if response.error:
            pass

    @gen.coroutine
    def change_availability(self, value):
        if value > self.capacity | value < 0:
            raise ValueError("Availability must be positive and no greater than the capacity")

        if not(isinstance(value, int)):
            raise TypeError("Availability must be an integer")

        msgbody = serialize_model(restmodels.SpaceAvailableMessage(value))
        request = httpclient.HTTPRequest(resturl + "/:" + str(self.id) + "/available", body=msgbody, method='POST')
        response = yield self.client.fetch(request)

        if response.error == "200":
            self.available = value
            return True
        else:
            return False


async def car_routine(startt, startx, starty, manager):
    await asyncio.sleep(startt)

    car = Car(startx, starty)
    manager.cars.append(car)

    print(str(car.waypoints[-2].time), str(car.waypoints[-2].lat), str(car.waypoints[-2].long))
    print(str(car.waypoints[-1].time), str(car.waypoints[-1].lat), str(car.waypoints[-1].long))

    await asyncio.sleep(10)
    car.state = "Searching"

    async with websockets.connect('ws://localhost:8765') as websocket:

        # Request a parking space
        x, y = car.get_initial_destination()
        message = wsmodels.ParkingRequestMessage(wsmodels.Location(float(x), float(y)))
        await websocket.send(serialize_model(message))

        # Recieve the parking allocation information
        space = await websocket.recv()
        spacedict = json.loads(space)
        lotdict = spacedict['lot']
        locdict = lotdict['location']
        car.set_allocated_destination(locdict['longitude'], locdict['latitude'])

        print(str(car.waypoints[-2].time), str(car.waypoints[-2].lat), str(car.waypoints[-2].long))
        print(str(car.waypoints[-1].time), str(car.waypoints[-1].lat), str(car.waypoints[-1].long))

        while True:
            # Send the location of the car at time intervals
            await asyncio.sleep(3)
            x, y = car.get_location()
            message = wsmodels.LocationUpdateMessage(wsmodels.Location(float(x), float(y)))
            await websocket.send(serialize_model(message))


async def space_routine(startt, lat, long, capacity, name, price, available, manager):
    await asyncio.sleep(startt)

    msgbody = serialize_model(restmodels.ParkingLot(capacity, name, price,
                                                    restmodels.Location(float(lat), float(long))))

    client = httpclient.AsyncHTTPClient()
    request = httpclient.HTTPRequest(resturl, body=msgbody, method='POST')
    await client.fetch(request)
    # newid = json.loads(response.body)["id"]
    newid = 1

    lot = ParkingLot(lat, long, capacity, name, price, newid, client, available)
    manager.lots.append(lot)

    await asyncio.sleep(10)

    lot.change_price(1.0)

if __name__ == '__main__':
    SimManager(2000, 20, 70, 50, 1000, 1000, 2, 4, 100)
