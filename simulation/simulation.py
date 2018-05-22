import asyncio
import websockets
import json
import random
import numpy as np
import requests
import sys
import time
import tkinter as tk
import math

from parking.shared.ws_models import *

#cars = []
resturl = 'https://a6fbc847-6668-47c6-8169-2effb5444a13.mock.pstmn.io/spaces'
    
class SimManager:
    def __init__(self, no_spaces, min_spaces_per_lot, max_spaces_per_lot, no_cars,
                 x, y, parking_lot_seed, car_seed, max_time):
        random.seed(parking_lot_seed)
        self.random_lot = random.getstate()
        random.seed(car_seed)
        self.random_car = random.getstate()
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
            
            px = self.random_lot_randint(0, self.x-1)
            py = self.random_lot_randint(0, self.y-1)

            max_al = min(max_spaces_per_lot, (no_spaces - count))
            if max_al < min_spaces_per_lot:
                n = max_al # could potentially be smaller than min spaces per lot
            else:
                n = self.random_lot_randint(min_spaces_per_lot, max_al)
            price = round(self.random_lot_uniform(0,10),2)
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
            self.car_tasks.append(asyncio.ensure_future(car_routine(round(self.random_car_uniform(0,self.max_time),1),
                                                                    self.random_car_randint(0, self.x-1),
                                                                    self.random_car_randint(0, self.y-1),
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
                    lotlat, lotlong = lot.get_location()
                    w.create_rectangle(lotlat, lotlong, lotlat + 20, lotlong + 20, width=0, fill="green")
                
                for car in self.cars:
                    if car.drawing:
                        dotlat, dotlong = car.get_position(now)
                        w.create_oval(dotlat, dotlong, dotlat + 5, dotlong + 5, width=0, fill='blue')
                root.update()
                yield from asyncio.sleep(interval)
        except tk.TclError as e:
            if "application has been destroyed" not in e.args[0]:
                raise
    
    def random_car_randint(self, a: int, b: int) -> int:
        random.setstate(self.random_car)
        output = random.randint(a, b)
        self.random_car = random.getstate()
        return output
    
    def random_car_uniform(self, a: int, b: int) -> float:
        random.setstate(self.random_car)
        output = random.uniform(a, b)
        self.random_car = random.getstate()
        return output
    
    def random_lot_randint(self, a:int, b: int) -> int:
        random.setstate(self.random_lot)
        output = random.randint(a, b)
        self.random_lot = random.getstate()
        return output
    
    def random_lot_uniform(self, a: int, b: int) -> float:
        random.setstate(self.random_lot)
        output = random.uniform(a, b)
        self.random_lot = random.getstate()
        return output
         


class Waypoint:
    def __init__(self, time, lat, long):
        self.time = time
        self.lat = lat
        self.long = long


class Car:
    def __init__(self, lat, long):
        self.lat = lat
        self.long = long
        self.destX = 0
        self.destY = 0
        self.aDestX = 0
        self.aDestY = 0
        self.drawing = False
        self.speed = 60
        self.waypoints = []
        self.waypoints.append(Waypoint(time.time(), self.lat, self.long))

    def get_location(self):
        return self.lat, self.long

    def distance_to(self, x, y):
        return math.sqrt((self.lat - x)**2 + (self.long - y)**2)

    def get_position(self, now):
        if len(self.waypoints) > 1:
            latdiff = self.waypoints[-1].lat - self.waypoints[-2].lat
            longdiff = self.waypoints[-1].long - self.waypoints[-2].long
            timediff = self.waypoints[-1].time - self.waypoints[-2].time
            progress = (now - self.waypoints[-1].time) / timediff
            poslat = self.waypoints[-2].lat + (latdiff * progress)
            poslong = self.waypoints[-2].long + (longdiff * progress)
            return poslat, poslong
        else:
            return self.lat, self.long

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
        newtime = time.time() + (self.distance_to(x, y) / self.speed)
        self.waypoints.append(Waypoint(newtime, x, y))

    def get_allocated_destination(self):
        return self.aDestX, self.aDestY


class ParkingLot:
    def __init__(self, lat: float, long: float,
                 capacity: int, name: str, price: int, available: int = 0):
        if (capacity < 1) | (available < 1):
            raise ValueError("Parking capacity/availability must be positive")
            
        if (not(isinstance(capacity,int))) | (not(isinstance(available,int))):
            raise TypeError("Capacity/availability must be an integer")
            
        if available > capacity:
            raise ValueError("Capacity has to be greater than available spaces")
            
        self.lat: float = lat
        self.long: float = long
        self.capacity: int = capacity
        self.available: int = available
        self.cars = []
        self.name = name
        self.price = price
        self.id = None

        body = {}
        data = {"capacity": capacity}
        location = {"latitude": lat, "longitude": long}
        data["location"] = location
        data["name"] = name
        data["price"] = price
        body["data"] = data
        
        self.data_to_send = json.dumps(body)

        self.status = requests.post(resturl, data=self.data_to_send)
        self.id = json.loads(self.status.text)["id"]

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

    def fill_space(self) -> bool:
            if self.available > 0:
                data = {"available": (self.available - 1)}
                status = requests.post(resturl + "/:" + str(self.id), data=json.dumps(data))
                if status.status_code != "200":
                    pass
                    # TODO server error handling
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

    def change_price(self, new_price):
        self.price = new_price
        data = {"price": self.price}
        # TODO clean up repeated code
        status = requests.post(resturl + "/:" + str(self.id), data=json.dumps(data))

        if status.status_code != "200":
            pass
            # TODO server error handling
        return True

    def delete(self):
        # TODO actual deletion
        status = requests.delete(resturl + "/:" + self.id)
        if status.status_code != "200":
            pass
            # TODO server error handling
    
    def change_availability(self, value):
        if value > self.capacity | value < 0:
            raise ValueError("Availability must be positive and no greater than the capacity")
        
        if (not(isinstance(value, int))):
            raise TypeError("Availability must be an integer")
        
        status = requests.post(resturl + "/:" + str(self.id) + "/available", data=json.dumps({"available": value}))
        
        
        if status.status_code == "200":
            self.available = value
            return True
        else:
            #TODO specific error handling i.e 404 or 405 etc
            return False
        


async def car_routine(startt, startx, starty, manager):
    await asyncio.sleep(startt)

    car = Car(startx, starty)
    manager.cars.append(car)

    async with websockets.connect('ws://localhost:8765') as websocket:
        car.drawing = True

        # Send the destination of the car to the server
        x, y = car.get_initial_destination()
        # await websocket.send(json.dumps({'location': {'latitude': x, 'longitude': y}, 'preferences': {}, '_type': 2},
        #                                indent=4,
        #                                separators=(',', ': ')))
        message = LocationUpdateMessage(Location(float(x), float(y)))
        await websocket.send(json.dumps(attr.asdict(message)))

        # Recieve the parking allocation information
        space = await websocket.recv()
        spacedict = json.loads(space)
        lotdict = spacedict['lot']
        locdict = lotdict['location']
        # message = deserialize_ws_message(space)
        car.set_allocated_destination(locdict['longitude'], locdict['latitude'])

        while True:
            # Send the location of the car at time intervals
            x, y = car.get_location()
            await asyncio.sleep(3)
            #await websocket.send(json.dumps({'location': {'latitude': x, 'longitude': y}, '_type': 1},
            #                                indent=4,
            #                                separators=(',', ': ')))
            message = LocationUpdateMessage(Location(float(x), float(y)))
            await websocket.send(json.dumps(attr.asdict(message)))


async def space_routine(startt, lat, long, capacity, name, price, available, manager):
    await asyncio.sleep(startt)
    manager.lots.append(ParkingLot(lat, long, capacity, name, price, available))

# mymap = Map((0, 1000), (0, 1000), 3000, 5, 50)
# space_tasks = [asyncio.ensure_future(space_routine(0, 0, 0, 10, "space", 0, 10))]
    '''
car_tasks = []
space_tasks = []

for i in range(1000):
    car_tasks.append(asyncio.ensure_future(car_routine((i/10), random.randint(0, 600), random.randint(0, 400))))

tasks = car_tasks + space_tasks'''

'''
@asyncio.coroutine
def run_tk(root, interval):
    w = tk.Canvas(root, width=600, height=400)
    w.pack()
    try:
        while True:
            w.delete("all")
            now = time.time()
            for car in cars:
                if car.drawing:
                    dotlat, dotlong = car.get_position(now)
                    w.create_oval(dotlat, dotlong, dotlat + 5, dotlong + 5, width=0, fill='blue')
            root.update()
            yield from asyncio.sleep(interval)
    except tk.TclError as e:
        if "application has been destroyed" not in e.args[0]:
            raise
            '''

'''
framerate = 1/60
root = tk.Tk()
tasks.append(asyncio.ensure_future(run_tk(root, framerate)))
asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))'''

SimManager(2000, 20, 70, 50, 1000, 1000, 2, 4, 100)