import asyncio
import random
import time
import tkinter as tk
import math
from tornado import httpclient
from concurrent import futures

import parking.shared.ws_models as wsmodels
import parking.shared.rest_models as restmodels
from parking.shared.clients import CarWebsocket, ParkingLotRest

resturl = 'http://127.0.0.1:5000'


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
        self.stop_flag = False

        count = 0
        name = 0
        while count < no_spaces:
            if self.stop_flag:
                break

            px = self.random_lot.randint(0, self.x-1)
            py = self.random_lot.randint(0, self.y-1)

            max_al = min(max_spaces_per_lot, (no_spaces - count))
            if max_al < min_spaces_per_lot:
                n = max_al  # could potentially be smaller than min spaces per lot
            else:
                n = self.random_lot.randint(min_spaces_per_lot, max_al)
            price = round(self.random_lot.uniform(0, 10), 2)
            spacero = space_routine(0, px, py, n, str(name), price, n, self)
            self.space_tasks.append(asyncio.ensure_future(spacero))

            count += n
            name += 1

        for i in range(self.no_cars):
            start_time = round(self.random_car.uniform(0, self.max_time), 1)
            start_x = self.random_car.randint(0, self.x - 1)
            start_y = self.random_car.randint(0, self.y - 1)
            coro = car_routine(start_time, start_x, start_y, self)
            self.car_tasks.append(asyncio.ensure_future(coro))

        self.tasks = self.car_tasks + self.space_tasks

    async def run_tk(self, root, interval):
        w = tk.Canvas(root, width=self.x, height=self.y)
        w.pack()
        try:
            while True:
                w.delete("all")
                now = time.time()
                for simlot in self.lots:
                    centering = (10 + (simlot.available / 2) / 2)
                    lotlat, lotlong = simlot.lot.location.latitude, simlot.lot.location.longitude
                    w.create_rectangle(lotlat - centering, lotlong - centering, lotlat + centering, lotlong + centering,
                                       width=0,
                                       fill="green")

                for car in self.cars:
                    if car.state == "Roaming":
                        colour = "blue"
                    elif car.state == "Searching":
                        colour = "red"
                    else:
                        colour = "black"
                    if car.drawing:
                        dotlat, dotlong = car.get_position(now)
                        w.create_oval(dotlat - 2.5, dotlong - 2.5, dotlat + 2.5, dotlong + 2.5, width=0, fill=colour)
                root.update()
                await asyncio.sleep(interval)
                if self.stop_flag:
                    break
            root.destroy()
        except tk.TclError as e:
            if "application has been destroyed" not in e.args[0]:
                raise

    async def run(self):
        framerate = 1 / 60
        root = tk.Tk()
        await self.run_tk(root, framerate)
        #self.tasks.append(asyncio.ensure_future(self.run_tk(root, framerate)))
        await asyncio.gather(*self.tasks)

    async def stop(self, delay):
        await asyncio.sleep(delay)
        self.stop_flag = True
        asyncio.get_event_loop().stop()


class Waypoint:
    def __init__(self, timestamp, lat, long):
        self.time = timestamp
        self.lat = lat
        self.long = long


class Car:
    def __init__(self, lat, long):
        self.lat = lat
        self.long = long
        self.destX = 200
        self.destY = 300
        self.aDestX = 0
        self.aDestY = 0
        self.drawing = False
        self.speed = 60
        self.waypoints = []
        self.waypoints.append(Waypoint(time.time(), self.lat, self.long))
        newtime = time.time() + (self.distance_to(self.destX, self.destY) / self.speed)
        self.waypoints.append(Waypoint(newtime, self.destX, self.destY))
        self.state = "Roaming"

    def distance_to(self, x, y):
        return math.sqrt((self.lat - x)**2 + (self.long - y)**2)

    def get_position(self, now):
        if len(self.waypoints) != 2:
            latdiff = self.waypoints[-1].lat - self.waypoints[-2].lat
            longdiff = self.waypoints[-1].long - self.waypoints[-2].long
            timediff = self.waypoints[-1].time - self.waypoints[-2].time
            progress = (now - self.waypoints[-2].time) / timediff
            progress_limit = min(progress, 1)

            if progress_limit == 1:
                self.state = "Parked"
                self.finish = time.time()

            poslat = self.waypoints[-2].lat + (latdiff * progress_limit)
            poslong = self.waypoints[-2].long + (longdiff * progress_limit)
            self.lat = poslat
            self.long = poslong
            return poslat, poslong
        else:
            latdiff = self.waypoints[-1].lat - self.waypoints[-2].lat
            longdiff = self.waypoints[-1].long - self.waypoints[-2].long
            timediff = self.waypoints[-1].time - self.waypoints[-2].time
            progress = (now - self.waypoints[-2].time) / timediff
            poslat = self.waypoints[-2].lat + (latdiff * progress)
            poslong = self.waypoints[-2].long + (longdiff * progress)
            self.lat = poslat
            self.long = poslong
            return poslat, poslong

    def set_initial_destination(self, x, y):
        self.destX = x
        self.destY = y
        newtime = time.time() + (self.distance_to(x, y) / self.speed)
        self.waypoints.append(Waypoint(newtime, x, y))

    def set_allocated_destination(self, x, y):
        self.aDestX = x
        self.aDestY = y
        lat, long = self.get_position(time.time())
        self.waypoints.append(Waypoint(time.time(), lat, long))
        newtime = time.time() + (self.distance_to(x, y) / self.speed)
        self.waypoints.append(Waypoint(newtime, x, y))
        self.state = "Searching"


class ParkingLot:
    def __init__(self, lot: restmodels.ParkingLot, client: ParkingLotRest, available: int = 0):

        self.lot = lot

        if (self.lot.capacity < 1) | (available < 1):
            raise ValueError("Parking capacity/availability must be positive")

        if (not(isinstance(self.lot.capacity, int))) | (not(isinstance(available, int))):
            raise TypeError("Capacity/availability must be an integer")

        if available > self.lot.capacity:
            raise ValueError("Capacity has to be greater than available spaces")

        self.available: int = available
        self.cars = []
        self.client = client

    async def fill_space(self) -> bool:
        if self.available > 0:
            response = await self.client.update_available(self.lot.id, self.available - 1)
            if response.error:
                pass
            self.available -= 1
            return True
        else:
            return False

    def free_space(self) -> bool:
        if self.available < self.lot.capacity:
            self.available += 1
            return True
        else:
            return False

    async def change_price(self, new_price):
        self.lot.price = new_price
        response = await self.client.update_price(self.lot.id, new_price)
        # TODO error handling
        # if response.error:
        #     pass

    async def delete(self):
        response = await self.client.delete_lot(self.lot.id)
        # if response.error:
        #     pass

    async def change_availability(self, value):
        if value > self.lot.capacity | value < 0:
            raise ValueError("Availability must be positive and no greater than the capacity")

        if not(isinstance(value, int)):
            raise TypeError("Availability must be an integer")

        response = await self.client.update_available(self.lot.id, value)

        # if response.error == "200":
        #     self.available = value
        #     return True
        # else:
        #     return False


async def car_routine(startt, startx, starty, manager):
    await asyncio.sleep(startt)

    car = Car(startx, starty)
    manager.cars.append(car)

    x, y = car.aDestX, car.aDestY
    cli = await CarWebsocket.create(base_url="ws://localhost:8765")
    # request a parking space
    await cli.send_parking_request(wsmodels.Location(float(x), float(y)), {})
    car.drawing = True

    await asyncio.sleep(7)

    # Receive the parking allocation information
    space = await cli.receive(wsmodels.ParkingAllocationMessage)
    car.set_allocated_destination(space.lot.location.longitude, space.lot.location.latitude)

    await cli.send_parking_acceptance(space.lot.id)

    # TODO await confirmation of acceptance

    while True:
        # Send the location of the car at time intervals, while listening for deallocation
        if manager.stop_flag:
            break
        try:
            deallocation = await asyncio.shield(asyncio.wait_for(cli.receive(wsmodels.ParkingCancellationMessage), 3))
        except futures.TimeoutError:
            deallocation = None
        x, y = car.get_position(time.time())
        await cli.send_location(wsmodels.Location(float(x), float(y)))

        if car.state == "Parked":
            await asyncio.sleep(5)
            car.drawing = False
            break


async def space_routine(startt, lat, long, capacity, name, price, available, manager):
    await asyncio.sleep(startt)

    cli = ParkingLotRest(resturl, httpclient.AsyncHTTPClient())
    lot = restmodels.ParkingLot(capacity, name, price, restmodels.Location(float(lat), float(long)))
    response = await cli.create_lot(lot)
    lot.id = response

    simlot = ParkingLot(lot, cli, capacity)
    manager.lots.append(simlot)

    await asyncio.sleep(10)

    await simlot.change_price(1.0)

if __name__ == '__main__':
    sim = SimManager(1500, 20, 70, 50, 1000, 1000, 0, 4, 100)
    asyncio.ensure_future(sim.run())
    #stop simulation after 10 seconds
    asyncio.ensure_future(sim.stop(100))
    asyncio.get_event_loop().run_forever()
    #can access sim variables here for tests

