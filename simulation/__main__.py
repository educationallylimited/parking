import time
import asyncio
import argparse

from .simulation import SimManager


def main(base_url, num_spaces=5, num_cars=1, num_rogues=1, parking_seed=time.time(), car_seed=time.time()):
    print(f'Starting simulation')
    print(f'Car Seed = {car_seed!r}')
    print(f'Parking Seed = {parking_seed!r}')

    sim = SimManager(
        num_spaces=num_spaces,
        min_spaces_per_lot=5,
        max_spaces_per_lot=5,
        num_cars=num_cars,
        num_rogues=num_rogues,
        width=500, height=500,
        parking_lot_seed=parking_seed,
        car_seed=car_seed,
        max_time=100,
        app_url=base_url
    )
    asyncio.ensure_future(sim.run())
    loop = asyncio.get_event_loop()
    loop.run_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parking Simulation.')
    parser.add_argument("base_url", default="http://127.0.0.1:8888", nargs='?', help="Base URL for the Engine")
    parser.add_argument("--num-spaces", default=5, help="Number of parking spaces")
    parser.add_argument("--car-seed", default=time.time(), help="Car Initialisation seed")
    parser.add_argument("--parking-seed", default=time.time(), help="Parking Lot Intialisation seed")
    parser.add_argument("--num-cars", default=1, help="Number of agents")
    parser.add_argument("--num-rogues", default=1, help="Number of rogue agents")
    args: argparse.Namespace = parser.parse_args()

    main(
        args.base_url,
        num_spaces=args.num_spaces,
        car_seed=args.car_seed,
        parking_seed=args.parking_seed,
        num_cars=args.num_cars,
        num_rogues=args.num_rogues,
    )
