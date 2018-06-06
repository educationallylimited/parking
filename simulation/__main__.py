import time
import asyncio
import argparse

from .simulation import SimManager


def main(base_url, num_spaces, num_cars, num_rogues, spaces_per_lot, parking_seed, car_seed):
    print(f'Starting simulation')
    print(f'Car Seed = {car_seed!r}')
    print(f'Parking Seed = {parking_seed!r}')

    sim = SimManager(
        num_spaces=num_spaces,
        min_spaces_per_lot=spaces_per_lot,
        max_spaces_per_lot=spaces_per_lot,
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
    parser.add_argument("base_url", default="http://127.0.0.1:8888", nargs='?', help="Base URL for the engine server")
    parser.add_argument("--num-spaces", type=int, default=5, help="Number of parking lots")
    parser.add_argument("--spaces-per-lot", type=int, default=5, help="Number of parking spaces per lot")
    parser.add_argument("--car-seed", type=int, default=time.time(), help="Car Initialisation seed")
    parser.add_argument("--parking-seed", type=int, default=time.time(), help="Parking Lot Intialisation seed")
    parser.add_argument("--num-cars", type=int, default=1, help="Number of agents")
    parser.add_argument("--num-rogues", type=int, default=1, help="Number of rogue agents")
    args: argparse.Namespace = parser.parse_args()

    main(
        args.base_url,
        num_spaces=args.num_spaces,
        spaces_per_lot=args.spaces_per_lot,
        car_seed=args.car_seed,
        parking_seed=args.parking_seed,
        num_cars=args.num_cars,
        num_rogues=args.num_rogues,
    )
