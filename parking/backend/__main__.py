import argparse
from asyncio import AbstractEventLoop

import testing.postgresql
import tornado

from parking.backend.db.dbaccess import DbAccess
from parking.backend.engine.alloc_engine import AllocationEngine
from parking.backend.sensor_server.rest_server import (IndividualLotDeleteHandler, IndividualLotAvailableHandler,
                                                       IndividualLotPriceHandler, ParkingLotsCreationHandler)
from parking.backend.user_server.session import UserSessions
from parking.backend.user_server.wsserver import UserWSHandler


def make_app(url: str, init_tables: bool = False, reset_tables: bool = False):
    loop: AbstractEventLoop = tornado.ioloop.IOLoop.current().asyncio_loop
    usessions: UserSessions = UserSessions()
    dba: DbAccess = tornado.ioloop.IOLoop.current().run_sync(
        lambda: DbAccess.create(url, loop=loop, init_tables=init_tables, reset_tables=reset_tables))
    engine: AllocationEngine = AllocationEngine(dba, usessions)
    app = tornado.web.Application([(r"/ws/(.*)", UserWSHandler, {'usessions': usessions, 'engine': engine}),
                                   (r'/spaces', ParkingLotsCreationHandler, {'dba': dba}),
                                   (r'/spaces/([0-9]+)', IndividualLotDeleteHandler, {'dba': dba}),
                                   (r'/spaces/([0-9]+)/available', IndividualLotAvailableHandler,
                                   {'dba': dba, 'engine': engine}),
                                   (r'/spaces/([0-9]+)/price', IndividualLotPriceHandler, {'dba': dba})])
    return app


def make_and_run_app(url: str, init_tables: bool = False, reset_tables: bool = False):
    app = make_app(url, init_tables=init_tables, reset_tables=reset_tables)
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()


def main(temp_db: bool, db_url: str, reset_tables: bool):
    if temp_db:
        with testing.postgresql.Postgresql() as postgresql:
            make_and_run_app(postgresql.url(), init_tables=True)
    else:
        make_and_run_app(db_url, reset_tables=reset_tables)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parking backend.')
    parser.add_argument("--db", default="postgresql://localhost/postgres", help="Database full url")
    parser.add_argument("--temp-db", action='store_true', help="Create and initialise a temporary database")
    parser.add_argument("--reset-tables", action='store_true', help="Drop and recreate database tables")
    args: argparse.Namespace = parser.parse_args()

    main(args.temp_db, args.db, args.reset_tables)
