from parking.backend.__main__ import make_app

import pytest
import testing.postgresql


@pytest.fixture(scope="module")
def postgresql():
    postgresql_con = testing.postgresql.Postgresql()
    yield postgresql_con
    postgresql_con.stop()


@pytest.fixture
def app(postgresql):
    return make_app(postgresql.url(), _init_tables=True, _reset_tables=True)
