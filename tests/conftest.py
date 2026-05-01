import pytest
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture(autouse=True)
def clear_resource_cache():
    from pelgo.agent import tools as _tools
    _tools._RESOURCE_CACHE.clear()
    yield
    _tools._RESOURCE_CACHE.clear()
