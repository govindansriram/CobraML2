import pytest


def pytest_addoption(parser):
    parser.addoption("--benchmark", action="store_true", default=False)


@pytest.fixture
def benchmark(request):
    return request.config.getoption("--benchmark")