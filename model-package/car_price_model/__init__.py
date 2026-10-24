from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("car-price-model")
except PackageNotFoundError:
    __version__ = "0.1.0"
