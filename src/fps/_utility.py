from typing import Union, Generator


def _as_generator(r: Union[float, Generator]):
    """A wrapper to uniformly treat parameters that are constant or changing every iteration (e.g. decaying stepsize) """ 
    if isinstance(r, Generator):
        return r
    elif isinstance(r, float):
        def rgen():
            while True:
                yield r
        return rgen()
    else:
        raise RuntimeError(f"Type of relaxation/regularization ({r}) not supported")


