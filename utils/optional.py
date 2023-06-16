from typing import Optional, TypeVar

T = TypeVar("T")


def or_else(x: Optional[T], y: T) -> T:
    return x if x is not None else y
