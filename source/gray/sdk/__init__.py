"""Compatibility imports for the public API now distributed as gray-sdk."""

try:
    import gray_sdk as _sdk
except ModuleNotFoundError as error:
    if error.name != "gray_sdk":
        raise
    raise ModuleNotFoundError(
        "Install gray-sdk to use the Gray inference API.", name="gray_sdk"
    ) from error

__all__ = _sdk.__all__
globals().update({name: getattr(_sdk, name) for name in __all__})
