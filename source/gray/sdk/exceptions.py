"""Stable exception categories for SDK callers."""


class GraySDKError(Exception):
    """Base for SDK operational and validation errors."""


class ManifestError(GraySDKError):
    """Invalid or unsupported model contract."""


class InputError(GraySDKError):
    """Input violates the explicit array contract."""


class BackendError(GraySDKError):
    """Backend dependency, device, or model initialization failure."""


class InferenceError(GraySDKError):
    """Inference execution or output validation failed."""


class RuntimeClosedError(GraySDKError):
    """A closed runtime cannot execute predictions."""
