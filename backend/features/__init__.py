# Import feature modules so @register_feature runs at package load time.
from backend.features import returns as _returns  # noqa: F401
