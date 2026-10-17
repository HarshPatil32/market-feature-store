# Import feature modules so @register_feature runs at package load time.
from backend.features import returns as _returns  # noqa: F401
from backend.features import volatility as _volatility  # noqa: F401
