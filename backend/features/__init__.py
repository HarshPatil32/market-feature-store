# Import feature modules so @register_feature runs at package load time.
from backend.features import moving_averages as _moving_averages  # noqa: F401
from backend.features import returns as _returns  # noqa: F401
from backend.features import rsi as _rsi  # noqa: F401
from backend.features import volatility as _volatility  # noqa: F401
from backend.features import volume as _volume  # noqa: F401
