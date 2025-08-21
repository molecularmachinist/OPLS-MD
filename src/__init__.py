__version__ = "0.1.0.dev0"

from .OPLS import OPLS
from .PLS import PLS
from .OPLS_PLS import OPLS_PLS
try:
    from .OPLS_MD import OPLS_MD, PLS_MD, OPLS_PLS_MD
except ImportError as e:
    class _FAILEDIMPORT():
        error = e

        def __init__(self, *args, **kwargs):
            raise _FAILEDIMPORT.error

    OPLS_MD = _FAILEDIMPORT  # type: ignore
    PLS_MD = _FAILEDIMPORT  # type: ignore
    OPLS_PLS_MD = _FAILEDIMPORT  # type: ignore
