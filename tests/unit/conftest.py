"""Unit-suite fixtures shared across ``tests/unit/``.

Stubs the optional backend SDKs the unit tests reference, so the unit suite
never depends on them. Some backend modules hard-import their SDK at module
load to fail-fast for real users (e.g. ``MirageBackend`` does
``from mirage import Workspace``; ``DaytonaBackend`` does
``from daytona_sdk import FileUpload, FileDownloadRequest``). Their unit tests
drive the backend through in-file fakes and never touch the real SDK, but the
module-level import would still break collection when the SDK is absent.

For each such SDK we register a minimal stub *only when the real package is
missing*, leaving genuine installs untouched. Each stub carries a sentinel so
the integration suite can tell it apart from a real install and refuse to run
conformance against it.
"""

from __future__ import annotations

import sys
import types

# SDK module name -> attribute names the importing backend module needs at
# import time. The attributes only need to exist (the unit tests never
# instantiate them); behavior is exercised through in-file fakes.
_OPTIONAL_SDK_STUBS = {
    "mirage": ("Workspace",),
    "daytona_sdk": ("FileUpload", "FileDownloadRequest"),
}


def _install_optional_sdk_stubs() -> None:
    for name, attrs in _OPTIONAL_SDK_STUBS.items():
        if name in sys.modules:
            continue
        try:
            __import__(name)
        except ImportError:
            stub = types.ModuleType(name)
            for attr in attrs:
                setattr(stub, attr, type(attr, (), {}))
            stub.__agentkit_test_stub__ = True
            sys.modules[name] = stub


_install_optional_sdk_stubs()
