from __future__ import annotations

from semver import Version

from .errors import BinsparseParseError

BINSPARSE_VERSION = "0.1.0"
_SUPPORTED_VERSION = Version.parse(BINSPARSE_VERSION)


def check_binsparse_version(version: object) -> None:
    """Raise if *version* is not a SemVer-compatible Binsparse version."""
    parsed = parse_binsparse_version(version)
    if not _SUPPORTED_VERSION.is_compatible(parsed):
        raise BinsparseParseError(
            f"unsupported Binsparse version {version!r}; "
            f"expected a SemVer-compatible version with {BINSPARSE_VERSION!r}"
        )


def parse_binsparse_version(version: object) -> Version:
    if not isinstance(version, str):
        raise BinsparseParseError("Binsparse version must be a string")
    try:
        return Version.parse(version)
    except ValueError as error:
        raise BinsparseParseError(
            f"invalid Binsparse SemVer version {version!r}"
        ) from error
