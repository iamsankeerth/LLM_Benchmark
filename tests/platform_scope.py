"""Platform scope for evidence-integrity assertions.

Recorded artifact digests in the sealed freeze files were captured from a
Windows checkout, where git materializes CRLF line endings. A POSIX checkout
materializes LF for the same committed blobs, so the byte-level sha256 of those
text artifacts differs even though the content is identical. The same applies
to population hashes derived from those files.

Rather than restating the recorded digests, which would silently re-baseline the
evidence trail, the integrity assertions that consume them are scoped to the
platform that can reproduce the recorded form. This is a known, accepted
limitation and is documented in README.md under Limitations.

These skips are deliberately visible: unittest reports them as skipped tests, so
the reduced Linux/macOS coverage is observable in CI output. Do not convert them
into silent conditionals or blanket early returns.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

WINDOWS_DIGEST_RECORDINGS = (
    'recorded artifact digests are the Windows CRLF materialization; a POSIX '
    'checkout yields LF bytes for identical content, so the recorded sha256 is '
    'only reproducible on a Windows checkout'
)

requires_windows_digests = unittest.skipUnless(
    sys.platform == 'win32', WINDOWS_DIGEST_RECORDINGS
)


def requires_repo_paths(
    root: Path, reason: str, *relative_paths: str
) -> object:
    """Skip when a required repository file is absent from the checkout.

    Used for evidence that project policy keeps out of git, so that a fresh
    clone does not fail on a file it was never meant to receive.
    """
    missing = tuple(path for path in relative_paths if not (root / path).is_file())
    return unittest.skipIf(
        bool(missing),
        f'{reason} (not in this checkout: {", ".join(missing)})',
    )
