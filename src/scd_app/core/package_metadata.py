"""Application package metadata shared by diagnostics and audit reports."""

from importlib.metadata import PackageNotFoundError, version

APP_DISTRIBUTION = "scd-studio"
LEGACY_DISTRIBUTION = "scd-edition"


def installed_app_version() -> str:
    """Return the installed app version, including legacy installations."""
    for distribution in (APP_DISTRIBUTION, LEGACY_DISTRIBUTION):
        try:
            return version(distribution)
        except PackageNotFoundError:
            continue
    return "development checkout"
