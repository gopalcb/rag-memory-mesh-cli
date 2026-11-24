from ipaddress import ip_address
from urllib.parse import urlsplit

from config import settings


def loopback_endpoint(url: str) -> bool:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or parts.username is not None or parts.password is not None:
        return False
    try:
        return ip_address(parts.hostname or "").is_loopback
    except ValueError:
        return False


def require_external_permission(operation: str, endpoint: str | None = None) -> None:
    if endpoint is not None and loopback_endpoint(endpoint):
        return
    if not settings.security.allow_external_models:
        raise ValueError(f"{operation} requires security.allow_external_models: true")
    if endpoint is not None:
        parts = urlsplit(endpoint)
        if parts.scheme != "https" or not parts.hostname or parts.username is not None or parts.password is not None:
            raise ValueError(f"{operation} requires an HTTPS endpoint without embedded credentials")
