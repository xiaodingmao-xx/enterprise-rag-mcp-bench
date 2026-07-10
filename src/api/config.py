"""Configuration accessors tolerant of Settings and lightweight test doubles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def get_value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def section(settings: Any, name: str, default: Any = None) -> Any:
    return get_value(settings, name, default)


def api_settings(settings: Any) -> Any:
    return section(settings, "api", None)


def environment(settings: Any) -> str:
    api = api_settings(settings)
    value = get_value(api, "environment", None)
    if value:
        return str(value).lower()
    observability = section(settings, "observability", None)
    value = get_value(observability, "environment", None)
    if value:
        return str(value).lower()
    return "development"


def auth_settings(settings: Any) -> Any:
    api = api_settings(settings)
    api_auth = get_value(api, "auth", None)
    if api_auth is not None:
        return api_auth
    return section(section(settings, "security", None), "jwt", None)


def auth_mode(settings: Any) -> str:
    api = api_settings(settings)
    auth = get_value(api, "auth", None)
    mode = get_value(auth, "mode", None)
    if mode:
        return str(mode).lower()
    return str(get_value(section(settings, "security", None), "mode", "local-dev")).lower()


def auth_value(settings: Any, name: str, default: Any = None) -> Any:
    auth = auth_settings(settings)
    value = get_value(auth, name, None)
    if value is None:
        value = get_value(get_value(auth, "jwt", None), name, None)
    if value is not None:
        return value
    security = section(settings, "security", None)
    if name in {"secret", "issuer", "audience"}:
        return get_value(section(security, "jwt", None), name, default)
    return default


def api_section(settings: Any, name: str, default: Any = None) -> Any:
    api = api_settings(settings)
    value = get_value(api, name, None)
    if value is not None:
        return value
    return section(settings, name, default)
