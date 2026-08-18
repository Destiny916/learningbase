from __future__ import annotations


def camera_name(key: str) -> str:
    return key.removeprefix("observation.images.")


def parse_camera_sources(specifications: list[str] | None) -> dict[str, str] | None:
    if not specifications:
        return None
    camera_sources: dict[str, str] = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"Camera source must use MODEL_INPUT=SOURCE: {specification!r}")
        key, source = (part.strip() for part in specification.split("=", 1))
        if not key or not source:
            raise ValueError(f"Camera source must use non-empty MODEL_INPUT=SOURCE: {specification!r}")
        if key in camera_sources:
            raise ValueError(f"Duplicate model image input: {key}")
        camera_sources[key] = source
    return camera_sources
