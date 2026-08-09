"""IBKR option-surface collection and GCS publication."""

__all__ = ["collect_and_publish", "publish_option_chain"]


def __getattr__(name: str):
    """Avoid importing IBKR/regime-app dependencies for contract-only consumers."""
    if name in __all__:
        from .publisher import collect_and_publish, publish_option_chain

        return {"collect_and_publish": collect_and_publish, "publish_option_chain": publish_option_chain}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
