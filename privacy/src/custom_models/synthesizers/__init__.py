"""Synthesizers module."""

from ctgan import CTGAN as CTGANSynthesizer
from ctgan import TVAE as TVAESynthesizer

__all__ = (
    'CTGANSynthesizer',
    'TVAESynthesizer'
)


def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
