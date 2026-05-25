"""Top-level package exports.

Legacy SPECTRA classes import heavier scientific dependencies. Keep them lazy
so lightweight modules such as the audit CLI can import quickly.
"""

__all__ = [
    "Spectra",
    "Spectra_Property_Graph_Constructor",
    "SpectraDataset",
    "Spectral_Property_Graph",
    "FlattenedAdjacency",
    "plot_split_stats",
]


def __getattr__(name):
    if name in {"Spectra", "Spectra_Property_Graph_Constructor"}:
        from .spectra import Spectra, Spectra_Property_Graph_Constructor

        return {
            "Spectra": Spectra,
            "Spectra_Property_Graph_Constructor": Spectra_Property_Graph_Constructor,
        }[name]
    if name == "SpectraDataset":
        from .dataset import SpectraDataset

        return SpectraDataset
    if name in {"Spectral_Property_Graph", "FlattenedAdjacency", "plot_split_stats"}:
        from .utils import Spectral_Property_Graph, FlattenedAdjacency, plot_split_stats

        return {
            "Spectral_Property_Graph": Spectral_Property_Graph,
            "FlattenedAdjacency": FlattenedAdjacency,
            "plot_split_stats": plot_split_stats,
        }[name]
    raise AttributeError("module 'spectrae' has no attribute %r" % name)
