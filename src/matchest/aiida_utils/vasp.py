"""
VASP relaxed utility functions
"""

from typing import List
from aiida import orm

try:
    from aiida_vasp.parsers.content_parsers.potcar import MultiPotcarIo
except ImportError:
    from aiida_vasp.parsers.file_parsers.potcar import MultiPotcarIo


def get_functional(incar: dict, pot: str) -> str:
    """
    Return the name of the functional

    Args:
        incar (dict): A dictionary for setting the INCAR
        pot (str): Potential family
    """
    if incar.get("metagga"):
        return incar.get("metagga").lower()

    if pot.startswith("LDA"):
        if incar.get("gga"):
            return "gga+ldapp"
        else:
            return "lda"
    elif pot.startswith("PBE"):
        gga = incar.get("gga")
        hf = incar.get("lhfcalc")
        if not hf:
            if (not gga) or gga.lower() == "pe":
                return "pbe"
            if gga.lower() == "ps":
                return "pbesol"
        elif (not gga) or gga.lower() == "pe":
            if incar.get("aexx") in [0.25, None] and (
                incar.get("hfscreen") - 0.2 < 0.01
            ):
                return "hse06"

    return "unknown"


def get_u_elem(struc, ldauu, elem):
    """
    Reliably get the value of U for a given element.
    Return -1 if the entry does not have the element - so compatible with any U calculations
    """
    species = MultiPotcarIo.potentials_order(struc)
    if elem in species:
        ife = species.index(elem)
        if ldauu is None:
            return 0.0
        return ldauu[ife]
    return -1


def get_u_map(struc: orm.StructureData, ldauu: List[int]) -> dict:
    """
    Reliably get the value of U for all elements.
    Return -1 if the entry does not have Fe - so compatible with any U calculations
    """
    species = MultiPotcarIo.potentials_order(struc)
    mapping = {}
    for symbol in species:
        isym = species.index(symbol)
        if ldauu is None:
            mapping[symbol] = 0.0
        else:
            mapping[symbol] = ldauu[isym]
    return mapping
