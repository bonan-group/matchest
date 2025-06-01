"""
Common materials science related process functions to use with AiiDA
"""

from typing import List, Tuple

import numpy as np
from aiida import orm
from aiida.engine import calcfunction
from aiida.orm import (
    CalcFunctionNode,
    Node,
    QueryBuilder,
    StructureData,
)
from ase import Atoms
from ase.build import sort
from ase.neb import NEB
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer



# pylint: disable=import-outside-toplevel


@calcfunction
def refine_symmetry(struct, symprec):
    """Symmetrize a structure using Pymetgen's interface"""
    pstruc = struct.get_pymatgen()
    ana = SpacegroupAnalyzer(pstruc, symprec=symprec.value)
    ostruc = ana.get_refined_structure()
    ostruc = StructureData(pymatgen=ostruc)
    ostruc.label = struct.label + " REFINED"
    return ostruc


@calcfunction
def extend_magnetic_orderings(struct, moment_map):
    """
    Use pymatgen to compute all possible magnetic orderings for
    a sructure.

    Returns a collection with structures containing a MAGMOM attribute
    for the per-site magnetisations.
    """
    from pymatgen.analysis.magnetism import MagneticStructureEnumerator

    moment_map = moment_map.get_dict()
    pstruc = struct.get_pymatgen()
    enum = MagneticStructureEnumerator(pstruc, moment_map)
    structs = {}
    for idx, ptemp in enumerate(enum.ordered_structures):
        magmom = _get_all_spins(ptemp)
        for site in ptemp.sites:
            # This is abit hacky - I set the specie to be the element
            # This avoids AiiDA added addition Kind to reflect the spins
            site.species = site.species.elements[0].name
        astruc = StructureData(pymatgen=ptemp)
        astruc.base.attributes.set("MAGMOM", magmom)
        structs[f"out_structure_{idx:03d}"] = astruc
    return structs


def _get_all_spins(pstruc):
    """Get all the spins from pymatgen structure"""
    from pymatgen.core import Element

    out_dict = []
    for site in pstruc.sites:
        if isinstance(site.specie, Element):
            out_dict.append(0.0)
            continue
        out_dict.append(site.specie._properties.get("spin", 0.0))
    return out_dict


@calcfunction
def magnetic_structure_decorate(structure, magmom):
    """
    Create Quantum Espresso style decroated structure with
    given magnetic moments
    """
    from aiida_user_addons.common.magmapping import (
        create_additional_species,
    )

    magmom = magmom.get_list()
    assert len(magmom) == len(structure.sites), (
        f"Mismatch between the magmom ({len(magmom)}) and the nubmer of sites ({len(structure.sites)})."
    )
    old_species = [
        structure.get_kind(site.kind_name).symbol for site in structure.sites
    ]
    new_species, magmom_mapping = create_additional_species(old_species, magmom)
    new_structure = StructureData()
    new_structure.set_cell(structure.cell)
    new_structure.set_pbc(structure.pbc)
    for site, name in zip(structure.sites, new_species):
        this_symbol = structure.get_kind(site.kind_name).symbol
        new_structure.append_atom(
            position=site.position, symbols=this_symbol, name=name
        )

    # Keep the label
    new_structure.label = structure.label
    return {"structure": new_structure, "mapping": orm.Dict(dict=magmom_mapping)}


@calcfunction
def magnetic_structure_dedecorate(structure, mapping):
    """
    Remove decorations of a structure with multiple names for the same specie
    given that the decoration was previously created to give different species
    name for different initialisation of magnetic moments.
    """
    from aiida_user_addons.common.magmapping import (
        convert_to_plain_list,
    )

    mapping = mapping.get_dict()
    # Get a list of decroated names
    old_species = [structure.get_kind(site.kind_name).name for site in structure.sites]
    new_species, magmom = convert_to_plain_list(old_species, mapping)

    new_structure = StructureData()
    new_structure.set_cell(structure.cell)
    new_structure.set_pbc(structure.pbc)

    for site, name in zip(structure.sites, new_species):
        this_symbol = structure.get_kind(site.kind_name).symbol
        new_structure.append_atom(
            position=site.position, symbols=this_symbol, name=name
        )
    new_structure.label = structure.label
    return {"structure": new_structure, "magmom": orm.List(list=magmom)}


@calcfunction
def make_vac(cell, indices, supercell, **kwargs):
    """
    Make a defect containing cell

    If sorting of atoms in the supercell can be controlled with the ``sort`` keyword argument.
    """
    from ase.build import make_supercell

    atoms = cell.get_ase()
    if isinstance(supercell.get_list()[0], int):
        supercell_atoms = atoms.repeat(supercell.get_list())
    else:
        supercell_atoms = make_supercell(atoms, np.array(supercell.get_list()))

    mask = np.in1d(np.arange(len(supercell_atoms)), indices.get_list())
    supercell_atoms = supercell_atoms[
        ~mask
    ]  ## Remove any atoms in the original indices
    supercell_atoms.set_tags(None)
    supercell_atoms.set_masses(None)
    # Now I sort the supercell in the order of chemical symbols
    if kwargs.get("sort", True):
        supercell_atoms = sort(supercell_atoms)
    output = StructureData(ase=supercell_atoms)
    return output


def _make_vac_at_elem(
    cell: orm.StructureData,
    elem: orm.Str,
    excluded_sites: orm.List,
    supercell: orm.List,
    nsub: orm.Int = 1,
):
    """
    Make lots of vacancy containing cells usnig BSYM

    Use BSYM to do the job, vacancies are subsituted with P and
    later removed. Excluded sites are subsituted with S and later
    converted back to elem.
    """
    from bsym.interface.pymatgen import unique_structure_substitutions
    from pymatgen.core import Composition

    elem = elem.value

    nsub = nsub.value
    struc = cell.get_pymatgen()
    excluded = excluded_sites.get_list()

    assert "Ts" not in struc.composition
    assert "Og" not in struc.composition

    # Run subsitution
    for n, site in enumerate(struc.sites):
        if n in excluded:
            site.species = Composition("Ts")

    # Expand the supercell with S subsituted strucutre
    struc = struc * supercell.get_list()
    nelem_atoms = int(struc.composition[elem])
    unique_structure = unique_structure_substitutions(
        struc, elem, {"Og": nsub, elem: nelem_atoms - nsub}
    )
    # Convert back to normal structure
    # Remove Og as they are vacancies, Convert Ts back to elem
    for ustruc in unique_structure:
        p_indices = [
            n
            for n, site in enumerate(ustruc.sites)
            if site.species == Composition("Og")
        ]
        ustruc.remove_sites(p_indices)
        # Convert S sites back to O
        ustruc["Ts"] = elem

    output_structs = {}
    for n, s in enumerate(unique_structure):
        stmp = StructureData(pymatgen=s)
        stmp.base.attributes.set("vac_id", n)
        stmp.base.attributes.set("supercell", " ".join(map(str, supercell.get_list())))
        stmp.label = cell.label + f" V_{elem} {n}"
        output_structs[f"structure_{n:04d}"] = stmp

    return output_structs


@calcfunction
def make_vac_at_elem(
    cell: orm.StructureData,
    elem: orm.Str,
    excluded_sites: orm.List,
    supercell: orm.List,
    nsub: orm.Int = 1,
):
    return _make_vac_at_elem(cell, elem, excluded_sites, supercell, nsub)


@calcfunction
def make_vac_at_elem_and_shake(
    cell, elem, excluded_sites, supercell, shake_amp, nsub=1
):
    """
    Make lots of vacancy containing cells usnig BSYM
    In addition, we shake the nearest neighbours with that given by shake_amp.
    """
    from pymatgen.transformations.standard_transformations import (
        PerturbStructureTransformation,
    )

    vac_structure = _make_vac_at_elem(cell, elem, excluded_sites, supercell, nsub)
    output_structures = {}

    trans = PerturbStructureTransformation(distance=float(shake_amp))
    for key, value in vac_structure.items():
        stmp = value.get_pymatgen()
        shaken = StructureData(pymatgen=trans.apply_transformation(stmp))
        shaken.base.attributes.set("vac_id", value.base.attributes.get("vac_id"))
        shaken.base.attributes.set("supercell", value.base.attributes.get("supercell"))
        shaken.label = value.label + " SHAKE"
        output_structures[key] = shaken
    return output_structures


@calcfunction
def rattle(structure: orm.StructureData, amp: orm.Float) -> orm.StructureData:
    """
    Rattle the structure by a certain amplitude
    """
    native_keys = ["cell", "pbc1", "pbc2", "pbc3", "kinds", "sites", "mp_id"]
    # Keep the foreign keys as it is
    foreign_attrs = {
        key: value
        for key, value in structure.attributes.items()
        if key not in native_keys
    }
    atoms = structure.get_ase()
    atoms.rattle(amp.value)
    # Clean any tags etc
    atoms.set_tags(None)
    atoms.set_masses(None)
    # Convert it back
    out = StructureData(ase=atoms)
    out.base.attributes.set_many(foreign_attrs)
    out.label = structure.label + " RATTLED"
    return out


def res2structure_smart(file):
    """Create StructureData from SingleFileData, return existing node if there is any"""
    q = QueryBuilder()
    q.append(Node, filters={"id": file.pk})
    q.append(CalcFunctionNode, filters={"attributes.function_name": "res2structure"})
    q.append(StructureData)
    if q.count() > 0:
        print("Existing StructureData found")
        return q.first()
    else:
        return res2structure(file)


@calcfunction
def res2structure(file):
    """Create StructureData from SingleFile data"""
    from aiida.orm import StructureData

    from aiida_user_addons.tools.resutils import read_res

    with file.open(file.filename) as fhandle:
        titls, atoms = read_res(fhandle.readlines())
    atoms.set_tags(None)
    atoms.set_masses(None)
    atoms.set_calculator(None)
    atoms.wrap()
    struct = StructureData(ase=atoms)
    struct.base.attributes.set("H", titls.enthalpy)
    struct.base.attributes.set("search_label", titls.label)
    struct.label = file.filename
    return struct


@calcfunction
def get_primitive(structure):
    """Create primitive structure use pymatgen interface"""
    from aiida.orm import StructureData

    pstruct = structure.get_pymatgen()
    ps = pstruct.get_primitive_structure()
    out = StructureData(pymatgen=ps)
    out.label = structure.label + " PRIMITIVE"
    return out


@calcfunction
def get_standard_primitive(structure, **kwargs):
    """Create the standard primitive structure via seekpath"""
    from aiida.tools.data.array.kpoints import get_kpoints_path

    parameters = kwargs.get("parameters", {"symprec": 1e-5})
    if isinstance(parameters, orm.Dict):
        parameters = parameters.get_dict()

    out = get_kpoints_path(structure, **parameters)["primitive_structure"]
    out.label = structure.label + " PRIMITIVE"
    return out


@calcfunction
def spglib_refine_cell(structure, symprec):
    """Create the standard primitive structure via seekpath"""
    from aiida.tools.data.structure import (
        spglib_tuple_to_structure,
        structure_to_spglib_tuple,
    )
    from spglib import refine_cell

    structure_tuple, kind_info, kinds = structure_to_spglib_tuple(structure)

    lattice, positions, types = refine_cell(structure_tuple, symprec.value)

    refined = spglib_tuple_to_structure((lattice, positions, types), kind_info, kinds)

    return refined


@calcfunction
def get_standard_conventional(structure):
    """Create the standard primitive structure via seekpath"""
    from aiida.tools.data.array.kpoints import get_kpoints_path

    out = get_kpoints_path(structure)["conv_structure"]
    out.label = structure.label + " PRIMITIVE"
    return out


@calcfunction
def get_refined_structure(structure, symprec, angle_tolerance):
    """Create refined structure use pymatgen's interface"""
    from aiida.orm import StructureData
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    pstruct = structure.get_pymatgen()
    ana = SpacegroupAnalyzer(
        pstruct, symprec=symprec.value, angle_tolerance=angle_tolerance.value
    )
    ps = ana.get_refined_structure()
    out = StructureData(pymatgen=ps)
    out.label = structure.label + " REFINED"
    return out


@calcfunction
def get_conventional_standard_structure(structure, symprec, angle_tolerance):
    """Create conventional standard structure use pymatgen's interface"""
    from aiida.orm import StructureData
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    pstruct = structure.get_pymatgen()
    ana = SpacegroupAnalyzer(
        pstruct, symprec=symprec.value, angle_tolerance=angle_tolerance.value
    )
    ps = ana.get_conventional_standard_structure()
    out = StructureData(pymatgen=ps)
    out.label = structure.label + " CONVENTIONAL STANDARD"
    return out


@calcfunction
def make_supercell(structure, supercell, **kwargs):
    """Make supercell structure, keep the tags in order"""
    from ase.build.supercells import make_supercell as ase_supercell

    if "tags" in kwargs:
        tags = kwargs["tags"]
    else:
        tags = None

    atoms = structure.get_ase()
    atoms.set_tags(tags)

    slist = supercell.get_list()
    if isinstance(slist[0], int):
        satoms = atoms.repeat(slist)
    else:
        satoms = ase_supercell(atoms, np.array(slist))
    if "no_sort" not in kwargs:
        satoms = sort(satoms)

    if tags:
        stags = satoms.get_tags().tolist()
    satoms.set_tags(None)

    out = StructureData(ase=satoms)
    out.label = structure.label + " SUPER {} {} {}".format(*slist)

    if tags:
        return {"structure": out, "tags": orm.List(list=stags)}
    else:
        return {"structure": out}


@calcfunction
def niggli_reduce(structure):
    """Peroform niggli reduction using ase as the backend - this will rotate the structure into the standard setting"""
    from ase.build import niggli_reduce as niggli_reduce_

    atoms = structure.get_ase()
    niggli_reduce_(atoms)
    new_structure = StructureData(ase=atoms)
    new_structure.label = structure.label + " NIGGLI"
    return new_structure


@calcfunction
def niggli_reduce_spglib(structure):
    """Peroform niggli reduction using spglib as backend - this does not rotate the structure"""
    from spglib import niggli_reduce as niggli_reduce_spg

    atoms = structure.get_ase()
    reduced_cell = niggli_reduce_spg(atoms.cell)
    atoms.set_cell(reduced_cell)
    atoms.wrap()
    new_structure = StructureData(ase=atoms)
    new_structure.label = structure.label + " NIGGLI"
    return new_structure


@calcfunction
def neb_interpolate(init_structure, final_strucrture, nimages):
    """
    Interplate NEB frames using the starting and the final structures

    Get around the PBC warpping problem by calculating the MIC displacements
    from the initial to the final structure
    """

    ainit = init_structure.get_ase()
    afinal = final_strucrture.get_ase()
    disps = []

    # Find distances
    acombined = ainit.copy()
    acombined.extend(afinal)
    # Get piece-wise MIC distances
    for i in range(len(ainit)):
        dist = acombined.get_distance(i, i + len(ainit), vector=True, mic=True)
        disps.append(dist.tolist())
    disps = np.asarray(disps)
    ainit.wrap(eps=1e-1)
    afinal = ainit.copy()

    # Displace the atoms according to MIC distances
    afinal.positions += disps
    neb = NEB([ainit.copy() for i in range(int(nimages) + 1)] + [afinal.copy()])
    neb.interpolate()
    out_init = StructureData(ase=neb.images[0])
    out_init.label = init_structure.label + " INIT"
    out_final = StructureData(ase=neb.images[-1])
    out_final.label = init_structure.label + " FINAL"

    outputs = {"image_init": out_init}
    for i, out in enumerate(neb.images[1:-1]):
        outputs[f"image_{i + 1:02d}"] = StructureData(ase=out)
        outputs[f"image_{i + 1:02d}"].label = (
            init_structure.label + f" FRAME {i + 1:02d}"
        )
    outputs["image_final"] = out_final
    return outputs


@calcfunction
def fix_atom_order(reference: StructureData, to_fix: StructureData):
    """
    Fix atom order by finding NN distances bet ween two frames. This resolves
    the issue where two closely matching structures having diffferent atomic orders.
    Note that the two frames must be close enough for this to work
    """

    aref = reference.get_ase()
    afix = to_fix.get_ase()

    # Index of the reference atom in the second structure
    new_indices = np.zeros(len(aref), dtype=int)

    # Find distances
    acombined = aref.copy()
    acombined.extend(afix)
    # Get piece-wise MIC distances
    for i in range(len(aref)):
        dists = []
        for j in range(len(aref)):
            dist = acombined.get_distance(i, j + len(aref), mic=True)
            dists.append(dist)
        min_idx = np.argmin(dists)
        min_dist = min(dists)
        if min_dist > 0.5:
            print(
                f"Large displacement found - moving atom {j} to {i} - please check if this is correct!"
            )
        new_indices[i] = min_idx

    afixed = afix[new_indices]
    fixed_structure = StructureData(ase=afixed)
    fixed_structure.label = to_fix.label + " UPDATED ORDER"
    return fixed_structure


def match_atomic_order_(atoms: Atoms, atoms_ref: Atoms) -> Tuple[Atoms, List[int]]:
    """
    Reorder the atoms to that of the reference.

    Only works for identical or nearly identical structures that are ordered differently.
    Returns a new `Atoms` object with order similar to that of `atoms_ref` as well as the sorting indices.
    """

    # Find distances
    acombined = atoms_ref.copy()
    acombined.extend(atoms)
    new_index = []
    # Get piece-wise MIC distances
    jidx = list(range(len(atoms), len(atoms) * 2))
    for i in range(len(atoms)):
        dists = acombined.get_distances(i, jidx, mic=True)
        # Find the index of the atom with the smallest distance
        min_idx = np.where(dists == dists.min())[0][0]
        new_index.append(min_idx)
    assert len(set(new_index)) == len(atoms), "The detected mapping is not unique!"
    return atoms[new_index], new_index
