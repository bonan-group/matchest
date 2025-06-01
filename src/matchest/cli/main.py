from pathlib import Path

import click


@click.group("mc")
def mc():
    """Main command group for mc CLI."""


@mc.command()
@click.argument(
    "input_file", type=click.Path(exists=True, path_type=Path), default="POSCAR"
)
@click.option(
    "--input-format",
    type=str,
    help="Format for input file (needed if ASE can't guess from filename)",
)
@click.option(
    "-t",
    "--threshold",
    type=float,
    default=1e-5,
    help="Distance threshold in AA for symmetry reduction (corresponds to spglib 'symprec' keyword)",
)
@click.option(
    "-a",
    "--angle-tolerance",
    type=float,
    default=-1.0,
    help="Angle tolerance for symmetry reduction",
)
@click.option(
    "-o",
    "--output-file",
    type=click.Path(path_type=Path),
    help="Path/filename for output",
)
@click.option(
    "--output-format",
    type=str,
    help="Format for output file (needed if ASE can't guess from filename)",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Print output to screen even when writing to file.",
)
@click.option(
    "-p",
    "--precision",
    type=int,
    default=6,
    help="Number of decimal places for float display. (Output files are not affected)",
)
def prim(
    input_file,
    input_format,
    threshold,
    angle_tolerance,
    output_file,
    output_format,
    verbose,
    precision,
):
    """Perform convert to primitive cell"""
    from .structure import get_primitive

    get_primitive(
        input_file=input_file,
        input_format=input_format,
        output_file=output_file,
        output_format=output_format,
        threshold=threshold,
        angle_tolerance=angle_tolerance,
        verbose=verbose,
        precision=precision,
    )


@mc.command()
@click.argument(
    "filename", type=click.Path(exists=True, path_type=Path), required=False
)
@click.option("--filetype", type=str, help="File format for ASE importer")
def spg(filename, filetype):
    """Get space group information for different thresholds"""
    from .structure import get_spacegroup

    get_spacegroup(filename=filename, filetype=filetype)


@mc.command()
@click.argument(
    "filename", type=click.Path(exists=True, path_type=Path), default="POSCAR"
)
@click.option("--type", "file_type", type=str, help="Format of crystal structure file")
@click.option(
    "--min",
    "l_min",
    type=float,
    default=10,
    help="Minimum real-space cutoff / angstroms",
)
@click.option(
    "--max",
    "l_max",
    type=float,
    default=30,
    help="Maximum real-space cutoff / angstroms",
)
@click.option(
    "--comma-sep",
    "comma_sep",
    is_flag=True,
    help="Output as comma-separated list on one line",
)
@click.option(
    "--vasp",
    "vasp",
    is_flag=True,
    help="Provide VASP-like KSPACING instead of CASTEP MP spacing.",
)
@click.option(
    "--realspace",
    "realspace",
    is_flag=True,
    help="Use real space lattice length for computation.",
)
def kpoints(filename, file_type, l_min, l_max, comma_sep, vasp, realspace):
    """Calculate a systematic series of k-point samples."""
    from .kpoints import kpoints_main

    kpoints_main(filename, file_type, l_min, l_max, comma_sep, vasp, realspace)


@mc.command("vasp-conv")
@click.argument("filename", default="OUTCAR")
def vasp_conv(filename):
    """Convergence analysis for VASP"""

    from .geomconv import print_vasp_conv

    print_vasp_conv(filename)


@mc.command("vasp-max-force")
@click.argument("file", type=click.File(mode="r"))
def vasp_max_force(file):
    """Get the maximum forces for each VASP cycle"""
    from math import sqrt

    in_section = False
    cycles = 0
    atom_count = 0
    max_force_idx = 0
    max_force = 0
    for line in file:
        if "TOTAL-FORCE" in line:
            in_section = True
            cycles += 1
            atom_count = 0
            continue
        if "total drift" in line:
            in_section = False
            print(
                f"Loop {cycles:<5} Maximum force: {max_force:<5.3g} at atom {max_force_idx}"
            )
            max_force = 0
            max_force_idx = 0
            continue
        if "---" in line:
            continue
        if not in_section:
            continue
        # Now we are in section of the forces
        atom_count += 1
        tokens = line.split()
        force_mag = sqrt(sum([float(x) ** 2 for x in tokens[3:]]))
        if force_mag > max_force:
            max_force = force_mag
            max_force_idx = atom_count


@mc.command("view-files-ovito", help="Show all files with OVITO")
@click.argument("files", nargs=-1)
@click.option(
    "--sort/--no-sort", default=True, help="Sort the structures by energy per atoms"
)
def view_res_ovito(files, sort):
    """A wrapper command for viewing files as a trajectory in ovito"""
    from ase.io import read
    from ase.visualize import view

    if not files:
        return

    atoms_list = [read(str(r)) for r in files]
    if sort:
        engs = []
        for atoms in atoms_list:
            try:
                eng = atoms.get_potential_energy() / len(atoms)
            except RuntimeError:
                eng = 0.0
            engs.append(eng)
        atoms_list = [
            atoms for _, atoms in sorted(zip(engs, atoms_list), key=lambda x: x[0])
        ]
    view(atoms_list, viewer="ovito")


@mc.command("trim-vasprun")
@click.argument("input-file", type=click.Path(exists=True))
@click.argument("tag-to-remove", type=str)
def trim_tags(input_file, tag_to_remove):
    """
    Reduce the size of vasprun.xml file by removing that under a tag

    The two commonly used tags are 'projected' and 'partial'.
    This tool may DESTROY your data, use with caution!

    Limitations:

    - Only one tag can be removed in one pass

    """
    import os
    import shutil

    tag_start = f"<{tag_to_remove}>"
    tag_end = f"</{tag_to_remove}>"
    trimmed = False
    with open(input_file) as fhi:
        with open(input_file + ".tmp", "w") as fho:
            in_section = False
            for line in fhi:
                if tag_start in line:
                    in_section = True
                    trimmed = True
                    continue
                if tag_end in line:
                    in_section = False
                    continue
                if not in_section:
                    fho.write(line)

    if trimmed is True:
        if in_section is False:
            click.echo(f"Trimmed tag {tag_to_remove}")
            shutil.move(input_file + ".tmp", input_file)
        else:
            click.echo(f"Did not find the end of the tag {tag_to_remove} - aborting")
            os.remove(input_file + ".tmp")
    else:
        click.echo(f"Tag {tag_to_remove} is not found")
        os.remove(input_file + ".tmp")


@mc.command("castep-scf-info", help="Print number of SCF steps per iteration")
@click.argument("dot-castep", type=click.File("r"))
def castep_scf_info(dot_castep):
    from matchest.casteputils import count_scf_lines

    lengths = count_scf_lines(dot_castep)
    click.echo("\n".join(map(str, lengths)))


@mc.command("castep-timing")
@click.argument("dot-castep", type=click.File(mode="r"))
def castep_timing(dot_castep):
    """
    Analyse timing information based on a .castep file
    """
    from matchest.dotcastep import DotCastep

    dot = DotCastep(dot_castep)

    click.echo(f"Average time on each electronic loop : {dot.mean_loop:.3f} s")
    click.echo(f"Average time on each ionic loop      : {dot.mean_ionic_loop:.3f} s")

    pinfo = dot.parallel_info
    kp = pinfo["k-parallel"]
    gp = pinfo["g-parallel"]
    mpi = pinfo["procs"]
    click.echo(f"K-point parallelisation              : {kp}")
    click.echo(f"G-vector parallelisation             : {gp}")
    click.echo(f"Total number of processsors          : {mpi}")


@mc.command("pmg-convert-cell")
@click.option("--symprec", default=1e-5, help="Symmetry tolerance as passed to spglib")
@click.option("--angtol", default=0.1, help="Angular tolerance as passed to spglib")
@click.argument("input", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=False))
@click.option(
    "--cell-type",
    "-c",
    default="standard",
    type=click.Choice(["standard", "primitive", "primitive-standard"]),
)
@click.option(
    "--output-type", "-o", default="cif", type=click.Choice(["cif", "poscar"])
)
def cmd_convert_cell(input, output, cell_type, symprec, output_type, angtol):
    """
    Convert a crystal structure to primitive/standard cell using pymatgen's SpaceGroupAnalyser"""
    from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
    from pymatgen.core import Structure

    structure = Structure.from_file(input)
    ana = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angtol)

    if cell_type == "standard":
        out = ana.get_conventional_standard_structure()
    elif cell_type == "primitive":
        out = ana.find_primitive()
    elif cell_type == "primitive-standard":
        out = ana.get_primitive_standard_structure()
    # Write the output file
    out.to(output_type, output)


@mc.command("charge-neutral-combinations")
@click.option("--charges")
@click.option("--species")
@click.option("--nmax", default=10)
def cmd_charge_neutral(charges, species, nmax):
    """Compute charge neutral combinations that makes unique empirical formulas."""
    import re
    from itertools import product

    from pymatgen.core import Composition

    forms = set()
    charges = [float(x) for x in re.split(r"[ ;,|:]+", charges)]
    species = re.split(r"[ ;,|:]+", species)
    for seq in product(*([range(1, nmax)] * len(charges))):
        tot = sum([seq[i] * charges[i] for i in range(len(charges))])

        if tot == 0:
            forms.add(
                Composition(
                    "".join([f"{sym}{num}" for sym, num in zip(species, seq)])
                ).reduced_formula
            )
    for item in forms:
        click.echo(item)
