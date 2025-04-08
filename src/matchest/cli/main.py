import click
from pathlib import Path
@click.group('mc')
def mc():
    """Main command group for mc CLI."""
    pass


@mc.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path), default='POSCAR')
@click.option('--input-format', type=str, help='Format for input file (needed if ASE can\'t guess from filename)')
@click.option('-t', '--threshold', type=float, default=1e-5, help='Distance threshold in AA for symmetry reduction (corresponds to spglib \'symprec\' keyword)')
@click.option('-a', '--angle-tolerance', type=float, default=-1.0, help='Angle tolerance for symmetry reduction')
@click.option('-o', '--output-file', type=click.Path(path_type=Path), help='Path/filename for output')
@click.option('--output-format', type=str, help='Format for output file (needed if ASE can\'t guess from filename)')
@click.option('-v', '--verbose', is_flag=True, help='Print output to screen even when writing to file.')
@click.option('-p', '--precision', type=int, default=6, help='Number of decimal places for float display. (Output files are not affected)')
def prim(input_file, input_format, threshold, angle_tolerance, output_file, output_format, verbose, precision):
    """Perform convert to primitive cell"""
    from .structure import  get_primitive
    get_primitive(
        input_file=input_file,
        input_format=input_format,
        output_file=output_file,
        output_format=output_format,
        threshold=threshold,
        angle_tolerance=angle_tolerance,
        verbose=verbose,
        precision=precision
    )


@mc.command()
@click.argument('filename', type=click.Path(exists=True, path_type=Path), required=False)
@click.option('--filetype', type=str, help='File format for ASE importer')
def spg(filename, filetype):
    """Get space group information for different thresholds"""
    from .structure import  get_spacegroup
    get_spacegroup(filename=filename, filetype=filetype)

    

@mc.command()
@click.argument('filename', type=click.Path(exists=True, path_type=Path), default='POSCAR')
@click.option('--type', 'file_type', type=str, help='Format of crystal structure file')
@click.option('--min', 'l_min', type=float, default=10, help='Minimum real-space cutoff / angstroms')
@click.option('--max', 'l_max', type=float, default=30, help='Maximum real-space cutoff / angstroms')
@click.option('--comma-sep', 'comma_sep', is_flag=True, help='Output as comma-separated list on one line')
@click.option('--vasp', 'vasp', is_flag=True, help='Provide VASP-like KSPACING instead of CASTEP MP spacing.')
@click.option('--realspace', 'realspace', is_flag=True, help='Use real space lattice length for computation.')
def kpoints(filename, file_type, l_min, l_max, comma_sep, vasp, realspace):
    """Calculate a systematic series of k-point samples."""
    from .kpoints import kpoints_main
    kpoints_main(filename, file_type, l_min, l_max, comma_sep, vasp, realspace)