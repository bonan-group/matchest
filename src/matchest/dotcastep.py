"""
A parser for dot casteps
"""
from __future__ import division
from __future__ import print_function

import re
import numpy as np

pattern_geom = r"""
^\ *
(?P<name>[A-Z]+):          # Optimisation name
\ +finished\ iteration
\ +(?P<number>[0-9]+)             # iteration number
\ +with\ enthalpy=
\ +(?P<H>[+-.E0-9]+)        # Enthalpy
\ +(?P<unit>[a-zA-Z]+)          # Enthalpy unit
\ *$
"""

pattern_SCF = r"""
^\ +
([0-9]+)                      # Loop number
\ +
([+-.E0-9]+)                   # Energy
\ +
([+-.E0-9]+)                   # Fermi Energy
\ +
([+-.E0-9]+)                   # Energy gain per atom
\ +
([.0-9]+)                      # Timer
\ +
<--\ SCF
$
"""
pattern_SCF = r"^ +(\d+) +([0-9E+-.]+) +([0-9E+-.]+) +([0-9E+-.]+) +([0-9.]+) +<-- SCF"

geom_iter_start = re.compile(r' Starting \w+ iteration')
sfc_line = re.compile(pattern_SCF)
match_geom = re.compile(pattern_geom, re.VERBOSE)


class ScfStats:
    """Class representation of SCF loops leading to electronic convergence"""
    def __init__(self, data, tags):
        """
        data - a dictionary of the SCF information
        """
        self.data = data
        self.tags = tags

    @property
    def avg_time(self):
        return np.diff(self.data["timming"]).mean()

    @property
    def mean_loop(self):
        """The mean time of a electronic (SCF) loop"""
        return self.avg_time

    @property
    def ffree(self):
        """Final free energy"""
        return self.data["ffree"]

    @property
    def duration(self):
        t = self.data["timming"]
        return t[-1] - t[0]

    @property
    def start(self):
        return self.data["timming"][0]

    @property
    def finish(self):
        return self.data["timming"][-1]

    def __len__(self):
        return len(self.data["timming"])

    def __repr__(self):
        return "SCF<avg_time={:.2f}, length={:d}, duration={:.2f}>".format(self.avg_time, len(self), self.duration)


class DotCastep:
    """Class for a .castep file"""
    def __init__(self, fhandle):
        """
        Initialise an DotCastep instance

        Parameters
        ----------
        fhandle: handle-like object 
            A file handle for the CASTEP file
        """

        self.fhandle = fhandle
        self.geom_info = {}

    @property
    def scfs(self):
        """Return a list of SCF stats objects"""
        return list(self._scfs())

    @property
    def mean_loop(self):
        """
        Return the average LOOP times

        This does not include the time for computing forces/stress etc.
        """
        scfs = self.scfs
        nloops = sum(map(lambda x: len(x), scfs))
        total_time = sum(map(lambda x: x.duration, scfs))
        return total_time / nloops

    @property
    def mean_scf_steps(self):
        """The average number of electronic steps per ionic step"""
        scfs = self.scfs
        nloops = [len(scf) for scf in scfs]
        return np.mean(nloops)

    @property
    def mean_ionic_loop(self):
        """The average time spent on each ionic step"""
        scfs = self.scfs
        times = [scf.duration for scf in scfs]
        return np.mean(times)

    def _scfs(self):
        """A generator for SCF objects"""
        eng = []
        engf = []
        enga = []
        timming = []

        data = False
        self.fhandle.seek(0)
        for line in self.fhandle:
            m = sfc_line.match(line)
            if m is not None:
                data = True
                eng.append(float(m.group(2)))
                engf.append(float(m.group(3)))
                enga.append(float(m.group(4)))
                timming.append(float(m.group(5)))
            elif data is True and "Final free" in line:
                ffree = float(line.split()[-2])
                yield ScfStats(data=dict(eng=eng,
                                    engf=engf,
                                    enga=enga,
                                    timming=timming,
                                    final_free=ffree),
                          tags=None)
                eng = []
                engf = []
                enga = []
                timming = []
                data = False
                
    @property
    def parallel_info(self):
        """Acquire the parallelisation information"""
        self.fhandle.seek(0)
        nmpi = 1
        g_parallel = 1
        k_parallel = 1
        gpatt = r'G-vector\((\d+)-way\)'
        kpatt = r'k-point\((\d+)-way\)'
        for i, line in enumerate(self.fhandle):
            if "Calculation parallelised over" in line:
                nmpi = int(line.split()[-2])
                continue
            match = re.search(gpatt, line)
            if match:
                g_parallel = int(match.group(1))
                continue

            match = re.search(kpatt, line)
            if match:
                k_parallel = int(match.group(1))
                continue

        return {'procs': nmpi, 
                'k-parallel': k_parallel,
                'g-parallel': g_parallel,}
            

    def parse_geom_info(self):

        f = []
        smax = []
        de = []
        dr = []
        self.fhandle.seek(0)
        for line in self.fhandle:
            if "|F" in line:
                f.append(float(line.split()[3]))

            if "Smax" in line:
                try:
                    smax.append(float(line.split()[3]))
                except ValueError:
                    continue

            if "dE/ion" in line:
                de.append(float(line.split()[3]))

            if "|dR|max" in line:
                dr.append(float(line.split()[3]))

        self.ginfo = dict(F=f, Smax=smax, dE=de, dR=dr)
        return self.ginfo

    def parse(self, aggregate=False):
        """Parse the entire file"""

        # Storage space for capture properties
        eng = []  # Enthalpy
        iter_num = []  # Raw iteration array
        iter_times = []  # Raw timing of finished iteration
        save_times = []  # Raw timing of saves
        save_iter = []  # Iteration in which writing checkpoint took place

        geom_name, unit = None, None

        # In loop variables
        last_time = 0
        current_iter = 0

        # Iterate through the file
        self.fhandle.seek(0)
        for line in self.fhandle:
            # Capture the start of iteration
            if geom_iter_start.match(line):
                current_iter += 1
                continue

            # Capture timing of save and iteration numbers(current)
            if 'Writing model' in line:
                save_times.append(last_time)
                save_iter.append(current_iter)

            # Capture timing of SFC
            scf_match = sfc_line.match(line)
            if scf_match:
                timming = float(scf_match.group(5))
                last_time = timming
                continue

            # Capture the end of gemo iteration
            geom_match = match_geom.match(line)
            if geom_match:
                eng.append(float(geom_match.group('H')))
                iter_num.append(int(geom_match.group('number')))
                iter_times.append(float(last_time))
                geom_name = geom_match.group('name')
                unit = geom_match.group('unit')
                continue

        out = dict(H=eng,
                   iter_num=iter_num,
                   name=geom_name,
                   unit=unit,
                   iter_times=iter_times,
                   save_iter=save_iter,
                   save_times=save_times)
        if aggregate:
            # Need to aggregate the timer
            import numpy as np
            timer_array = np.array(iter_times)
            last_record = 0
            for i, time_record in enumerate(iter_times):
                if time_record < last_record:
                    timer_array[i:] = timer_array[i:] + last_record
                last_record = time_record
            out.update(time=timer_array)

        return out
