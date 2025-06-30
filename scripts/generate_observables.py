"""
    A script for loading lattice gauge field configurations and computing observables to be used as labels in ML tasks.
"""

import os
os.environ["MY_NUMBA_TARGET"] = "cuda"
os.environ["GAUGE_GROUP"] = "su3"
os.environ["PRECISION"] = "double"

import sys
sys.path.append('..')

import argparse
import numpy as np
import h5py
from tqdm import tqdm

from lge_cnn.ym.core import Simulation


def create_dataset(file, name, shape, dtype):
    if name not in file.keys():
        file.create_dataset(name, shape, dtype=dtype)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path(s) to pre-computed configurations
    parser.add_argument("--paths", type=str, nargs='+', required=True)

    # optional labels and observables
    parser.add_argument("--loops", "-l", type=str, nargs='+')           # 1x1, 1x2, ..., 4x4
    parser.add_argument("--loop_axes", "-la", type=int, nargs='+')      # axes to use for all loops
    parser.add_argument("--polyakov", "-p", action="store_true")        # temporal polyakov loop
    parser.add_argument("--charge_plaq", "-qp", action="store_true")    # topological charge (plaquette version)
    parser.add_argument("--charge_clov", "-qc", action="store_true")    # topological charge (clover version)

    # parse arguments
    args = parser.parse_args()

    # quick sanity checks
    if len(args.loop_axes) != 2:
        raise ValueError('Need exactly two axes for Wilson loop calculations.')

    # iterate through paths and sample numbers
    pbar0 = tqdm(total=len(args.paths), position=0)
    for path in args.paths:
        # set progress description
        pbar0.set_description("Current file: {}".format(path))

        # open configuration file
        with h5py.File(path, 'a') as f:
            # get meta data
            dims = np.array(f['dims'])
            num_samples = f['u'].shape[0]

            # create new datasets if necessary
            if args.loops is not None:
                for size in args.loops:
                    create_dataset(f, 'trW_{}'.format(size), (num_samples, np.prod(dims)), 'complex64')

            if args.polyakov:
                create_dataset(f, 'trP', (num_samples, np.prod(dims) / dims[0]), dtype='float32')

            if args.charge_plaq:
                create_dataset(f, 'QP', (num_samples, np.prod(dims)), dtype='float32')

            if args.charge_clov:
                create_dataset(f, 'QC', (num_samples, np.prod(dims)), dtype='float32')

            # initialize simulation
            s = Simulation(dims, 1.0, seed=0)

            pbar1 = tqdm(total=num_samples, position=1)
            for sample_index in range(num_samples):
                s.load_config(f['u'][sample_index])

                # compute required observables
                if args.loops is not None:
                    # compute traces of Wilson loops (as complex numbers)
                    mu, nu = args.loop_axes
                    for size in args.loops:
                        n_mu, n_nu = [int(x) for x in size.split('x')]
                        trW = s.wilson_large(mu, nu, n_mu, n_nu)
                        f['trW_{}'.format(size)][sample_index] = trW

                if args.polyakov:
                    # compute (real) traces of temporal polyakov loops from pre-computed loops
                    p = f['p'][sample_index, :, 0]
                    trP = np.trace(p, axis1=-2, axis2=-1).real
                    # select only one time slice
                    trP = trP[:np.prod(dims) // dims[0]]
                    f['trP'][sample_index] = trP

                if args.charge_plaq:
                    qp = s.topological_charge(mode=0)
                    f['QP'][sample_index] = qp

                if args.charge_clov:
                    qc = s.topological_charge(mode=1)
                    f['QC'][sample_index] = qc

                # advance progress bar
                pbar1.update(1)

            f.flush()
            f.close()

            # close progress bar
            pbar1.close()

        # advance progress bar
        pbar0.update(1)

    # close progress bar
    pbar0.close()

