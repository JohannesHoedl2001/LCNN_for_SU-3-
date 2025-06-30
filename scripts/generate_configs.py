"""
    A script to generate lattice gauge field configurations based on the included MHMC Yang-Mills code.
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # path(s) to output file(s)
    parser.add_argument("--paths", type=str, nargs='+', required=True)

    # lattice and monte carlo options
    parser.add_argument("--nums", "-n", type=int, nargs='+', required=True)
    parser.add_argument("--sweeps", "-s", type=int, required=True)
    parser.add_argument("--reinit", "-ri", type=int, default=0)
    parser.add_argument("--warmup", "-w", type=int, required=True)
    parser.add_argument("--beta_min", "-bmin", type=float, required=True)
    parser.add_argument("--beta_max", "-bmax", type=float, required=True)
    parser.add_argument("--beta_steps", "-bs", type=int, required=True)
    parser.add_argument("--dims", "-d", type=int, nargs='+', required=True)
    parser.add_argument("--conjugate", "-c", action="store_true")
    parser.add_argument("--random_seed", "-r", type=int, required=True)

    # parse arguments
    args = parser.parse_args()

    # iterate through paths and sample numbers
    pbar0 = tqdm(total=len(args.paths), position=0)
    counter = 0
    for path, num in zip(args.paths, args.nums):
        # set progress description
        pbar0.set_description("Current file: {}".format(path))

        # run monte carlo simulation
        with h5py.File(path, "w") as f:
            # beta range
            betas = np.linspace(args.beta_min, args.beta_max, args.beta_steps)

            # create required attributes and datasets in the hdf5 file
            num_betas = len(betas)
            total_samples = num_betas * num
            D = len(args.dims)

            # determine number of plaquettes
            if args.conjugate:
                W = D * (D - 1)
            else:
                W = D * (D - 1) // 2

            # determine number of polyakov loops
            if args.conjugate:
                P = 2 * D
            else:
                P = D

            NC = 3

            u_shape = (np.prod(args.dims), D, NC, NC)
            w_shape = (np.prod(args.dims), W, NC, NC)
            p_shape = (np.prod(args.dims), P, NC, NC)

            f.create_dataset('u', (total_samples, *u_shape), dtype='complex64')
            f.create_dataset('w', (total_samples, *w_shape), dtype='complex64')
            f.create_dataset('p', (total_samples, *p_shape), dtype='complex64')

            f.create_dataset('beta', (total_samples, ), dtype='float32')
            f.create_dataset('dims', data=np.array(args.dims))

            pbar1 = tqdm(total=total_samples, position=1)
            for beta_index, beta in enumerate(betas):
                # set progress description
                pbar1.set_description("Beta = {:3.2f}".format(beta))
                counter += 1

                # choose random seed
                # add counter to random seed so that every created dataset is different
                random_seed = args.random_seed * (1 + counter)

                s = Simulation(args.dims, beta, random_seed)
                s.init(steps=5, use_flips=True)
                s.metropolis(args.warmup)

                for i in range(num):
                    # check if reinitialization is required
                    if args.reinit > 0:
                        if i % args.reinit == 0:
                            s.init(steps=5, use_flips=True)
                            s.metropolis(args.warmup)

                    # determine sample index
                    sample_index = beta_index * num + i

                    # perform monte carlo sweeps
                    s.metropolis(args.sweeps)

                    # save coupling constant
                    f['beta'][sample_index] = beta

                    # get links, wilson loops and polyakov loops
                    u = s.get_links()
                    f['u'][sample_index] = u

                    w = s.get_plaquettes(include_hermitian=args.conjugate)
                    f['w'][sample_index] = w

                    p = s.get_polyakov_loops(include_hermitian=args.conjugate)
                    f['p'][sample_index] = p

                    # advance progress bar
                    pbar1.update(1)

            # close progress bar
            pbar1.close()

            f.flush()
            f.close()

        # advance progress bar
        pbar0.update(1)

    # close progress bar
    pbar0.close()
