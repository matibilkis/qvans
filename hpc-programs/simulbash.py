from utilities.misc import dict_to_json
import os
import argparse
import numpy as np
#this file takes input from the submit.sh so we easily talk.
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--bonds", type=float, default=1.5)
parser.add_argument("--ratesiid", type=float, default=1.)
parser.add_argument("--nrun", type=float, default=1.)
parser.add_argument("--itvar", type=float, default=2.)
parser.add_argument("--optimizer", type=str, default="adam")
parser.add_argument("--problem", type=str, default="XXZ")
parser.add_argument("--init_layers", type=int, default=2)

args = parser.parse_args()
ratesiid=args.ratesiid
nrun=args.nrun

#problem_config = dict_to_json({"problem" : "H4", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bond)), ('H', (0., 0., 2*bond)), ('H', (0., 0., 3*bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"});q=8

#problem_config = dict_to_json({"problem" : "LiH", "geometry": [('Li', (0., 0., 0.)), ('H', (0., 0., bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"});q=12

############## CONDENSED MATTER ############
            ##### XXZ #######
#problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": args.itvar});q=12
            ##### tfim #######
if args.problem.upper() == "TFIM":
    problem_config = dict_to_json({"problem" : "TFIM", "g":1.0, "J": args.itvar});q=8
elif args.problem.upper() == "XXZ":
    problem_config = dict_to_json({"problem" : "XXZ", "g":1.0, "J": args.itvar});q=8
elif args.problem.upper() == "H4":
    bond=np.round(args.itvar,2)
    problem_config = dict_to_json({"problem" : "H4", "geometry": [('H', (0., 0., 0.)), ('H', (0., 0., bond)), ('H', (0., 0., 2*bond)), ('H', (0., 0., 3*bond))], "multiplicity":1, "charge":0, "basis":"sto-3g"});q=8
else:
    raise NameError("Che, no pusiste el --problem correctly")
### POSSIBLE PATHS
path="/data/uab-giq/scratch/matias/data-vans/"
# path = "../data-vans/"
st = "python3 main.py --path_results \"{}\" --qlr 0.01 --acceptance_percentage 1e-3 --n_qubits {} --reps 500 --qepochs 10000 --problem_config {} --optimizer adam --training_patience 1000 --rate_iids_per_step 3.0 --wait_to_get_back 25 --initialization hea --init_layers_hea 2".format(path,q,problem_config)

os.system(st)
