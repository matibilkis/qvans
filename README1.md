# VANS

We present VANS, an algorithm that uses a variable ansatz structure to do VQE. In short, identity-resolution block of gates are proposed to be appended in the circuit, then VQE is run on the proposed circuit and if the obtained energy is lowered, the modification is accepted.

#### important changes in variational..., there are also some tips in case things do not work ####

1. deleted the give_energy method in variational.
2. killed the nasty lower_bound_Eg
3. killed the max_time_training depeding on noise/qubits. Set to 5 mins now.
4. changed the adam optimizer for the SGD with algoritm 4 of qacq (actually implemented options there)
5. changed the strength of adding the random_perturbations; now it's almost vanishing (THIS IS IMPORTANT, DISCUSS IT WITH LUKASZ). In case we want to calim the BP mitigation...

6. changed the min_delta to 1e-6 (this can be changed as well..) THIS IS ALSO SOMETHING SUBTLE
7. changed the give_energy thing as well. In vqe_handler there's no more give_energy, I retrieve the energy from the last value of the training history (we could actually take the best one instead, but I wouldn't know how to fetch the trainable variables, maybe with a callback).

8. killed train_model function

9. List of hamiltonians.

### About hamiltonians.

The idea is to have some control when developing the code, so I put the hamiltonians that have been working in a .txt (at utilities/hamiltonians/*.txt)


#### Installation

In your favorite directory, clone the github branch by typying git clone https://github.com/matibilkis/vans.git --depth=1 --branch numerics

You will need to create a virtual environment. To do that, go to the vans directory (the one we have created by the "clone" command in the last item) and type virtualenv qvans

We will need to install the libraries. For this, we first activate the virtual environment. To do that, type . qvans/bin/activate  and then install dependencies by typying pip3 install -r requirements.txt

Once things got installed, we should be able to run VANS! In case we have access to the HPC, we can send many jobs by typying  bash <b>submit.sh</b> Otherwise we have two options: use multiple cores (we can do that by running meta_main.py) or doing a single configuration by <b>main.py</b>. One may play around with hyperparameters and Pool options according to computing resources.

#### Algorithm description

[To be done. In the meantime, the file <i>main.py</i> is a good overview, in particular the main loop.]

#### Code structure

[To be done, this description is considered necessary since the code is structure by calling different mododules (which are quite documented, see utilities folder). For instance, if in the future the VQE module is willing to be modified, enhance (for example replace adam with rosalin), this description will help a lot.]

## Noiseless circuit results

In the following we consider 4 qubit circuits and showcase VANS on two different scenarios: Transverse Field Ising Model (TFIM) and XXZ model. The initial circuit parametrization is a <i>product ansatz</i>, consisting on rotations around x-axis at each qubit.

Xxz model was introduced since this [meta-VQE paper](https://arxiv.org/abs/2009.13545) was out recently. Their task is a bit more general than just minimize find the ground state: they try to <i>learn</i> ground state energy profiles as hamiltonian hyperparameters change (like predicting how the circuit that prepares ground state in between of unseen points is). Besides the paper (which works with 14 qubits), they show off their method in a [certain tutorial](https://github.com/aspuru-guzik-group/Meta-VQE/blob/master/Meta-VQE.ipynb) on a 4-qubit xxz model. I was curious to see how our algorithm compares with the plot they report, and it seems to be better ours (no idea of the details though, maybe that plot is not the best meta-VQE can do; still they present it). ¿Maybe we can try xxz with even more qubits (14 as a goal) if we consider this relevant?


#### VANS vs. true ground energy
In the following we present the results obtained after running VANS for 100 (50) iterations, for TFIM (XXZ) model. We despict the lowest energy found for a sweep over different hamiltonian parameters. For each configuration, the evolution of the ansatz can be found in the github directory <i> results/model/4Q-configuration/evolution.txt </i>.

<img src="results/xxz/display_results/xxz_4q_20_10.png" alt="xxz4" width="600"/>
<img src="results/TFIM/tfim4.png" alt="tfim4" width="600"/>

#### "Learning" curves
we also depict (i) the lowest energy found, and (ii) the energy of the current circuit accepted by VANS, as evolved during algorithm iterations.

Due to the acceptance of a higher energy circuit (up to some threshold, set to be %1 of relative difference w.r.t lowest energy found so far), in some cases energy in (ii) goes up. We also observe that, for some other cases, the initial ansatz appears to be sufficiently good. Nonetheless notice that this <i> sufficiently good </i> will depend -among other things- on the learning rate set used to perform VQE, which in this example was set to 0.005 (this parameter choice can be found in the run info.txt file, placed inside each run directory, under the name qlr.)

<img src="results/TFIM/evolution_energy_TFIM.png" alt="evol_xxz4" width="600"/>
<img src="results/xxz/display_results/plotting_history_energies.png" alt="tfim4" width="600"/>

## Noisy circuits

<i>Note:</i> we are currently not estimating the cost function, from some finite number of shots, but rather using the expected value (infinite shots). A finite number of shots can be easily included, but we have not done it yet.

We implement quantum channels that can be decomposed as a sum of unitary transformations; for this we take a batch of circuits, each affected by a possible unitary transformation with some probability. This permits the usage of the fast C++ TFQ simulator, since Density Matrix Simulator is not implemented (yet) - check the open issue [here](https://github.com/tensorflow/quantum/issues/250).

Quite arbitrarily from our side, the channel acts before each gate that appears in the circuit (read from left to right). In case of CNOT gates, the channel is applied at both control and target. We illustrate this in a simple 4-qubit circuit, made up of rotations around the x-axis: we take the symmetric depolarizing channel that with probability <i>1-p</i> acts as the identity, whereas with probability <i>p</i> it applies a Pauli gate (X, Y, Z), uniform-randomly chosen.

<img src="results/optimized_product_ansatz_noisy/noise_model.png" alt="noise model" width="600"/>

### Some trivial checks
On the way, we have also checked that this kind of procedure approximates well the Density Matrix Simulator of cirq; this can be found [here](results/optimized_product_ansatz_noisy/noise_VANS_and_TFQ.ipynb). Furthermore, we checked how the results of doing VQE varies on this simple ansatz, for TFIM with J=0, if the noise strength is increased. Find that notebook [here](results/optimized_product_ansatz_noisy/vqe_depolarizing_range_product_ansatz.ipynb).

<img src="results/optimized_product_ansatz_noisy/ges_energy.png" alt="noise model" width="400"/>

### Some results
Approximating noisy channels in the context described above and running VANS on top of this is feasible, but sligthly expensive. On a rather powerful laptop, but without GPU and depending on the particular circuit, each VQE optimization takes more than 30 minutes and hence that VANS-iteration step is considered skipped). On the way, we note that VANS can now detect whether a GPU is available to be used, and if so, uses it (we did some little experiments using google colab's GPU, noticing a considerable increment on the speed that VQE is done, but colab is not handy to run an entire simulation).

Nonetheless, we did a run of 50-iterations-VANS. We considered 3 qubits and TFIM with a fixed value of g and J, for three different values of depolarizing channel strength. We observe a nice reduction of circuit's number of CNOTS for only one of the cases, since the remaining two got stuck. Find the circuits generating the results [here](https://github.com/matibilkis/vans/blob/implicit_noise/noisy_TFIM_3qubits):

![depo](results/noisy_TFIM_3qubits/depolarizing_tfim_3qubits.png)

### What to do next
We can either try different channels (that admit this convex sum of unitary transformations: it'd be straightforward to implement this in the current version of the code). We can also think on more qubits, although memory requirements are already quite high for my laptop (to generate the results on depolarizing channel on 3 qubits using currently 13 out of 16 GB available, which delayed other projects I should not delay).

It would be desirable to run more simulations if there is enough interest.

## Some ideas for the future
<ul>
<li> In case that it is desired to limit the circuit's depth, a simple way out is to just limit it: do not allow new identity resolutions that would make the circuit longer than a certain threshold. </li>

<li> In case that we want to scale the algorithm up (in terms of number of qubits), we can think on increasing the number of identity resolutions added per VANS step (code-wise this is super easy to do)</li>

<li> If many cores are available, it sounds reasonable to parallelize the VQE in the batched approximation of noisy channels. I do remember Arthur Pesah mentioning that this greatly improves the speed of TFQ on VQE. </li>

<li> Although the whole idea of VANS is quite simple, it serves as a nice way to generate datasets of approximate-ground-state-preparing circuits of the corresponding hamiltonians. It would be cool if there's something to tell about these circuits, which opens the door to play with machine learning tools. For instance one try to see if there's some correlation between the discovered circuits and the hamiltonian parameters, or just if some pattern can be found. (But this I'd say is beyond the scope for now, since it may take some time and maybe there's nothing to tell).</li>

<li>Explore how could we use a real device to do this (preferable from Google since all code is written in Cirq)</li>
</ul>

## Things that are patched and better solutions are welcome

Rule 5 of utilities.simplifier, we use sympy.solve to reduce many consecutive 1-qubit unitary gates to Rz Rx Rz, but this solution is not very elegant. Also in case we want to implement a 3-CNOT identity resolution, this will be likely to crash. Nonetheless I could not find any promissing method to solve the non-linear system of equations with python (or even other stuff, also tried with Mathematica).
