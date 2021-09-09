import pickle
from utilities.circuit_basics import Basic
import numpy as np
import os
from datetime import datetime

class Evaluator(Basic):
    def __init__(self, args, info=None, loading=False, acceptance_percentage = 0.01, reduce_acceptance_percentage=True,accuracy_to_end=-np.inf,
                nrun_load=0, path="/data/uab-giq/scratch/matias/data-vans/"):
        """

        This class serves as evaluating the energy, admiting the new circuit or not. Also stores the results either if there's a relevant modification or not.
        Finally, it allows for the possibilty of loading previous results, an example for the TFIM is:


        if_finish_ok: bool, if accuracy to end is reached, then VANS stops
        accuraccy to end: ground_state_energy (exact diagonalization or FCI) (+ chemical_accuracy in case of chemical hamiltonians). TODO: add some relative error for condensed matter hamiltonians (not super important though.)


        args: is a dictionary version of the parser.args.
        Example:
        args={"n_qubits":8,"problem_config":{"problem" : "XXZ", "g":1.0, "J": j}, "specific_name":"XXZ/8Q - J {} g 1.0".format(j)}

        Note that if specific name is used, it will load the results from the path folder, but will use the format "specific_name" instead of the pre-defined.

        Also, if a run requires more details (say, different values for the depolarizing channel), results can be labeled using args["specific_name"].

        (cheat-sheet for jupyter notebook):
            %load_ext autoreload
            %autoreload 2
            from utilities.circuit_basics import Evaluator
            evaluator = Evaluator(loading=True, args={"n_qubits":3, "J":4.5})
            unitary, energy, indices, resolver = evaluator.raw_history[47]
            {"channel":"depolarizing", "channel_params":[p], "q_batch_size":10**3}
        """
        super(Evaluator, self).__init__(n_qubits=args["n_qubits"])
        self.path = path
        self.reduce_acceptance_percentage=reduce_acceptance_percentage

        if not loading:
            self.raw_history = {}
            self.evolution = {}
            self.displaying={"information":"\n Hola, I'm VANS, and current local time is {} \n".format(datetime.now())}
            self.lowest_energy = None
            self.if_finish_ok = False
            self.accuracy_to_end = accuracy_to_end
            self.its_without_improvig=0
            if "specific_name" not in list(args.keys()):
                args["specific_name"] = ""

            problem_identifier = self.get_problem_identifier(args["problem_config"])
            self.identifier = "{}/N{}_{}_{}".format(args["problem_config"]["problem"],args["n_qubits"],problem_identifier)+args["specific_name"]

            self.directory = self.create_folder(info)
            self.acceptance_percentage = acceptance_percentage

        else:
            args_load={}
            for str,default in zip(["n_qubits", "problem_config","specific_name","load_displaying"], [4, {"problem":"TFIM", "g":1.0, "J": 0.0}, None,False]):
                if str not in list(args.keys()):
                    args_load[str] = default
                else:
                    args_load[str] = args[str]
            problem_identifier = self.get_problem_identifier(args_load["problem_config"])

            if "specific_folder_name" not in list(args.keys()):
                if args_load["specific_name"] is None:
                    ap=""
                else:
                    ap = args_load["specific_name"]
                self.identifier = "{}/N{}_{}_".format(args["problem_config"]["problem"],args_load["n_qubits"],problem_identifier)+ap
            else:
                self.identifier = "{}/{}".format(args["problem_config"]["problem"], args["specific_folder_name"])
            self.load(args_load,nrun=nrun_load, load_displaying = args_load["load_displaying"]) #this is because i changed this abit...

                #if args_load["specific_name"] is None:
                           # else:
               #     self.load_from_name(args_load["specific_name"], nrun=nrun_load)

    def get_problem_identifier(self, args):
        #### read possible hamiltonians to get id structure
        with open("utilities/hamiltonians/cm_hamiltonians.txt") as f:
            hams = f.readlines()
        cm_hamiltonians = [x.strip().upper() for x in hams]

        with open("utilities/hamiltonians/chemical_hamiltonians.txt") as f:
            hams = f.readlines()
        chemical_hamiltonians=([x.strip().upper() for x in hams])

        if args["problem"].upper() in cm_hamiltonians:
            id = "g{}J{}".format(args["g"],args["J"])
        elif args["problem"].upper() in chemical_hamiltonians:
            id = "geometry_{}_multip_{}_charge_{}_basis{}".format(args["geometry"], args["multiplicity"], args["charge"], args["basis"])
        elif args["problem"].upper() == "AUTOENCODERH2":
            id = "AUTOENCODERH2_{}".format(args["n_qubits"])
            # raise NameError("Check that your args.problem_config is correct")
        else:
            id="BAD_LABEL_{}".format(args)
        return id


    def create_folder(self, info):
        """
        self.path is data-vans
        """
        name_folder=self.path+self.identifier
        if not os.path.exists(name_folder):
            os.makedirs(name_folder)
            nrun=0
            final_folder = name_folder+"/run_"+str(nrun)
            with open(name_folder+"/runs.txt", "w+") as f:
                f.write(info)
                f.close()
            os.makedirs(final_folder)
        else:
            folder = os.walk(name_folder)
            nrun=0
            for k in list(folder)[0][1]:
                if k[0]!=".":
                    nrun+=1
            final_folder = name_folder+"/run_"+str(nrun)
            with open(name_folder+"/runs.txt", "r") as f:
                a = f.readlines()
                f.close()
            with open(name_folder+"/runs.txt", "w") as f:
                f.write(str(nrun)+"\n")
                f.write(info)
                f.write("\n")
                f.close()
            os.makedirs(final_folder)
        return final_folder

    def load(self,args, nrun=0, load_displaying=False):
        name_folder = self.path+self.identifier+"/run_{}".format(nrun)
        self.load_dicts_and_displaying(name_folder, load_displaying=load_displaying)
        return

    def load_from_name(self,name, nrun=0):
        name_folder = self.path+name+"/run_{}".format(nrun)
        self.load_dicts_and_displaying(name_folder)
        return

    def save_dicts_and_displaying(self):
        output = open(self.directory+"/raw_history.pkl", "wb")
        pickle.dump(self.raw_history, output)
        output.close()
        output = open(self.directory+"/evolution.pkl", "wb")
        pickle.dump(self.evolution, output)
        output.close()
        output = open(self.directory+"/displaying.pkl", "wb")
        pickle.dump(self.displaying, output)
        output.close()
        #with open(self.directory+"/evolution.txt","wb") as f:
        #    f.write(self.displaying)
        #    f.close()
        np.save(self.directory+"/accuracy_to_end",np.array([self.accuracy_to_end]))
        return

    def load_dicts_and_displaying(self,folder, load_displaying=False):
        with open(folder+"/raw_history.pkl" ,"rb") as h:
            self.raw_history = pickle.load(h)
        with open(folder+"/evolution.pkl", "rb") as hh:
            self.evolution = pickle.load(hh)
        if load_displaying is True:
            with open(folder+"/displaying.pkl", "rb") as hhh:
                self.displaying = pickle.load(hhh)
        # if load_txt is True:
        #     with open(folder+"/evolution.txt", "r") as f:
        #        a = f.readlines()
        #        f.close()
        #     self.displaying = a
        return

    def accept_energy(self, E):
        """
        E: energy after some optimization (to be accepted or not).
        For the moment we leave the same criteria for the noisy scenario also.
        """
        if self.lowest_energy is None:
            return True
        else:
            return (E-self.lowest_energy)/np.abs(self.lowest_energy) < self.acceptance_percentage

    def get_best_iteration(self):
        """
        returns minimum in evolution.
        """
        bests = list(np.where(np.array(list(self.evolution.values()))[:,1] == np.min(np.array(list(self.evolution.values()))[:,4]))[0])
        if len([np.squeeze(bests)])==0:
            return int(bests)
        else:
            return int(np.squeeze(bests[0])) #say... actually would be nice to keep the most shallow one

    def number_cnots_best(self):
        cn=0
        for k in self.evolution[self.get_best_iteration()][2]: #get the indices
            if k < self.number_of_cnots:
                cn+=1
        return cn

    def number_of_parameters_best(self):
        indices = self.evolution[self.get_best_iteration()][2]
        return len(indices)-self.number_cnots_best()

    def decrease_acceptance_range(self):
        """
        This function is to guarantee that you don't move so far away
        TODO: smarter scheduling, some meta-learning?
        """
        if self.decrease_acceptance_range == True:
            self.acceptance_percentage*=0.9
        return

    def add_step(self,indices, resolver, energy, relevant=True):
        """
        indices: list of integers describing circuit to save
        resolver: dictionary with the corresponding circuit's symbols
        energy: expected value of target hamiltonian on prepared circuit.
        relevant: if energy was minimized on that step
        """
        if self.lowest_energy is None:
            self.lowest_energy = energy
            self.its_without_improvig = 0

        elif energy < self.lowest_energy:
            self.lowest_energy = energy
            self.its_without_improvig = 0
            self.decrease_acceptance_range()

        else:
            self.its_without_improvig+=1

        if self.lowest_energy <= self.accuracy_to_end:
            self.if_finish_ok = True

        self.raw_history[len(list(self.raw_history.keys()))] = [self.give_unitary(indices, resolver), energy, indices, resolver, self.lowest_energy, self.accuracy_to_end]
        if relevant == True:
            self.evolution[len(list(self.evolution.keys()))] = [self.give_unitary(indices, resolver), energy, indices,resolver, self.lowest_energy, self.accuracy_to_end]
        self.save_dicts_and_displaying()
        return
