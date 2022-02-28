import pickle
from utilities.circuit_database import CirqTranslater
import numpy as np
import os
from datetime import datetime
from utilities.misc import get_def_path

class Evaluator(CirqTranslater):
    def __init__(self, args,
                acceptance_percentage = 0.01,
                acceptance_reduction_rate = 10,
                lowest_acceptance_percentage = 1e-4,
                lower_bound_cost=-np.inf,
                increase_acceptance_percentage_after_its = 5,
                nrun=0):
        """
        This class evaluates the cost at each iteration, and decides whether to accept the new circuit or not.

        It also stores the results either if there's a relevant modification or not.

        Finally, it allows for the possibilty of loading previous results.

        *** args = {"problem_name":str, params":list}
        *** acceptance_percentage: up to which value an increase in relative energy is accepted or not
        *** path:
            get_def_path() or not
        """
        super(Evaluator, self).__init__(n_qubits=args["n_qubits"])

        self.raw_history = {}
        self.evolution = {}
        self.displaying={"information":"\n VAns started at {} \n".format(datetime.now())}

        self.lowest_cost = None
        self.end_vans = False
        self.lower_bound = lower_bound_cost
        self.its_without_improving=0

        args["params"] = np.round(args["params"],2)
        self.args = args
        self.identifier =  get_def_path() + "{}/{}/{}/".format(args["problem"],args["params"], nrun)

        os.makedirs(self.identifier, exist_ok=True)

        self.initial_acceptance_percentage = acceptance_percentage
        self.acceptance_percentage = acceptance_percentage
        self.acceptance_reduction_rate = acceptance_reduction_rate
        self.lowest_acceptance_percentage = lowest_acceptance_percentage
        self.increase_acceptance_percentage_after_its = increase_acceptance_percentage_after_its

    def save_dicts_and_displaying(self):
        output = open(self.identifier+"/raw_history.pkl", "wb")
        pickle.dump(self.raw_history, output)
        output.close()
        output = open(self.identifier+"/evolution.pkl", "wb")
        pickle.dump(self.evolution, output)
        output.close()
        output = open(self.identifier+"/displaying.pkl", "wb")
        pickle.dump(self.displaying, output)
        output.close()
        return

    def load_dicts_and_displaying(self, folder, load_displaying=False):
        with open(folder+"raw_history.pkl" ,"rb") as h:
            self.raw_history = pickle.load(h)
        with open(folder+"evolution.pkl", "rb") as hh:
            self.evolution = pickle.load(hh)
        if load_displaying is True:
            with open(folder+"displaying.pkl", "rb") as hhh:
                self.displaying = pickle.load(hhh)
        return

    def accept_cost(self, C, decrease_only=True):
        """
        C: cost after some optimization (to be accepted or not).
        """
        if self.lowest_cost is None: ###accept initial modification
            return True
        else:
            return (C-self.lowest_cost)/np.abs(self.lowest_cost) < self.acceptance_percentage

    def decrease_acceptance_range(self):
        self.acceptance_percentage = max(self.lowest_acceptance_percentage, self.acceptance_percentage/self.acceptance_reduction_rate)
        return

    def increase_acceptance_range(self):
        self.acceptance_percentage = min(self.initial_acceptance_percentage, self.acceptance_percentage*self.acceptance_reduction_rate)
        return

    def add_step(self, database, cost,relevant=True):
        """
        database: pandas db encoding circuit
        cost: cost at current iteration
        relevant: if cost was decreased on that step as compared to previous one(s)
        """
        if self.lowest_cost is None:
            self.lowest_cost = cost
            self.its_without_improving = 0

        elif cost < self.lowest_cost:
            self.lowest_cost = cost
            self.its_without_improving = 0
            self.decrease_acceptance_range()

        else:
            self.its_without_improving+=1
            if self.its_without_improving > self.increase_acceptance_percentage_after_its:
                self.increase_acceptance_range()

        if self.lowest_cost <= self.lower_bound:
            self.end_vans = True
        self.raw_history[len(list(self.raw_history.keys()))] = [database, cost, self.lowest_cost, self.lower_bound, self.acceptance_percentage]
        if relevant == True:
            self.evolution[len(list(self.evolution.keys()))] = [database, cost, self.lowest_cost, self.lower_bound, self.acceptance_percentage]
        self.save_dicts_and_displaying()
        return
