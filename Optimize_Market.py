import numpy as np
import gurobipy as gb
import pandas as pd
import defaults
from itertools import izip
from collections import defaultdict

####
#  Class to do the nodal day-ahead dispatch.
#  Init: Load network, load initial data, build model.
#  Optimize: Optimize the model.
#  Load_new_data: Takes new blocks of wind, solar and load data as input.
#                   Inserts them into the model.
####


# Class which can have attributes set.
class expando(object):
    pass


# Optimization class
class Optimize_Market:
    '''
        initial_(wind,load,solar) are (N,t) arrays where
        N is the number of nodes in the network, and
        t is the number of timesteps to optimize over.
        Note, that t is fixed by this inital assignment
    '''
    def __init__(self, wind_scenarios, load_signal):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self.results = expando()
        self.data = expando()
        self._load_data(wind_scenarios, load_signal)
        self._build_model()

    def optimize(self):
        self.model.optimize()

    def load_new_data(self, wind_scenarios, load_signal):
        self._add_new_data(wind_scenarios, load_signal)
        self._update_constraints()

    ###
    #   Loading functions
    ###

    def _load_data(self, wind_scenarios, load_signal):
        self._load_network()
        self._load_generator_data()
        self._load_intial_data(wind_scenarios, load_signal)

    def _load_network(self):
        self.data.nodedf = pd.read_csv(defaults.nodefile).set_index('ID')
        self.data.linedf = pd.read_csv(defaults.linefile).set_index(['fromNode', 'toNode'])
        # # Node and edge ordering
        self.data.nodeorder = self.data.nodedf.index.tolist()
        self.data.lineorder = [tuple(x) for x in self.data.linedf.index]
        # # Line limits
        self.data.linelimit = self.data.linedf.limit.to_dict()

        def zero_to_inf(x):
            if x > 0.0001:
                return x
            else:
                return gb.GRB.INFINITY
        
        self.data.linelimit = {k: zero_to_inf(v) for k, v in self.data.linelimit.iteritems()}
        self.data.lineadmittance = self.data.linedf.Y.to_dict()

        self.data.nodetooutlines = defaultdict(list)
        self.data.nodetoinlines = defaultdict(list)
        for l in self.data.lineorder:
            self.data.nodetooutlines[l[0]].append(l)
            self.data.nodetoinlines[l[1]].append(l)
        self.data.slackbuses = [self.data.nodeorder[0]]

    def _load_generator_data(self):
        self.data.generatorinfo = pd.read_csv(defaults.generatorfile, index_col=0)
        self.data.generators = self.data.generatorinfo.index.tolist()
        self.data.generatorsfornode = defaultdict(list)
        origodict = self.data.generatorinfo['origin']
        for gen, n in origodict.iteritems():
            self.data.generatorsfornode[n].append(gen)

    def _load_intial_data(self, wind_scenarios, load_signal):
        self.data.scenarios = wind_scenarios.items.tolist()
        self.data.taus = wind_scenarios.major_axis.tolist()
        self.data.wind_scenarios = wind_scenarios
        self.data.load = load_signal
        self.data.scenarioprob = {s: 1.0/len(self.data.scenarios) for s in self.data.scenarios}

    ###
    #   Model Building
    ###
    def _build_model(self):
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
        self.model.update()

    def _build_variables(self):
        taus = self.data.taus
        scenarios = self.data.scenarios
        generators = self.data.generators
        gendata = self.data.generatorinfo.T.to_dict()
        nodes = self.data.nodeorder
        lines = self.data.lineorder
        wind = self.data.wind_scenarios
        load = self.data.load.to_dict()

        m = self.model

        # Production of generator g at time t
        gprod = {}
        for t in taus:
            for g in generators:
                for s in scenarios:
                    gprod[s, g, t] = m.addVar(lb=0.0, ub=gendata[g]['capacity'])
        self.variables.gprod = gprod

        # Renewables and load spilled in node n at time t
        renewused, loadshed = {}, {}
        for t in taus:
            for n in nodes:
                for s in scenarios:
                    renewused[s, n, t] = m.addVar(lb=0.0, ub=wind[s, t, n])
                    loadshed[s, n, t] = m.addVar(lb=0.0, ub=load[n][t])
        self.variables.renewused = renewused
        self.variables.loadshed = loadshed

        # Flow on lines
        lineflow = {}
        for t in taus:
            for l in lines:
                for s in scenarios:
                    lineflow[s, l, t] = m.addVar(lb=-self.data.linelimit[l], ub=self.data.linelimit[l])
        self.variables.lineflow = lineflow

        # Nodal phase angles
        nodeangle = {}
        for t in taus:
            for n in nodes:
                for s in scenarios:
                    nodeangle[s, n, t] = m.addVar(lb=-gb.GRB.INFINITY, ub=gb.GRB.INFINITY)
        self.variables.nodeangle = nodeangle

        m.update()
        
        for t in taus:
            for n in self.data.slackbuses:
                for s in scenarios:
                    nodeangle[s, n, t].ub = 0
                    nodeangle[s, n, t].lb = 0


        # Slack bus setting

    def _build_objective(self):
        taus = self.data.taus
        nodes = self.data.nodeorder
        scenarios = self.data.scenarios
        generators = self.data.generators
        gendata = self.data.generatorinfo.T.to_dict()

        m = self.model
        m.setObjective(
            gb.quicksum(self.data.scenarioprob[s]*gendata[gen]['lincost']*self.variables.gprod[s, gen, t] for s in scenarios for gen in generators for t in taus) +
            gb.quicksum(self.data.scenarioprob[s]*self.variables.loadshed[s, n, t]*defaults.VOLL for s in scenarios for n in nodes for t in taus),
            gb.GRB.MINIMIZE)

    def _build_constraints(self):
        taus = self.data.taus
        scenarios = self.data.scenarios
        generators = self.data.generators
        gendata = self.data.generatorinfo.T.to_dict()
        nodes = self.data.nodeorder
        lines = self.data.lineorder
        wind = self.data.wind_scenarios
        load = self.data.load.to_dict()

        m = self.model
        loadshed, renewused, gprod = self.variables.loadshed, self.variables.renewused, self.variables.gprod
        lineflow, nodeangle = self.variables.lineflow, self.variables.nodeangle

        powerbalance = {}
        for t in taus:
            for n in nodes:
                for s in scenarios:
                    powerbalance[s, n, t] = m.addConstr(
                        gb.quicksum(gprod[s, g, t] for g in self.data.generatorsfornode[n]) +
                        loadshed[s, n, t] + renewused[s, n, t],
                        gb.GRB.EQUAL,
                        load[n][t] +
                        gb.quicksum(lineflow[s, l, t] for l in self.data.nodetooutlines[n]) -
                        gb.quicksum(lineflow[s, l, t] for l in self.data.nodetoinlines[n]))
        self.constraints.powerbalance = powerbalance

        flow_to_angle = {}
        for t in taus:
            for l in lines:
                for s in scenarios:
                    n1, n2 = l
                    flow_to_angle[s, l, t] = m.addConstr(
                        lineflow[s, l, t],
                        gb.GRB.EQUAL,
                        self.data.lineadmittance[l]*(nodeangle[s, n1, t] - nodeangle[s, n2, t]))

    ###
    #   Data updating
    ###
    def _add_new_data(self, wind_scenarios, load_signal):
        self.data.wind_scenarios = wind_scenarios
        self.data.load = load_signal
        pass

    def _update_constraints(self):
        taus = self.data.taus
        scenarios = self.data.scenarios
        generators = self.data.generators
        gendata = self.data.generatorinfo.T.to_dict()
        nodes = self.data.nodeorder
        edges = self.data.lineorder
        wind = self.data.wind_scenarios
        load = self.data.load.to_dict()
        powerbalance = self.constraints.powerbalance

        for n in nodes:
            for t in taus:
                for s in scenarios:
                    renewused[s, n, t].ub = wind[s, t, n]
                    loadshed[s, n, t].ub = load[n][t]

        for n in nodes:
            for t in taus:
                for s in scenarios:
                    powerbalance[s, n, t].rhs = load[n][t]
