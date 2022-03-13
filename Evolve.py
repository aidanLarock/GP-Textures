#    This file is part of EAP.
#
#    EAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    EAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with EAP. If not, see <http://www.gnu.org/licenses/>.

import operator
import math
import random
import pandas as pd
import numpy as np
from functools import reduce
from operator import add, itemgetter
import params as p

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp


def runs(in_, sizeX, sizeY, colour):
    # Define new functions
    def protectedDiv(left, right):
        try:
            return left / right
        except ZeroDivisionError:
            return 1

    pset = gp.PrimitiveSet("MAIN", p.terminals)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    pset.addPrimitive(protectedDiv, 2)
    pset.addPrimitive(operator.neg, 1)

    #pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
    pset.renameArguments(ARG0='x')
    pset.renameArguments(ARG1='y')

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    def eval(individual):
        dist = 0
        func = toolbox.compile(expr=individual)
        i = 0
        for Y in range(sizeY):
            for X in range(sizeX):
                dist = dist + (func(X,Y)-in_[i])**2
                i += 1
        return math.sqrt(dist),

    toolbox.register("evaluate", eval)
    toolbox.register("select", tools.selTournament, tournsize=p.tournamentSize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=p.maxDepth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=p.maxDepth))

    def go(colour):
        random.seed(p.seed)

        pop = toolbox.population(n=p.popSize)
        hof = tools.HallOfFame(1)

        stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, toolbox, p.crossoverRate, p.mutateRate, p.numGenerations, stats=mstats,halloffame=hof, verbose=True)    

        chapter_keys = log.chapters.keys()
        sub_chaper_keys = [c[0].keys() for c in log.chapters.values()]

        data = [list(map(itemgetter(*skey), chapter)) for skey, chapter 
                    in zip(sub_chaper_keys, log.chapters.values())]
        data = np.array([[*a, *b] for a, b in zip(*data)])

        columns = reduce(add, [["_".join([x, y]) for y in s] 
                            for x, s in zip(chapter_keys, sub_chaper_keys)])
        df = pd.DataFrame(data, columns=columns)

        keys = log[0].keys()
        data = [[d[k] for d in log] for k in keys]
        for d, k in zip(data, keys):
            df[k] = d
        df.to_csv(colour+".csv")

        # print log and best
        best = tools.selBest(pop, k=1)
        print("Best Individual: ")
        #print(best[0])
        best = best[0]
        print(best)
        func = gp.compile(best,pset)
        print(func(1))
    go(colour)