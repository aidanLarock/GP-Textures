# Imports
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp
import operator
# Supporting Imports
import math
from PIL import Image
import numpy
# Parameter File
import params as params


# Reading in from text file
image = Image.open(params.image)
pixels = list(image.getdata())
(sizeX, sizeY) = (image.size[0],image.size[1])

# Define new functions
def protectedDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def evalSymbReg(individual):
    R = toolbox.compile(expr=individual[0]) #f1(x)
    G = toolbox.compile(expr=individual[1]) #f2(x)
    B = toolbox.compile(expr=individual[2]) #f3(x)

    x = 0
    for pixel in pixels:
        dR = (R[x] - pixel[0])
        dG = (G[x] - pixel[1])
        dB = (B[x] - pixel[2])
        Dist = math.sqrt((dR**2 + dG**2 + dB**2))
        x+=1
    return Dist,

pset = gp.PrimitiveSet("MAIN", params.terminals)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(protectedDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=17)
toolbox.register("individualR", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("individualG", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("individualB", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("populationR", tools.initRepeat, list, toolbox.individualR)
toolbox.register("populationG", tools.initRepeat, list, toolbox.individualG)
toolbox.register("populationB", tools.initRepeat, list, toolbox.individualB)
toolbox.register("compile", gp.compile, pset=pset)

#Registers the parameters for the genetic program. 
toolbox.register("evaluate", evalSymbReg) # In list
toolbox.register("select", tools.selTournament, tournsize=params.tournamentSize)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=params.maxDepth))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=params.maxDepth))

def run():
    pop = toolbox.population(n=params.popSize)
    hof = tools.HallOfFame(1)
    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    #Runs the genetic program
    pop, log = algorithms.eaSimple(pop, toolbox, params.crossoverRate, params.mutateRate, params.numGenerations, stats=mstats,
                                halloffame=hof, verbose=True) 
