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
    y = sizeY
    for pixel in pixels:
        dR = (R(x,y) - pixel[0])
        dG = (G(x,y) - pixel[1])
        dB = (B(x,y) - pixel[2])
        Dist = math.sqrt((dR**2 + dG**2 + dB**2))
        x+=1
        if x == sizeX+1:
            x = 0
            y -= 1
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

def run(pop, toolbox, halloffame, crossoverRate = params.crossoverRate, mutateRate = params.mutateRate, numGenerations = params.numGenerations, verbose=True):
    popr = pop[0]
    popg = pop[1]
    popb = pop[2]

    hofR = halloffame[0]
    hofG = halloffame[0]
    hofB = halloffame[0]

    for n in range(1, numGenerations+1): 

        redSpring = tools.select(popr, len(popr))
        greenSpring = tools.select(popg, len(popg))
        blueSpring = tools.select(popb, len(popb))

        redSpring = algorithms.varAnd(redSpring, toolbox, crossoverRate, mutateRate)
        greenSpring = algorithms.varAnd(greenSpring, toolbox, crossoverRate, mutateRate)
        blueSpring = algorithms.varAnd(blueSpring, toolbox, crossoverRate, mutateRate)

        indR = [ind for ind in popr if not ind.fitness.valid]
        indG = [ind for ind in popg if not ind.fitness.valid]
        indB = [ind for ind in popb if not ind.fitness.valid]

        fitnessRed = toolbox.map(toolbox.evaluate, zip(indR, indG, indB))
        for ind, fit in zip(indR, fitnessRed):
            ind.fitness.values = fit

        fitnessGreen = toolbox.map(toolbox.evaluate, zip(indR, indG, indB))
        for ind, fit in zip(indG, fitnessGreen):
            ind.fitness.values = fit

        fitnessBlue = toolbox.map(toolbox.evaluate, zip(indR, indG, indB))
        for ind, fit in zip(indB, fitnessBlue):
            ind.fitness.values = fit

        hofR.update(redSpring)
        hofG.update(greenSpring)
        hofB.update(blueSpring)
        popr[:] = redSpring
        popg[:] = greenSpring
        popb[:] = blueSpring

    return [popr, popg, popb]


def main():
    #Registers the parameters for the genetic program. 
    toolbox.register("evaluate", evalSymbReg) # In list
    toolbox.register("select", tools.selTournament, tournsize=params.tournamentSize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=params.maxDepth))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=params.maxDepth))


    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    popr = toolbox.populationR(n=params.popSize)
    popg = toolbox.populationG(n=params.popSize)
    popb = toolbox.populationB(n=params.popSize)

    pop = [popr, popg, popb]
    hor = tools.HallOfFame(1)
    hog = tools.HallOfFame(1)
    hob = tools.HallOfFame(1)
    
    hof = [hor,hog,hob]

    pop = run(pop, toolbox, hof)


    """
    #Selects the best individual from the final generation
    best = tools.selBest(pop, k=1)
    best = best[0]
    print("Best Individual: ")
    print(best)
    tree = gp.compile(best,pset)

    from datetime import datetime

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S.%f)")

    path = "\NewImages\*"+timestampStr

    #Using the best selected individual runs the testing data
    #on the best selected tree. Then creates the confusion matrix
    #for this particular run. 
    print("Running Tests: ")
    for x in sizeX:
        for y in sizeY:
            valR = tree(x,y)
            valG = tree(x,y)
            valB = tree(x,y)

            (valR,valG,valB)
    """    

if __name__ == "__main__":
    main()