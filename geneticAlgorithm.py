import numpy 

def call_fitnees(inputs , pop):
    fitness = numpy.sum(pop*inputs, axis=1)
    return fitness

def select_pool(pop , fitness , num_parents):
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness = numpy.where(fitness == numpy.max(fitness))
        max_fitness = max_fitness[0][0]
        parents[parent_num,:] = pop[max_fitness,:]
        fitness[max_fitness] = -99999999999
    return parents

def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    crossover_point = numpy.uint8(offspring_size[1]/2)

    for k in range(offspring_size[0]):
        parent1 = k%parents.shape[0]
        parent2 = (k+1)%parents.shape[0]
        offspring[k, 0:crossover_point] = parents[parent1,0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2, crossover_point:]
    return offspring

def mutation(crossover):
    for id in range(crossover.shape[0]):
        random_value = numpy.random.uniform(-1.0,1.0,1)
        crossover[id,4] = crossover[id,4] + random_value
    return crossover