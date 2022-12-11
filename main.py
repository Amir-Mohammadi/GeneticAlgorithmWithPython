
import numpy
import matplotlib.pyplot as pl
import geneticAlgorithm as ga

inputs = [2346,5678,8345,5723,6723,2345,2348,4652,2222,1111]
num_weights = 10
sol_per_pop = 8
num_parents_mating = 4

pop_size = (sol_per_pop, num_weights)
new_population = numpy.random.uniform(low=0, high=1.0, size=pop_size)
print(new_population,'\n')



num_generations = 5
for generation in range(num_generations):
    print("Generation : ", generation,'\n')
    fitness = ga.call_fitnees(inputs,new_population)
    print("Fitness is : ", fitness,'\n')

    parents = ga.select_pool(new_population,fitness,num_parents_mating)
    crossover = ga.crossover(parents,offspring_size=(pop_size[0]-parents.shape[0],num_weights))

    print("Crossover is : ", crossover,'\n')

    mutation = ga.mutation(crossover)
    print("Mutation is : " , mutation,'\n')

    new_population[0:parents.shape[0],:] = parents
    new_population[parents.shape[0]:,:] = mutation

    print("Best Result : ", numpy.max(numpy.sum(new_population*inputs,axis=1)),'\n')

fitness = ga.call_fitnees(inputs,new_population)
best_match = numpy.where(fitness == numpy.max(fitness))

best = new_population[best_match,:]
print('\n\n')
print("Best Solution : ", best,'\n')


print("Best Solution fitness : ", fitness[best_match],'\n')