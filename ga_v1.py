# Python3 program to create target string, starting from 
# random string using Genetic Algorithm 
  
import random 
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import copy


df = pd.read_csv(r"/home/ali/Desktop/GA-new-Version/dataset/course_dataset.csv")

# Number of individuals in each generation 
POPULATION_SIZE = 100
  
  
class Individual(object): 
    ''' 
    Class representing individual in population 
    '''
    def __init__(self, chromosome): 
        self.chromosome = chromosome  
        self.fitness = self.cal_fitness() 

    @classmethod
    def create_gnome(self): 
        ''' 
        create chromosome or string of genes 
        '''
        
        week = []
        for i in range(50):
            lecture_hall = []
            for j in range(0,8):
                    lecture_hall.append(random.randint(0, 102))
            week.append(lecture_hall)
            
        return week
  
    def mate(self, par2): 
        ''' 
        Perform mating and produce new offspring 
        '''
  
        # chromosome for offspring 
        child_chromosome = [] 
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):     
  
            # random probability   
            prob = random.random() 
  
            # if prob is less than 0.45, insert gene 
            # from parent 1  
            if prob < 0.45: 
                child_chromosome.append(gp1) 
  
            # if prob is between 0.45 and 0.90, insert 
            # gene from parent 2 
            elif prob < 0.90: 
                child_chromosome.append(gp2) 
  
            # otherwise insert random gene(mutate),  
            # for maintaining diversity 
            else: 
                lecture_hall = []
                for j in range(0,8):
                    lecture_hall.append(random.randint(0, 102))
                child_chromosome.append(lecture_hall) 
  
        # create new Individual(offspring) using  
        # generated chromosome for offspring 
        return Individual(child_chromosome) 
  
    def cal_fitness(self):
        def unique_classes_fitness():
            fitness = 0
            temp_list = []
            ctr = 0
            for chromosom in self.chromosome:
                temp_list.append(chromosom)
                ctr += 1
                if ctr % 10 == 0:
                    flatList = [e for l in temp_list for e in l]
                    counts = Counter(flatList)
                    fitness += sum(c for c in counts.values() if c > 2)
                    temp_list = []
            return fitness

        def teacher_availability_fitness():
            fitness = 0
            temp_list = []
            ctr = 0
            for chromosom in self.chromosome:
                temp_list.append(chromosom)
                ctr += 1
                if ctr % 10 == 0:
                    flatList = [e for l in temp_list for e in l]
                    for i in range(0, 8):
                        teachers = [df.loc[flatList[j]]['Teacher ID'] for j in range(i, 50, 9)]
                        counts = Counter(teachers)
                        fitness += sum(c for c in counts.values() if c > 1)
                    temp_list = []
            return fitness

        def section_semester_fitness():
            fitness = 0
            temp_list = []
            ctr = 0
            for chromosom in self.chromosome:
                temp_list.append(chromosom)
                ctr += 1
                if ctr % 10 == 0:
                    flatList = [e for l in temp_list for e in l]
                    for i in range(0, 8):
                        pairs = [(df.loc[flatList[j]]['Section'], df.loc[flatList[j]]['Semester']) for j in range(i, 50, 9)]
                        counts = Counter(pairs)
                        fitness += sum(c for c in counts.values() if c > 1)
                    temp_list = []
            return fitness

        def credit_hours_fitness():
            flatList = [e for l in self.chromosome for e in l]
            counts = Counter(flatList)
            return sum(c for c in counts.values() if c > 3)

        def wednesday_slot_fitness():
            wednesday = self.chromosome[18:29]
            flat = [e for l in wednesday for e in l]
            return -sum(1 for i in range(4, 50, 9) if flat[i] == 102)

        def preferences_fitness():
            flatList = [e for l in self.chromosome for e in l]
            fitness = 0
            pref = {4: [20, 21, 22, 23, 79, 2, 6, 34, 88, 101], 7: [81, 39, 66, 33, 99]}
            for i in range(0, 50):
                if flatList[i] in pref[4] and (i % 4 == 0 or i % 3 == 0 or i % 2 == 0):
                    fitness -= 1
                if flatList[i] in pref[7] and (i % 7 == 0 or i % 5 == 0 or i % 6 == 0):
                    fitness -= 1
            return fitness

    # Thread execution
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(unique_classes_fitness),
                executor.submit(teacher_availability_fitness),
                executor.submit(section_semester_fitness),
                executor.submit(credit_hours_fitness),
                executor.submit(wednesday_slot_fitness),
                executor.submit(preferences_fitness)
            ]
            results = [f.result() for f in futures]
        
        return sum(results)
            
        

def simulated_annealing(individual, temperature=1.0, cooling_rate=0.95):
        current = individual
        best = Individual(current.chromosome.copy())  # Initialize best with a copy of the current individual
        NUM_LISTS = 50
        NUM_ELEMENTS_PER_LIST = 8
        GENES = list(range(0, 102))
        sample = current.chromosome.copy()  # Use the current individual's chromosome as the sample
        while temperature > 0.01:
            # Generate a slightly mutated individu

            # Deep copy to avoid mutating original sample
            new_population = copy.deepcopy(sample)

            # Total number of elements = 50 lists * 8 = 400
            # Choose how many to change
            num_changes = random.randint(0, 5)

            # Get all possible (i, j) positions
            all_indices = [(i, j) for i in range(len(sample)) for j in range(len(sample[0]))]

            # Randomly choose indices to change
            indices_to_change = random.sample(all_indices, num_changes)

            # Apply changes
            for i, j in indices_to_change:
                old_value = new_population[i][j]
                # Choose a new value different from the old one
                new_value = random.choice([x for x in range(102) if x != old_value])
                new_population[i][j] = new_value

            new_individual = Individual(new_population)
            print(new_individual.chromosome)
            print("Fitness: ", new_individual.fitness)
            # Update best if we find a better solution
            if new_individual.fitness < best.fitness:
                best = new_individual
                current = new_individual  # Set current to new better solution immediately
                sample = best.chromosome.copy()  # Update sample to the best found solution
            # Otherwise, accept new_individual with some probability to allow exploration
            elif random.random() < np.exp((current.fitness - new_individual.fitness) / temperature):
                current = new_individual

            # Reduce the temperature
            temperature *= cooling_rate

        return best  # Return the best solution found



def main(): 
    generation = 1
    ANNEALING_THRESHOLD = 50
    no_improvement_count = 0
    found = False
    with ThreadPoolExecutor() as executor:
        population = list(executor.map(lambda _: Individual(Individual.create_gnome()), range(POPULATION_SIZE)))
    while not found:
        population = sorted(population, key=lambda x: x.fitness) 
        
        if population[0].fitness == 0: 
            found = True
            break
        else:
            no_improvement_count += 1
            if no_improvement_count >= ANNEALING_THRESHOLD:
                print("Applying Simulated Annealing...")
                population[0] = simulated_annealing(population[0])
                no_improvement_count = 0
        s = int(0.10 * POPULATION_SIZE)
        
        new_generation = population[:s]
        def mate_random():
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            return parent1.mate(parent2)
        
        with ThreadPoolExecutor() as executor:
            children = list(executor.map(lambda _: mate_random(), range(POPULATION_SIZE - s)))
            
        new_generation.extend(children)
        population = new_generation
        
        
        print("Generation: {}\tString: {}\tFitness: {}",generation, population[0].fitness)
        generation += 1

     
if __name__ == '__main__': 
	main()