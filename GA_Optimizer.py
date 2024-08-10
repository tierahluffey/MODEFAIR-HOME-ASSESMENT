import random
import math
import json
import time
import numpy as np
from abc import ABC, abstractmethod
from scipy.spatial import distance_matrix

class GeneticAlgorithm(ABC):
    def __init__(self, pop_size, selection_size, num_gens, mutation_rate,
                 elite_size, crossover_rate, num_pts):
        assert pop_size > elite_size
        self.pop_size = pop_size
        self.selection_size = selection_size
        self.num_gens = num_gens
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.crossover_rate = crossover_rate
        self.num_pts = num_pts
        self.population = []

    @abstractmethod
    def crossover(self, parent1, parent2):
        pass

    @abstractmethod
    def mutate(self, individual):
        pass

    @abstractmethod
    def init_population(self):
        pass

    @abstractmethod
    def calc_fitness(self, individual):
        pass

    def rank_population(self):
        fit_results = [(individual, self.calc_fitness(individual)) for individual, _ in self.population]
        return sorted(fit_results, key=lambda x: x[1], reverse=True)

    def next_generation(self):
        ranked_population = self.rank_population()
        selected_population = ranked_population[:self.elite_size].copy()

        for _ in range(int((self.pop_size - self.elite_size) / 2)):
            if random.random() < self.crossover_rate:
                parent1 = sorted(
                    random.choices(ranked_population, k=self.selection_size),
                    key=lambda x: x[1], reverse=True
                )[0][0]
                parent2 = sorted(
                    random.choices(ranked_population, k=self.selection_size),
                    key=lambda x: x[1], reverse=True
                )[0][0]
                child1, child2 = self.crossover(parent1, parent2)
            else:
                child1 = random.choices(selected_population)[0][0]
                child2 = random.choices(selected_population)[0][0]

            if random.random() < self.mutation_rate:
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

            selected_population.append((child1, self.calc_fitness(child1)))
            selected_population.append((child2, self.calc_fitness(child2)))

        self.population = selected_population

    def get_sol(self):
        solution = self.rank_population()[0]
        individual = solution[0]
        fitness = solution[1]
        return individual, fitness

    def solve(self, early_stop: bool = True):
        if len(self.population) == 0:
            self.init_population()
        solution_lists = []
        for i in range(self.num_gens):
            self.next_generation()
            if i % 10 == 0:
                solution_lists.append(self.get_sol())
            if early_stop and len(solution_lists) >= 3 and abs(solution_lists[-1][1] - np.mean([fitness for _, fitness in solution_lists[-3:]])) < 1e-6:
                print("Early termination. Reason: converged.")
                break
            if len(solution_lists) > 10:
                solution_lists = solution_lists[-3:]

class MVPGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self, car_types, cars, custs, depot, *kwargs):
        self.car_types = car_types
        self.cars = cars
        self.custs = custs
        self.dist_matrix = self._generate_dist_matrix(depot, custs, len(cars))
        super().__init__(*kwargs)

    def _generate_dist_matrix(self, depot, custs, num_cars):
        points = [cust['pt'] for cust in custs]
        dist_car = [math.dist(depot, point) * 100 for point in points]
        dist_matrix = distance_matrix(points, points) * 100

        out_arr = np.tile(np.array(dist_car)[:, np.newaxis], (1, num_cars))
        up_arr = np.hstack((dist_matrix, out_arr))
        low_arr = np.array([dist_car + [0] * num_cars] * num_cars)
        dist_matrix = np.vstack((up_arr, low_arr))
        return dist_matrix

    def init_population(self):
        self.population = []
        for _ in range(self.pop_size):
            individual = np.hstack((np.array([0]), np.random.permutation([i for i in range(1, self.num_pts)]), np.array([self.num_pts])))
            self.population.append((individual, None))
        self.population = self.rank_population()

    def mutate(self, individual):
        for first_id in range(1, len(individual) - 1):
            if random.random() < 0.2:
                sec_id = random.randint(1, len(individual) - 2)
                individual[first_id], individual[sec_id] = individual[sec_id], individual[first_id]
        return individual

    def crossover(self, parent1, parent2):
        parent1_last_id = parent1[-1]
        parent2_last_id = parent2[-1]
        parent1 = parent1[1:-1].copy()
        parent2 = parent2[1:-1].copy()
        changable_node_nums = self.num_pts - 1
        c1 = int(random.random() * changable_node_nums) + 1
        c2 = int(random.random() * changable_node_nums) + 1
        if c2 < c1:
            c1, c2 = c2, c1
        C1 = [None for _ in parent2]
        C2 = [None for _ in parent1]
        C1[c1 - 1: c2] = parent2[c1 - 1: c2]
        C2[c1 - 1: c2] = parent1[c1 - 1: c2]
        L1 = np.hstack((parent1[c2:], parent1[:c2]))
        L2 = np.hstack((parent2[c2:], parent2[:c2]))
        L1_p = [i for i in L1 if i not in C1]
        L2_p = [i for i in L2 if i not in C2]
        for i, j in enumerate(L1_p):
            C1[(c2+i)%len(C1)] = j
        for i, j in enumerate(L2_p):
            C2[(c2+i)%len(C2)] = j
        C1 = np.hstack(([0], C1, [parent1_last_id]))
        C2 = np.hstack(([0], C2, [parent2_last_id]))
        return C1, C2

    def calc_fitness(self, individual, show_log=False):
        total_cost, total_distance = 0, 0
        assigned_cars = [-1 for _ in range(len(individual))]
        cur_car_id = -1
        individual_cars = [{"cost": 0, "distance": 0, "demand": 0, "route": []} for _ in self.cars]

        for i_r, j in enumerate(reversed(individual)):
            i = len(individual) - i_r - 1
            if j == 0:
                continue
            if j > len(self.custs):
                cur_car_id = j
                weight = self.car_types[self.cars[cur_car_id - 1 - len(self.custs)]['type']]['cost']
                distance = self.dist_matrix[individual[i-1] - 1, individual[i] - 1]
                assigned_cars[i] = cur_car_id
                total_cost += weight * distance
                total_distance += distance
                individual_cars[cur_car_id - 1 - len(self.custs)]['distance'] += distance
                individual_cars[cur_car_id - 1 - len(self.custs)]['cost'] += weight * distance
                individual_cars[cur_car_id - 1 - len(self.custs)]['route'].append(f"C{individual[i]} ({distance:.3f} km)")
            else:
                assert cur_car_id != -1
                weight = self.car_types[self.cars[cur_car_id - 1 - len(self.custs)]['type']]['cost']
                distance = self.dist_matrix[individual[i-1] - 1, individual[i] - 1]
                assigned_cars[i] = cur_car_id
                total_cost += weight * distance
                total_distance += distance
                individual_cars[cur_car_id - 1 - len(self.custs)]['demand'] += self.custs[j - 1]['demand']
                individual_cars[cur_car_id - 1 - len(self.custs)]['distance'] += distance
                individual_cars[cur_car_id - 1 - len(self.custs)]['cost'] += weight * distance
                individual_cars[cur_car_id - 1 - len(self.custs)]['route'].append(f"C{individual[i]} ({distance:.3f} km)")

        for car in individual_cars:
            car['route'].append(f"Depot ({self.dist_matrix[individual[-2] - 1, individual[-1] - 1]:.3f} km)")

        fitness = total_cost
        max_demand = sum([c['demand'] for c in self.custs])

        if show_log:
            self.print_results(total_distance, total_cost, individual_cars)

        if fitness > 120:
            fitness = 1e+10
        if max_demand == 0:
            return fitness

        total_demand = sum([c['demand'] for c in self.custs])
        fitness += abs(total_demand - sum([c['demand'] for c in individual_cars]))

        return fitness

    def print_results(self, total_distance, total_cost, individual_cars):
        print(f"Total Distance = {total_distance:.3f} km")
        print(f"Total Cost = RM {total_cost:.2f}")
        for i, car in enumerate(individual_cars):
            if car['route']:
                print(f"Vehicle {i + 1} (Type {'A' if i < len(self.cars) // 2 else 'B'}):")
                print(f"Round Trip Distance: {car['distance']:.3f} km, Cost: RM {car['cost']:.2f}, Demand: {car['demand']}")
                route_str = " -> ".join(car['route'])
                print(f"Depot -> {route_str}")
                print()
