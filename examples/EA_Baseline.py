# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pickle
import os

# Local libraries
from ariel.utils.renderers import video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld

# import prebuilt robot phenotypes
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko

# For evolutionary algorithm
from dataclasses import dataclass
import random
from typing import List, Tuple

# Deterministic seed for reproducibility
SEED = 42

@dataclass
class Individual:
    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray
    fitness: float = 0.0

    @classmethod
    def create_random(cls, input_size: int, hidden_size: int, output_size: int):
        return cls(
            w1=np.random.randn(input_size, hidden_size) * 0.1,
            w2=np.random.randn(hidden_size, hidden_size) * 0.1,
            w3=np.random.randn(hidden_size, output_size) * 0.1
        )
    

def crossover(parent1: Individual, parent2: Individual) -> Individual:
    child = Individual(
        w1=parent1.w1.copy(),
        w2=parent1.w2.copy(),
        w3=parent1.w3.copy()
    )
    
    # List all weights 
    for w_child, w_p1, w_p2 in [(child.w1, parent1.w1, parent2.w1),
                               (child.w2, parent1.w2, parent2.w2),
                               (child.w3, parent1.w3, parent2.w3)]:
        # Get 2 random crossover points
        points = sorted(np.random.choice(w_child.shape[0], size=2, replace=False))
        
        # Apply crossover
        if np.random.random() < 0.5:  
            w_child[:points[0]] = w_p2[:points[0]]
        if np.random.random() < 0.5: 
            w_child[points[0]:points[1]] = w_p2[points[0]:points[1]]
        if np.random.random() < 0.5: 
            w_child[points[1]:] = w_p2[points[1]:]
    
    return child

# Global parameters 
MUTATION_STEP_SIZE = 0.1  
ADAPTATION_PERIOD = 3   
ADAPTATION_FACTOR = 1.15  
SUCCESS_COUNTER = 0       
# Min and Max for step_size
MIN_MUTATION_STEP_SIZE = 1e-3
MAX_MUTATION_STEP_SIZE = 1.0

def mutate(individual: Individual, mutation_rate: float = 0.1) -> Individual:
    global MUTATION_STEP_SIZE, SUCCESS_COUNTER
    
    child = Individual(
        w1=individual.w1.copy(),
        w2=individual.w2.copy(),
        w3=individual.w3.copy()
    )
    
    # Apply normal distribution
    mutations = [
        np.random.normal(0, MUTATION_STEP_SIZE, child.w1.shape),
        np.random.normal(0, MUTATION_STEP_SIZE, child.w2.shape),
        np.random.normal(0, MUTATION_STEP_SIZE, child.w3.shape)
    ]
    # Create mutation masks
    masks = [
        np.random.random(child.w1.shape) < mutation_rate,
        np.random.random(child.w2.shape) < mutation_rate,
        np.random.random(child.w3.shape) < mutation_rate
    ]
    # Apply mutation 
    child.w1 += mutations[0] * masks[0]
    child.w2 += mutations[1] * masks[1]
    child.w3 += mutations[2] * masks[2]
    
    return child

def tournament_selection(population: List[Individual], tournament_size: int = 3) -> Individual:
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)

CURRENT_INDIVIDUAL = None

HISTORY = []
def controller(model, data, to_track):
    global CURRENT_INDIVIDUAL
    
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    inputs = data.qpos.copy()
    layer1 = sigmoid(np.dot(inputs, CURRENT_INDIVIDUAL.w1))
    layer2 = sigmoid(np.dot(layer1, CURRENT_INDIVIDUAL.w2))
    outputs = sigmoid(np.dot(layer2, CURRENT_INDIVIDUAL.w3))
    
    data.ctrl += ((outputs * np.pi) - np.pi/2) * 0.1
    HISTORY.append(to_track[0].xpos.copy())



def evaluate_individual(individual: Individual, input_size: int, output_size: int, duration: float = 15.0) -> float:
    global CURRENT_INDIVIDUAL, HISTORY
    CURRENT_INDIVIDUAL = individual
    HISTORY = [] 
    
    # Create a fresh world so that each robot is tested on its brain and not the world
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)
    
    # Run simulation
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    mujoco.set_mjcb_control(lambda m,d: controller(m, d, to_track))
    
    # Simple Run
    simple_runner(model, data, duration=duration)
    
    # Calculate fitness based on euclidean distance from origin
    final_pos = HISTORY[-1]
    initial_pos = (0, 0)
    fitness = (
        (initial_pos[0] - final_pos[0]) ** 2
        + (initial_pos[1] - final_pos[1]) ** 2
    ) ** 0.5  

    return fitness

def rank_based_selection(population: List[Individual]) -> Individual:
    # Sort population by fitness
    sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    
    # Calculate rank-based probabilities 
    ranks = np.arange(len(sorted_pop), 0, -1)
    probabilities = ranks / ranks.sum()
    
    # Select based on rank probabilities
    return np.random.choice(sorted_pop, p=probabilities)

def evolutionary_algorithm(input_size: int, output_size: int, pop_size: int = 50, generations: int = 3):
    global MUTATION_STEP_SIZE, SUCCESS_COUNTER
    hidden_size = 8
    
    population = [
        Individual.create_random(input_size, hidden_size, output_size)
        for _ in range(pop_size)
    ]
    # Create list to track all fitnesses per run
    fitness_scores = []

    best_fitness = float('-inf')
    best_individual = None
    generation_counter = 0
    
    for gen in range(generations):
        # Create list to track all fitnesses per generation
        fitness_gen = []
        for ind in population:
            ind.fitness = evaluate_individual(ind, input_size, output_size)
            fitness_gen.append(ind.fitness)

            if ind.fitness > best_fitness:
                best_fitness = ind.fitness
                best_individual = Individual(ind.w1.copy(), ind.w2.copy(), ind.w3.copy(), ind.fitness)
                SUCCESS_COUNTER += 1  
        
        fitness_scores.append(fitness_gen)

        # Adapt mutation step size 
        if gen > 0 and gen % ADAPTATION_PERIOD == 0:
            success_rate = SUCCESS_COUNTER / (pop_size * ADAPTATION_PERIOD)
            
            # Apply 1/5 success rule
            if success_rate > 0.2:  
                MUTATION_STEP_SIZE /= ADAPTATION_FACTOR
            elif success_rate < 0.2:  
                MUTATION_STEP_SIZE *= ADAPTATION_FACTOR
            
            
            # Make step size bound to min and max
            MUTATION_STEP_SIZE = max(MIN_MUTATION_STEP_SIZE, min(MAX_MUTATION_STEP_SIZE, MUTATION_STEP_SIZE))

            SUCCESS_COUNTER = 0  
            print(f"Generation {gen}: Mutation step size adjusted to {MUTATION_STEP_SIZE}")
        
        # Create new population and add best individual (elitism)
        new_population = [best_individual] 

        # Generate population with children based on rank selection
        while len(new_population) < 3*pop_size:
            parent1  = rank_based_selection(population)
            parent2 = rank_based_selection(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        #generate offspring based on tournament selection
        offspring_pop = [best_individual]
        while len(offspring_pop) < pop_size:
            offspring = tournament_selection(new_population[1:])
            offspring_pop.append(offspring)

        population = offspring_pop
        print(f"Generation {gen}: Best Fitness = {best_fitness}")
    
    return best_individual, fitness_scores  
    

def show_qpos_history(history:list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    
    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    plt.show()


def plot_fitness_over_generations(iterations_scores):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    num_generations = len(iterations_scores[0])
    
    # Extract best fitness per generation for each run
    best_fitness_per_run = []
    for run_scores in iterations_scores:
        run_best = [max(generation_scores) for generation_scores in run_scores]
        best_fitness_per_run.append(run_best)
    
    fitness_array = np.array(best_fitness_per_run)  
    
    # Calculate mean, std and max for all runs
    mean_fitness = np.mean(fitness_array, axis=0)
    std_fitness = np.std(fitness_array, axis=0)
    max_fitness = np.max(fitness_array, axis=0) 
    
    # Save data to file with timestamp
    data_filename = f"data_{timestamp}.pkl"
    data_to_save = {
        'iterations_scores': iterations_scores,
        'best_fitness_per_run': best_fitness_per_run,
        'fitness_array': fitness_array,
        'mean_fitness': mean_fitness,
        'std_fitness': std_fitness,
        'max_fitness': max_fitness,
        'num_generations': num_generations,
        'timestamp': timestamp
    }
    
    # Create directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    with open(os.path.join('results', data_filename), 'wb') as f:
        pickle.dump(data_to_save, f)
    print(f"Data saved to: results/{data_filename}")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    generations = np.arange(num_generations)
    
    # Plot mean line
    plt.plot(generations, mean_fitness, 
            color='black', 
            linewidth=3, 
            label='Mean')
    
    # Plot maximum line
    plt.plot(generations, max_fitness, 
            color='red', 
            linewidth=3, 
            linestyle='--',
            label='Max across runs')
    
    # Plot std area
    plt.fill_between(generations, 
                     mean_fitness - std_fitness, 
                     mean_fitness + std_fitness,
                     color='gray', 
                     alpha=0.3, 
                     label='±1 Standard Deviation')
    
    # Customize the plot
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title('Fitness Evolution Over Generations\n(3 Independent Runs)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add statistics as text
    final_mean = mean_fitness[-1]
    final_std = std_fitness[-1]
    final_max = max_fitness[-1]
    
    stats_text = f'Final Mean: {final_mean:.3f} ± {final_std:.3f}\nFinal Max: {final_max:.3f}'
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot to file with timestamp
    plot_filename = f"plots_{timestamp}.png"
    plt.savefig(os.path.join('results', plot_filename), dpi=300, bbox_inches='tight')
    print(f"Plot saved to: results/{plot_filename}")
    
    plt.show()

def main():
    np.random.seed(SEED)
    random.seed(SEED)

    # Create fresh world for visualization
    world = SimpleFlatWorld()
    gecko_core = gecko()
    world.spawn(gecko_core.spec, spawn_position=[0, 0, 0])
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Get dimensions for neural network
    input_size = len(model.qpos0)
    output_size = model.nu

    # Run evolutionary algorithm with dimensions
    iteration_scores = []
    for i in range(3):
        # Reset global variables for each run
        global MUTATION_STEP_SIZE, SUCCESS_COUNTER, CURRENT_INDIVIDUAL, HISTORY
        MUTATION_STEP_SIZE = 0.1  
        SUCCESS_COUNTER = 0       
        CURRENT_INDIVIDUAL = None 
        HISTORY = []              
        
        np.random.seed(SEED+i)
        random.seed(SEED+i)   
        best_individual, fitness_scores = evolutionary_algorithm(input_size, output_size)
        iteration_scores.append(fitness_scores)

    # Plot fitness across runs
    try:
        plot_fitness_over_generations(iteration_scores)
    except Exception as e:
        print("Failed to plot fitness over generations:", e)
    global CURRENT_INDIVIDUAL
    CURRENT_INDIVIDUAL = best_individual

    PATH_TO_VIDEO_FOLDER = "./__videos__"
    video_recorder = VideoRecorder(output_folder=PATH_TO_VIDEO_FOLDER)

    # Render with video recorder
    video_renderer(
        model,
        data,
        duration=30,
        video_recorder=video_recorder,
    )
    show_qpos_history(HISTORY)


if __name__ == "__main__":
    main()