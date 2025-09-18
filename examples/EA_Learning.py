# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import time

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
    
    # 2-point crossover for each weight matrix
    for w_child, w_p1, w_p2 in [(child.w1, parent1.w1, parent2.w1),
                               (child.w2, parent1.w2, parent2.w2),
                               (child.w3, parent1.w3, parent2.w3)]:
        # Get random crossover points
        points = sorted(np.random.choice(w_child.shape[0], size=2, replace=False))
        
        # Apply crossover
        if np.random.random() < 0.5:  
            w_child[:points[0]] = w_p2[:points[0]]
        if np.random.random() < 0.5: 
            w_child[points[0]:points[1]] = w_p2[points[0]:points[1]]
        if np.random.random() < 0.5: 
            w_child[points[1]:] = w_p2[points[1]:]
    
    return child

# Add these global variables at the top with other globals
MUTATION_STEP_SIZE = 0.1  # Initial sigma value
ADAPTATION_PERIOD = 3    # k iterations before adapting
ADAPTATION_FACTOR = 1.15  
SUCCESS_COUNTER = 0       # Track successful mutations
# Bounds for mutation step size to avoid vanishing or exploding sigma
MIN_MUTATION_STEP_SIZE = 1e-3
MAX_MUTATION_STEP_SIZE = 1.0

def mutate(individual: Individual, mutation_rate: float = 0.1) -> Individual:
    global MUTATION_STEP_SIZE, SUCCESS_COUNTER
    
    child = Individual(
        w1=individual.w1.copy(),
        w2=individual.w2.copy(),
        w3=individual.w3.copy()
    )
    
    # Apply Gaussian mutation with current step size
    mutations = [
        np.random.normal(0, MUTATION_STEP_SIZE, child.w1.shape),
        np.random.normal(0, MUTATION_STEP_SIZE, child.w2.shape),
        np.random.normal(0, MUTATION_STEP_SIZE, child.w3.shape)
    ]
    
    masks = [
        np.random.random(child.w1.shape) < mutation_rate,
        np.random.random(child.w2.shape) < mutation_rate,
        np.random.random(child.w3.shape) < mutation_rate
    ]
    
    child.w1 += mutations[0] * masks[0]
    child.w2 += mutations[1] * masks[1]
    child.w3 += mutations[2] * masks[2]
    
    return child

def tournament_selection(population: List[Individual], tournament_size: int = 3) -> Individual:
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda ind: ind.fitness)

# Global variable to store current individual being evaluated
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
    HISTORY = []  # Reset history
    
    # Create fresh world instance for each evaluation
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
    
    # Run without graphics
    simple_runner(model, data, duration=duration)
    
    # Calculate fitness (distance traveled in x direction)
    final_pos = HISTORY[-1]
    initial_pos = (0, 0)
    fitness = (
        (initial_pos[0] - final_pos[0]) ** 2
        + (initial_pos[1] - final_pos[1]) ** 2
    ) ** 0.5  # Euclidean distance

    return fitness

def rank_based_selection(population: List[Individual]) -> Individual:
    """Select individual based on fitness rank."""
    # Sort population by fitness
    sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    
    # Calculate rank-based probabilities (higher rank = higher probability)
    ranks = np.arange(len(sorted_pop), 0, -1)
    probabilities = ranks / ranks.sum()
    
    # Select based on rank probabilities
    return np.random.choice(sorted_pop, p=probabilities)

def evolutionary_algorithm(input_size: int, output_size: int, pop_size: int = 100, generations: int = 100):
    global MUTATION_STEP_SIZE, SUCCESS_COUNTER
    hidden_size = 8
    
    population = [
        Individual.create_random(input_size, hidden_size, output_size)
        for _ in range(pop_size)
    ]
    fitness_scores = []

    best_fitness = float('-inf')
    best_individual = None
    generation_counter = 0
    
    for gen in range(generations):
        # Evaluate population
        fitness_gen = []
        for ind in population:
            ind.fitness = evaluate_individual(ind, input_size, output_size)
            fitness_gen.append(ind.fitness)

            if ind.fitness > best_fitness:
                best_fitness = ind.fitness
                best_individual = Individual(ind.w1.copy(), ind.w2.copy(), ind.w3.copy(), ind.fitness)
                SUCCESS_COUNTER += 1  # Count successful mutation
        
        fitness_scores.append(fitness_gen)

        # Adapt mutation step size every ADAPTATION_PERIOD generations
        if gen > 0 and gen % ADAPTATION_PERIOD == 0:
            success_rate = SUCCESS_COUNTER / (pop_size * ADAPTATION_PERIOD)
            
            # Apply 1/5 success rule
            if success_rate > 0.2:  # More than 1/5 successful
                MUTATION_STEP_SIZE /= ADAPTATION_FACTOR
            elif success_rate < 0.2:  # Less than 1/5 successful
                MUTATION_STEP_SIZE *= ADAPTATION_FACTOR
            # If success_rate == 0.2, keep current step size
            
            # Clamp step size to safe bounds
            MUTATION_STEP_SIZE = max(MIN_MUTATION_STEP_SIZE, min(MAX_MUTATION_STEP_SIZE, MUTATION_STEP_SIZE))

            SUCCESS_COUNTER = 0  # Reset counter
            print(f"Generation {gen}: Mutation step size adjusted to {MUTATION_STEP_SIZE}")
        
        # Create new population
        new_population = [best_individual]  # elitism
        
        while len(new_population) < 3*pop_size:
            parent1  = rank_based_selection(population)
            parent2 = rank_based_selection(population)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

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
    """
    Plot fitness over generations showing mean, std deviation, and individual runs.
    
    Args:
        iterations_scores: List of lists, where each inner list contains fitness scores 
                          for each generation in one evolutionary run
    """
    num_generations = len(iterations_scores[0])
    
    # Extract best fitness per generation for each run
    best_fitness_per_run = []
    for run_scores in iterations_scores:
        run_best = [max(generation_scores) for generation_scores in run_scores]
        best_fitness_per_run.append(run_best)
    
    # Convert to numpy array for statistics
    fitness_array = np.array(best_fitness_per_run)  # Shape: (num_runs, num_generations)
    
    # Calculate statistics across runs
    mean_fitness = np.mean(fitness_array, axis=0)
    std_fitness = np.std(fitness_array, axis=0)
    max_fitness = np.max(fitness_array, axis=0)  # Maximum across all runs for each generation
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Generation numbers
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
    
    # Add standard deviation as shaded area
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
    
    # Add some statistics as text
    final_mean = mean_fitness[-1]
    final_std = std_fitness[-1]
    final_max = max_fitness[-1]
    
    stats_text = f'Final Mean: {final_mean:.3f} ± {final_std:.3f}\nFinal Max: {final_max:.3f}'
    plt.text(0.02, 0.98, stats_text, 
             transform=plt.gca().transAxes, 
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
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
        np.random.seed(SEED+i)
        random.seed(SEED+i)   
        best_individual, fitness_scores = evolutionary_algorithm(input_size, output_size)
        iteration_scores.append(fitness_scores)

    # Plot fitness across runs
    try:
        plot_fitness_over_generations(iteration_scores)
    except Exception as e:
        print("Failed to plot fitness over generations:", e)
    # Final visualization with best individual
    global CURRENT_INDIVIDUAL
    CURRENT_INDIVIDUAL = best_individual

    viewer.launch(model=model, data=data)
    show_qpos_history(HISTORY)


if __name__ == "__main__":
    main()


