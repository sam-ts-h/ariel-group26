# Third-party libraries
import numpy as np
import mujoco
from mujoco import viewer
import matplotlib.pyplot as plt
import time
import nevergrad as ng
from ariel.utils.renderers import video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from dataclasses import dataclass
import random
from typing import List, Tuple

#seed so all runs are the same (everytime)
SEED = 42
#Variables used in all code moved here for better readability
MUTATION_STEP_SIZE = 0.1 
ADAPTATION_PERIOD = 3    
ADAPTATION_FACTOR = 1.15  
SUCCESS_COUNTER = 0       
MIN_MUTATION_STEP_SIZE = 1e-3
MAX_MUTATION_STEP_SIZE = 1.0
CURRENT_INDIVIDUAL = None
CURRENT_INDIVIDUAL_FOR_LEARNING = None
DURATION = 30.0
#represent individual in pop
@dataclass
class Individual:
    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray
    fitness: float = 0.0
    #random init
    @classmethod
    def create_random(cls, input_size: int, hidden_size: int, output_size: int):
        return cls(
            w1=np.random.randn(input_size, hidden_size) * 0.1,
            w2=np.random.randn(hidden_size, hidden_size) * 0.1,
            w3=np.random.randn(hidden_size, output_size) * 0.1
        )
    
#crossover wbased on two points. All three segments have 50/50 chance of being 1 or 2
def crossover(parent1: Individual, parent2: Individual) -> Individual:
    child = Individual(
        w1=parent1.w1.copy(),
        w2=parent1.w2.copy(),
        w3=parent1.w3.copy()
    )
    
    # 2-point crossover
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


#mutate with params that change when individuals dont get better
def mutate(individual: Individual, mutation_rate: float = 0.1) -> Individual:
    global MUTATION_STEP_SIZE, SUCCESS_COUNTER
    
    child = Individual(
        w1=individual.w1.copy(),
        w2=individual.w2.copy(),
        w3=individual.w3.copy()
    )
    
    # normal distribution with mutation step size
    mutations = [
        np.random.normal(0, MUTATION_STEP_SIZE, child.w1.shape),
        np.random.normal(0, MUTATION_STEP_SIZE, child.w2.shape),
        np.random.normal(0, MUTATION_STEP_SIZE, child.w3.shape)
    ]
    # mask to decide which weights to mutate
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

#fitness only used for nevergrad with learning individual
def fitness_function_for_nevergrad(*weights_args):
    global CURRENT_INDIVIDUAL_FOR_LEARNING
    
    weights = np.array(weights_args)
    
    # Get the dimensions
    w1_size = CURRENT_INDIVIDUAL_FOR_LEARNING.w1.size
    w2_size = CURRENT_INDIVIDUAL_FOR_LEARNING.w2.size
    w3_size = CURRENT_INDIVIDUAL_FOR_LEARNING.w3.size
    
    # Reshape flattened weights
    w1 = weights[:w1_size].reshape(CURRENT_INDIVIDUAL_FOR_LEARNING.w1.shape)
    w2 = weights[w1_size:w1_size+w2_size].reshape(CURRENT_INDIVIDUAL_FOR_LEARNING.w2.shape)
    w3 = weights[w1_size+w2_size:w1_size+w2_size+w3_size].reshape(CURRENT_INDIVIDUAL_FOR_LEARNING.w3.shape)
    
    # Create temporary individual
    temp_individual = Individual(w1=w1, w2=w2, w3=w3)
    
    # Evaluate fitness
    fitness = evaluate_individual_internal(temp_individual, duration=DURATION)
    
    # Return negative fitness since nevergrad minimizes
    return -fitness

#Internal eval method used for both nevergrad and final evaluation
def evaluate_individual_internal(individual: Individual, duration: float = DURATION) -> float:
    """Internal evaluation function used during learning (shorter duration)"""
    global CURRENT_INDIVIDUAL, HISTORY
    CURRENT_INDIVIDUAL = individual
    HISTORY = []
    
    # Create fresh world instance eval
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
    if len(HISTORY) == 0:
        return 0.0
        
    final_pos = HISTORY[-1]
    initial_pos = (0, 0)
    fitness = (
        (initial_pos[0] - final_pos[0]) ** 2
        + (initial_pos[1] - final_pos[1]) ** 2
    ) ** 0.5  # Euclidean distance

    return fitness

#learn using nevergrad package
def learn_individual_with_nevergrad(individual: Individual, num_iterations: int = 10) -> Individual:

    global CURRENT_INDIVIDUAL_FOR_LEARNING
    CURRENT_INDIVIDUAL_FOR_LEARNING = individual
    
    # Flatten the individual's weights
    initial_weights = np.concatenate([
        individual.w1.flatten(),
        individual.w2.flatten(),
        individual.w3.flatten()
    ])
    parametrization = ng.p.Array(init=initial_weights)
    
    #using CMA optimizer from nevergrad
    optimizer = ng.optimizers.CMA(parametrization=parametrization, budget=num_iterations)
    
    # Run optimization
    best_fitness = float('inf')
    for i in range(num_iterations):
        # Get candidate solution
        x = optimizer.ask()
        
        # Evaluate fitness
        fitness = fitness_function_for_nevergrad(*x.value)
        
        # Tell optimizer the result
        optimizer.tell(x, fitness)
        
        # Track best fitness (nevergrad uses negative fitness)
        if fitness < best_fitness:
            best_fitness = fitness
    
    # Get the best solution
    best_solution = optimizer.recommend()
    final_fitness = fitness_function_for_nevergrad(*best_solution.value)
    
    
    # Convert best weights back to Individual format
    best_weights = best_solution.value
    w1_size = individual.w1.size
    w2_size = individual.w2.size
    w3_size = individual.w3.size
    
    learned_w1 = best_weights[:w1_size].reshape(individual.w1.shape)
    learned_w2 = best_weights[w1_size:w1_size+w2_size].reshape(individual.w2.shape)
    learned_w3 = best_weights[w1_size+w2_size:w1_size+w2_size+w3_size].reshape(individual.w3.shape)
    
    return Individual(w1=learned_w1, w2=learned_w2, w3=learned_w3)

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



def evaluate_individual(individual: Individual, input_size: int, output_size: int, duration: float = DURATION) -> float:
    # learning stage before final evaluation
    learned_individual = learn_individual_with_nevergrad(individual, num_iterations=10)
    
    # FINAL EVALUATION: Evaluate the learned individual on the full task
    global CURRENT_INDIVIDUAL, HISTORY
    CURRENT_INDIVIDUAL = learned_individual
    HISTORY = []
    
    # Create fresh world instance eval
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
    
    # Calculate final fitness (distance traveled in x direction)
    if len(HISTORY) == 0:
        return 0.0
        
    final_pos = HISTORY[-1]
    initial_pos = (0, 0)
    fitness = (
        (initial_pos[0] - final_pos[0]) ** 2
        + (initial_pos[1] - final_pos[1]) ** 2
    ) ** 0.5  # Euclidean distance

    # Set fitness in learned individual
    learned_individual.fitness = fitness
    
    # Return both fitness and learned individual for Lamarckian evolution
    return fitness, learned_individual
#rank based looking at percentage by rank value
def rank_based_selection(population: List[Individual]) -> Individual:
    # Sort population by fitness
    sorted_pop = sorted(population, key=lambda ind: ind.fitness, reverse=True)
    
    # Calculate rank-based probabilities (higher rank = higher probability)
    ranks = np.arange(len(sorted_pop), 0, -1)
    probabilities = ranks / ranks.sum()
    
    # Select based on rank probabilities
    return np.random.choice(sorted_pop, p=probabilities)

def evolutionary_algorithm(input_size: int, output_size: int, pop_size: int = 100, generations: int = 3):
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
        # Evaluate population with Lamarckian evolution
        fitness_gen = []
        
        for i, ind in enumerate(population):
            # Get both fitness and learned individual
            fitness, learned_ind = evaluate_individual(ind, input_size, output_size)
            
            # lamarckian evolution
            # TLerned weightsbecome the new individual
            population[i] = learned_ind
            fitness_gen.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = Individual(learned_ind.w1.copy(), learned_ind.w2.copy(), learned_ind.w3.copy(), fitness)
                SUCCESS_COUNTER += 1  # Count successful individual
        
        fitness_scores.append(fitness_gen)

        # Adapt mutation step size every ADAPTATION_PERIOD generations
        if gen > 0 and gen % ADAPTATION_PERIOD == 0:
            success_rate = SUCCESS_COUNTER / (pop_size * ADAPTATION_PERIOD)
            
            if success_rate > 0.2:  
                MUTATION_STEP_SIZE /= ADAPTATION_FACTOR
            elif success_rate < 0.2: 
                MUTATION_STEP_SIZE *= ADAPTATION_FACTOR
            # If success_rate == 0.2, keep current mutation step size
            
            # Clamp step size to safe bounds
            MUTATION_STEP_SIZE = max(MIN_MUTATION_STEP_SIZE, min(MAX_MUTATION_STEP_SIZE, MUTATION_STEP_SIZE))

            SUCCESS_COUNTER = 0 
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
    num_generations = len(iterations_scores[0])
    
    # Extract best fitness per generation for each run
    best_fitness_per_run = []
    for run_scores in iterations_scores:
        run_best = [max(generation_scores) for generation_scores in run_scores]
        best_fitness_per_run.append(run_best)
    
    # Convert to numpy array for statistics
    fitness_array = np.array(best_fitness_per_run)
    
    # Calculate statistics across runs
    mean_fitness = np.mean(fitness_array, axis=0)
    std_fitness = np.std(fitness_array, axis=0)
    max_fitness = np.max(fitness_array, axis=0)
    
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


