import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

# Set non-interactive backend
matplotlib.use('Agg')

# Monkeypatch plt.show to avoid trying to display plots
plt.show = lambda: None

# Add content/ipynb to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ipynb')))
import utils

# Simulation and problem functions adapted from the notebook
def simulate(problem_func, n_students=365, n_simulations=1000):
    matches = 0
    for _ in range(n_simulations):
        if problem_func(n_students):
            matches += 1
    return matches/n_simulations

def problem_1(n_students):
    predef_bday = np.random.randint(0, 365)
    gen_bdays = np.random.randint(0, 365, (n_students))
    return predef_bday in gen_bdays

def problem_2(n_students):
    gen_bdays = np.random.randint(0, 365, (n_students))
    rnd_index = np.random.randint(0, len(gen_bdays))
    rnd_bday = gen_bdays[rnd_index]
    remaining_bdays = np.delete(gen_bdays, rnd_index, axis=0)
    return rnd_bday in remaining_bdays

def problem_3(n_students):
    gen_bdays = np.random.randint(0, 365, (n_students))
    return len(np.unique(gen_bdays)) != len(gen_bdays)

def problem_4(n_students):
    # Generate birthdays for every student in classroom 1
    gen_bdays_1 = np.random.randint(0, 365, (n_students))
    # Generate birthdays for every student in classroom 2
    gen_bdays_2 = np.random.randint(0, 365, (n_students))
    # Check for any match between both classrooms
    return np.isin(gen_bdays_1, gen_bdays_2).any()

# Ensure output directory exists
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'static', 'images', 'probability-102'))
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def run_and_save(prob_func, classroom_sizes, filename):
    print(f"Generating {filename}...")
    sim_probs = [simulate(prob_func, n_students=n) for n in classroom_sizes]
    utils.plot_simulated_probs(sim_probs, classroom_sizes)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Problem 1 & 2 use big_classroom_sizes
    run_and_save(problem_1, utils.big_classroom_sizes, 'prob_1_sim.png')
    run_and_save(problem_2, utils.big_classroom_sizes, 'prob_2_sim.png')
    
    # Problem 3 & 4 use small_classroom_sizes
    run_and_save(problem_3, utils.small_classroom_sizes, 'prob_3_sim.png')
    run_and_save(problem_4, utils.small_classroom_sizes, 'prob_4_sim.png')
    
    print(f"All assets generated successfully in {output_dir}")
