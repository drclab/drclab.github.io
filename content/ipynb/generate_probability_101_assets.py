import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set output directory
output_dir = os.path.join(os.getcwd(), "static/images/probability-101")
os.makedirs(output_dir, exist_ok=True)

def monty_hall(switch):
    # All doors have a goat initially
    doors = np.array([0, 0, 0])
    # Randomly decide which door will have a car
    winner_index = np.random.randint(0, 3)
    # Place the car in the winner door
    doors[winner_index] = 1
    # Participant selects a door at random
    choice = np.random.randint(0, 3)
    # Get doors that can be opened (host cannot open the door chosen or the one with the car)
    openable_doors = [i for i in range(3) if i not in (winner_index, choice)]
    # Host opens one of the available doors at random
    door_to_open = np.random.choice(openable_doors)
    # Switch to the other available door
    if switch:
        choice = [i for i in range(3) if i not in (choice, door_to_open)][0]
    # Return 1 if you open a door with a car, 0 otherwise
    return doors[choice]

def generalized_monty_hall(switch, n=3, k=1):
    if not (0 <= k <= n-2):
        raise ValueError('k must be between 0 and n-2')
    # All doors have a goat initially
    doors = np.zeros(n)
    # Decide which door will have a car
    winner = np.random.randint(0, n)
    # Place the car in the winner door
    doors[winner] = 1.0
    # Participant selects a door at random
    choice = np.random.randint(0, n)
    # Get doors that can be opened
    openable_doors = [i for i in range(n) if i not in (winner, choice)]
    # Host open k of the available doors at random
    door_to_open = np.random.choice(openable_doors, size=k, replace=False)
    # Switch to the other available door
    if switch:
        choices = [i for i in range(n) if i != choice and i not in door_to_open]
        choice = np.random.choice(choices)
    # Return 1 if you open a door with a car, 0 otherwise
    return doors[choice]

def save_pie_chart(switch, n_iterations, filename, generalized=False, n=3, k=1):
    wins = 0
    for _ in range(n_iterations):
        if generalized:
            wins += generalized_monty_hall(switch=switch, n=n, k=k)
        else:
            wins += monty_hall(switch=switch)
    
    win_rate = wins / n_iterations
    loss_rate = 1 - win_rate

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.pie(
        [win_rate, loss_rate],
        labels=["Win a car", "Win a goat"],
        colors=sns.color_palette("pastel")[2:4],
        autopct="%.0f%%",
        startangle=90
    )
    
    msg = "always" if switch else "never"
    title = f"Win rate if you {msg} switch doors\n({n_iterations} simulations)"
    if generalized:
        title += f" (n={n}, k={k})"
    
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Saved {filename}")

if __name__ == "__main__":
    np.random.seed(42) # For reproducibility
    
    # Standard Monty Hall
    save_pie_chart(switch=True, n_iterations=1000, filename="monty_hall_switch.png")
    save_pie_chart(switch=False, n_iterations=1000, filename="monty_hall_no_switch.png")
    
    # Generalized Monty Hall (n=10, k=8)
    # Host leaves only 1 door to switch to
    save_pie_chart(switch=True, n_iterations=1000, filename="generalized_monty_hall_switch.png", generalized=True, n=10, k=8)
    save_pie_chart(switch=False, n_iterations=1000, filename="generalized_monty_hall_no_switch.png", generalized=True, n=10, k=8)
