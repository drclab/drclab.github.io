import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create directory for images if it doesn't exist
output_dir = "/home/cjduan/drclab.github.io/static/images/probability-103"
os.makedirs(output_dir, exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

# Define die
n_sides = 6
dice = np.array([i for i in range(1, n_sides+1)])

def save_plot(filename):
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()

# 1. Histogram of 20 rolls
n_rolls = 20
rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
n_rolls_hist = sns.histplot(rolls, discrete=True)
n_rolls_hist.set(title=f"Histogram of {n_rolls} rolls")
save_plot("histogram_20_rolls.png")

# 2. Histogram of 20,000 rolls
n_rolls = 20_000
rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
n_rolls_hist = sns.histplot(rolls, discrete=True)
n_rolls_hist.set(title=f"Histogram of {n_rolls} rolls")
save_plot("histogram_20000_rolls.png")

# 3. Histogram of sum of rolling twice (20,000 simulations)
first_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
second_rolls = np.array([np.random.choice(dice) for _ in range(n_rolls)])
sum_of_rolls = first_rolls + second_rolls
sum_2_rolls_hist = sns.histplot(sum_of_rolls, stat="probability", discrete=True)
sum_2_rolls_hist.set(title=f"Histogram of {n_rolls} rolls (sum of rolling twice)")
save_plot("histogram_sum_2_rolls.png")

# 4. Probabilities for a fair die
probs_fair_dice = np.array([1/n_sides]*n_sides)
fair_dice_plot = sns.barplot(x=dice, y=probs_fair_dice)
fair_dice_plot.set(title=f"Probabilities for fair die with {n_sides} sides", ylabel="Probability")
fair_dice_plot.set_ylim(0, 0.5)
save_plot("probs_fair_dice.png")

# 5. Probabilities for a loaded die (side 2 is twice as likely)
def load_dice(n_sides, loaded_number):
    probs = np.array([1/(n_sides+1) for _ in range(n_sides)])
    probs[loaded_number-1] = 1 - sum(probs[:-1])
    return probs

probs_loaded_dice = load_dice(n_sides, loaded_number=2)
loaded_dice_plot = sns.barplot(x=dice, y=probs_loaded_dice)
loaded_dice_plot.set(title=f"Probabilities for loaded die with {n_sides} sides", ylabel="Probability")
loaded_dice_plot.set_ylim(0, 0.5)
save_plot("probs_loaded_dice.png")

# 6. Histogram of loaded die (20,000 rolls)
rolls_loaded = np.random.choice(dice, size=n_rolls, p=probs_loaded_dice)
loaded_hist = sns.histplot(rolls_loaded, discrete=True)
loaded_hist.set(title=f"Histogram of {n_rolls} rolls (Loaded Die)")
save_plot("histogram_loaded_20000_rolls.png")

print(f"Assets generated in {output_dir}")
