import os
import sys
import time
import random
import numpy as np
import copy
from sklearn.ensemble import RandomForestRegressor # Keep only the model actually used
from scipy.stats import norm
import math
import argparse # Import argparse for command-line arguments

# Import your utility class
from util import Util, cbench # Assuming cbench list is useful here

# --- BOCA Configuration ---
random.seed(456)
iters = 60           # Total optimization iterations
begin2end = 5        # Number of independent runs for statistical analysis
# md = int(os.environ.get('MODEL', 1)) # Seems unused if only RF is used
fnum = int(os.environ.get('FNUM', 8)) # Feature importance selection count
decay = float(os.environ.get('DECAY', 0.5))   # Parameters for rnum calculation (exploration decay)
scale = float(os.environ.get('SCALE', 10))
offset = float(os.environ.get('OFFSET', 20))
rnum0 = int(os.environ.get('RNUM', 2 ** 8)) # Initial neighborhood size factor

# --- Instantiate Utility and Get LLVM Flags ---
util = Util()
# Use the LLVM flags defined in your Util class
options = util.gain_flags() # Or util.gcc_flags if that's the intended list name
n_flags = len(options)
print(f"Loaded {n_flags} LLVM flags for tuning.")

# --- Remove Hardcoded GCC Commands ---
# cmd2, cmd3, cmd4, cmd5 are no longer needed as util.py handles this.

# --- Helper Functions (Mostly Unchanged, but check dependencies) ---

# generate_opts is likely NOT needed if run_procedure takes the binary vector directly
# def generate_opts(independent):
#     result = []
#     for k, s in enumerate(independent):
#         if s == 1:
#             result.append(options[k])
#     return result # Return the list of flag strings

# --- Objective Function (MAJOR CHANGE) ---
def get_objective_score(independent, program_name):
    """
    Evaluates a given flag configuration (binary vector) for a specific program.
    Uses util.run_procedure to handle compilation, execution, and speedup calculation.

    Args:
        independent (list or np.array): Binary vector (0 or 1) representing flag selection.
        program_name (str): The name of the Cbench program to evaluate (e.g., 'automotive_susan_c').

    Returns:
        float: The negative speedup compared to the baseline (O3 in util.run_procedure).
               BOCA will try to minimize this value.
    """
    # Ensure 'independent' is a list of 0s and 1s if needed by run_procedure,
    # or keep as numpy array if that's okay. Let's assume list of ints.
    eval_start_time = time.time() # <
    flags_binary_vector = list(map(int, independent)) # Ensure correct type if necessary

    print(f"\nEvaluating config for {program_name}...")
    # print(f"Binary flags: {flags_binary_vector}") # Optional: Debug print

    # Call the utility function which handles everything:
    # update Makefile, make clean, make, run, get time, compare to O3, return -speedup
    try:
        # Using run_procedure which calculates speedup vs O3 and returns negative speedup
        negative_speedup = util.run_procedure(program_name, flags_binary_vector)

        # Handle potential errors if run_procedure doesn't raise them
        if negative_speedup is None or not isinstance(negative_speedup, (int, float)):
             print(f"Warning: Evaluation failed for {program_name}. Returning poor score.")
             return 0.0 # Return a very bad score (0 speedup = infinitely bad for minimization)

        print(f"Evaluation result (negative speedup) for {program_name}: {negative_speedup}")
        return negative_speedup

    except Exception as e:
        print(f"Error during evaluation for {program_name}: {e}")
        # Return a very poor score to discourage BOCA from selecting this again
        return 0.0 # Or a large positive number if minimizing positive speedup
    finally: # <--- 添加：确保计时结束
        eval_end_time = time.time()
        eval_duration = eval_end_time - eval_start_time
        print(f"    [计时] 本次 get_objective_score 耗时: {eval_duration:.2f} 秒")

# --- Configuration Generation (Unchanged) ---
def generate_conf(x):
    # Converts integer x to a binary vector of length n_flags
    comb = bin(x).replace('0b', '')
    comb = '0' * (n_flags - len(comb)) + comb # Use n_flags dynamically
    conf = []
    for k, s in enumerate(comb):
        if s == '1':
            conf.append(1)
        else:
            conf.append(0)
    return conf

# --- Search Algorithm Internals (Mostly Unchanged) ---
# These rely on the objective function but their logic remains the same.

class get_exchange(object):
    def __init__(self, incumbent):
        self.incumbent = incumbent # List of (index, value=0 or 1) for important features

    def to_next(self, feature_id): # feature_id = random subset of *all* features to flip/set to 1?
        ans = [0] * n_flags # Use dynamic n_flags
        # This part seems to randomly set some features to 1? Revisit BOCA paper if needed.
        for f in feature_id:
            ans[f] = 1
        # Override with the incumbent values for the 'important' features
        for f in self.incumbent:
            ans[f[0]] = f[1]
        return ans

def do_search(train_indep, model, eta, rnum):
    # Uses the trained model to explore the neighborhood based on feature importance
    try:
        features = model.feature_importances_
    except AttributeError:
        print("Warning: Model does not have feature_importances_. Using equal importance.")
        features = np.ones(n_flags) / n_flags

    print('Feature importances:')
    print(features)

    b = time.time()
    # Select top 'fnum' features based on importance
    feature_sort = [[i, x] for i, x in enumerate(features)]
    # Handle case where n_flags < fnum
    current_fnum = min(fnum, n_flags)
    feature_selected = sorted(feature_sort, key=lambda x: x[1], reverse=True)[:current_fnum]
    # Get indices of all features
    feature_ids = list(range(n_flags)) # More robust way to get all indices

    neighborhood_iterators = []
    # Generate all 2^fnum combinations for the *important* features
    for i in range(2 ** current_fnum):
        comb = bin(i).replace('0b', '')
        comb = '0' * (current_fnum - len(comb)) + comb
        inc = [] # Incumbent settings for important features for this neighborhood center
        for k, s in enumerate(comb):
            if s == '1':
                inc.append((feature_selected[k][0], 1))
            else:
                inc.append((feature_selected[k][0], 0))
        neighborhood_iterators.append(get_exchange(inc))
    print(f'Time to setup {len(neighborhood_iterators)} neighborhood centers: {time.time() - b:.2f}s')

    s = time.time()
    neighbors = [] # List to store generated neighbor configurations (binary vectors)
    print(f'Generating neighbors with rnum: {rnum:.2f}')
    # For each neighborhood center...
    for i, neighborhood_center_configurator in enumerate(neighborhood_iterators):
        # Generate rnum random variations around it
        # The range(1 + int(rnum)) seems odd, maybe just int(rnum)? Check BOCA logic.
        # Let's assume it means generate roughly 'rnum' neighbors per center.
        for _ in range(max(1, int(rnum))): # Ensure at least 1 neighbor per center
            # Randomly select *some other* features (non-important ones)
            # to potentially flip/change from the base 'inc' configuration.
            num_random_features = random.randint(0, n_flags - current_fnum) # Sample from non-important ones? Logic unclear.
            # Original code samples from *all* feature_ids, which might override incumbent. Let's stick to original for now.
            selected_feature_ids = random.sample(feature_ids, random.randint(0, n_flags))

            # Generate the neighbor config using the center and random flips
            n = neighborhood_center_configurator.to_next(selected_feature_ids)
            neighbors.append(n)

    print(f'Generated {len(neighbors)} neighbors.')
    print(f'Time to generate neighbors: {time.time()-s:.2f}s')

    # --- Predict performance of neighbors and calculate EI ---
    if not neighbors:
      print("Warning: No neighbors generated.")
      return []

    pred = [] # List of predictions from each tree in the forest
    estimators = model.estimators_
    s = time.time()
    neighbors_np = np.array(neighbors)
    try:
        for e in estimators:
            pred.append(e.predict(neighbors_np))
        # Calculate Expected Improvement (EI) using predictions
        acq_val_incumbent = get_ei(pred, eta) # eta is the best score found so far
        print(f'Time for prediction and EI calculation: {time.time()-s:.2f}s')
        # Return list of [neighbor_config, ei_score]
        return [[neighbor_config, ei_score] for ei_score, neighbor_config in zip(acq_val_incumbent, neighbors)]
    except Exception as e_pred:
        print(f"Error during prediction/EI: {e_pred}")
        # Return empty or some default if prediction fails
        return [[n, 0.0] for n in neighbors] # Assign 0 EI if prediction fails


def get_ei(pred, eta):
    # Calculates Expected Improvement (EI)
    pred = np.array(pred).transpose(1, 0) # Shape: (n_neighbors, n_estimators)
    m = np.mean(pred, axis=1) # Mean prediction for each neighbor
    s = np.std(pred, axis=1)  # Std dev prediction for each neighbor

    # Handle cases where std dev is zero to avoid division by zero
    s_copy = np.copy(s)
    s[s_copy == 0.0] = 1e-9 # Replace 0 std dev with a tiny number

    # Standard EI calculation
    z = (eta - m) / s
    ei = (eta - m) * norm.cdf(z) + s * norm.pdf(z)

    # Set EI to 0 where original std dev was 0 (no uncertainty)
    ei[s_copy == 0.0] = 0.0

    return ei

def get_nd_solutions(train_indep, training_dep, eta, rnum):
    # Fits the RF model and calls do_search to find the next best candidate
    print("Fitting RandomForestRegressor...")
    fit_start_time = time.time()
    model = RandomForestRegressor(random_state=456) # Use random_state for reproducibility
    model.fit(np.array(train_indep), np.array(training_dep))
    fit_end_time = time.time() # <--- 添加：训练结束计时
    fit_duration = fit_end_time - fit_start_time
    print(f"    [计时] 模型训练耗时: {fit_duration:.2f} 秒")

    # Do the search based on the current model and best score (eta)
    print(f"Searching for next configuration with eta = {eta:.4f} and rnum = {rnum:.2f}")
    begin_search = time.time()
    # merged_predicted_objectives format: [[config_vector, ei_score], ...]
    merged_predicted_objectives = do_search(train_indep, model, eta, rnum)

    if not merged_predicted_objectives:
        print("Warning: Search returned no candidates. Returning random.")
        # Fallback: return a random configuration not yet tried
        while True:
            x = random.randint(0, 2 ** n_flags)
            conf = generate_conf(x)
            if conf not in train_indep:
                return conf, 0.0 # Return random conf with 0 EI score

    # Sort candidates by EI score (higher is better)
    merged_predicted_objectives = sorted(merged_predicted_objectives, key=lambda x: x[1], reverse=True)
    end_search = time.time()
    print(f'Search time: {end_search - begin_search:.2f}s')

    # Find the best candidate (highest EI) that hasn't been evaluated yet
    begin_select = time.time()
    for config_vector, ei_score in merged_predicted_objectives:
        # Convert numpy array/list to tuple for set membership check if needed
        # Or ensure train_indep stores elements consistently (e.g., all lists)
        # Check if this exact configuration vector has been seen before.
        # Need efficient lookup - convert train_indep to set of tuples?
        # For now, simple list comparison (might be slow for large train_indep)
        is_new = True
        for existing_config in train_indep:
            if np.array_equal(config_vector, existing_config):
                is_new = False
                break
        if is_new:
            print(f'Selected new configuration with EI: {ei_score:.4f}. Time: {time.time() - begin_select:.2f}s')
            return config_vector, ei_score # Return the config vector and its EI score

    # Fallback if all high-EI candidates were already evaluated (should be rare)
    print("Warning: All top candidates already evaluated. Returning random unseen.")
    while True:
        x = random.randint(0, 2 ** n_flags)
        conf = generate_conf(x)
        is_new = True
        for existing_config in train_indep:
             if np.array_equal(conf, existing_config):
                 is_new = False
                 break
        if is_new:
             return conf, 0.0 # Return random conf with 0 EI score


def get_training_sequence(training_indep, training_dep, testing_indep, rnum):
    # Wrapper function, testing_indep seems to be 'eta' (best score so far)
    eta = testing_indep
    return_nd_independent, predicted_objectives = get_nd_solutions(training_indep, training_dep, eta, rnum)
    return return_nd_independent, predicted_objectives

# --- Main Optimization Loop ---
def run_boca(program_name):
    """Runs the BOCA optimization process for a given program."""
    training_indep = [] # List to store evaluated configurations (binary vectors)
    training_dep = []   # List to store corresponding objective scores (-speedup)
    ts = []             # List to store timestamps of evaluations
    initial_sample_size = 10 # Increase initial random samples for better model start
    b = time.time()

    # --- Calculate Sigma for rnum decay ---
    # Ensure decay is < 1 to avoid log(negative)
    safe_decay = min(decay, 0.999)
    if safe_decay <= 0:
        print("Warning: Decay must be positive. Setting to 0.5")
        safe_decay = 0.5
    sigma_squared = -scale ** 2 / (2 * math.log(safe_decay)) if safe_decay < 1 else float('inf')
    print(f"Calculated sigma^2 for rnum decay: {sigma_squared:.2f}")


    # --- Initial Random Sampling ---
    print(f"Generating {initial_sample_size} initial random samples...")
    while len(training_indep) < initial_sample_size:
        x = random.randint(0, 2 ** n_flags)
        conf = generate_conf(x)
        # Ensure configuration is new before evaluating
        is_new = True
        for existing_config in training_indep:
             if np.array_equal(conf, existing_config):
                 is_new = False
                 break
        if is_new:
            score = get_objective_score(conf, program_name)
            # Only add if evaluation was successful (score is not None/bad)
            if score is not None and score != 0.0: # Assuming 0.0 is the bad score marker
                 training_indep.append(conf)
                 training_dep.append(score)
                 ts.append(time.time() - b)
                 print(f"Initial sample {len(training_indep)}/{initial_sample_size} added. Score: {score:.4f}")
            else:
                 print(f"Skipping failed initial sample.")
        # Avoid infinite loop if evaluations consistently fail
        if time.time() - b > 3600 and len(training_indep) == 0: # Timeout after 1 hour if no success
            print("Error: Failed to get any successful initial samples after 1 hour.")
            return [], [] # Return empty results

    # Check if initial sampling yielded any results
    if not training_dep:
        print("Error: Initial sampling failed to produce any valid results.")
        return [], []

    # --- Main Optimization Loop ---
    steps = 0
    budget = iters # Total number of *evaluations* (including initial)
    result = min(training_dep) # Best score found so far (minimum -speedup)
    print(f"\nStarting BOCA optimization loop. Initial best score: {result:.4f}")

    # Loop until budget is reached (evaluated configs = budget)
    while len(training_indep) < budget:
        iter_start_time = time.time()
        steps += 1
        # print(f"\n--- Iteration {len(training_indep)}/{budget} ---")
        print(f"\n--- 迭代 {len(training_indep) + 1}/{budget} ---") # BOCA

        # Calculate adaptive neighborhood size (rnum)
        if sigma_squared > 0 and not math.isinf(sigma_squared):
             exponent = -max(0, len(training_indep) - offset) ** 2 / (2 * sigma_squared)
             rnum = rnum0 * math.exp(exponent)
        else: # Handle infinite or non-positive sigma_squared
             rnum = rnum0 if len(training_indep) <= offset else rnum0 * decay**(len(training_indep)-offset) # Alternative decay

        # Get the next configuration to evaluate using the model and EI
        # 'result' (best score so far) is passed as eta
        best_solution_vector, ei_score = get_training_sequence(training_indep, training_dep, result, rnum)

        print(f"Selected solution with predicted EI: {ei_score:.4f}")
        # print(f"Solution vector: {best_solution_vector}") # Optional debug

        # Evaluate the selected configuration
        current_time = time.time()
        best_result_score = get_objective_score(best_solution_vector, program_name)
        ts.append(current_time - b)

        # Add the evaluated configuration and its score
        # Only add if evaluation was successful
        if best_result_score is not None and best_result_score != 0.0:
             training_indep.append(best_solution_vector)
             training_dep.append(best_result_score)

             # Update the best score found so far
             if best_result_score < result:
                 result = best_result_score
                 print(f"*** New best score found: {result:.4f} ***")
             else:
                 print(f"Current score: {best_result_score:.4f}, Best score: {result:.4f}")
        else:
            print(f"Evaluation failed for selected solution. Skipping add.")
            # Optionally add placeholder or handle differently? For now, just skip.
        iter_end_time = time.time() # <--- 添加：迭代结束计时
        iter_duration = iter_end_time - iter_start_time
        print(f"    [计时] 本次迭代耗时: {iter_duration:.2f} 秒")

    print("\nOptimization finished.")
    return training_dep, ts # Return list of scores and timestamps

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Tune LLVM flags for a Cbench program using BOCA.")
    # Make program selection mandatory
    parser.add_argument("program_name", help="Name of the Cbench program to tune (e.g., automotive_susan_c).", choices=cbench) # Use cbench list from util
    args = parser.parse_args()

    program_to_tune = args.program_name
    print(f"=== Starting BOCA tuning for: {program_to_tune} ===")
    init_time1 = time.time() # <--- 添加：初始化计时
    # --- Run Multiple Times for Stats ---
    all_stats = [] # Stores the list of scores (-speedup) for each run
    all_times = [] # Stores the list of timestamps for each run

    for i in range(begin2end):
        print(f"\n--- Run {i+1}/{begin2end} ---")
        run_dep, run_ts = run_boca(program_to_tune)
        if run_dep and run_ts: # Only append if the run was successful
            print(f'Run {i+1} final scores: {run_dep}')
            all_stats.append(run_dep)
            all_times.append(run_ts)
        else:
            print(f"Run {i+1} failed or produced no results.")

    # --- Process and Print Results ---
    if not all_stats:
        print("\nNo successful runs completed. Exiting.")
        sys.exit(1)
    end_time1 = time.time() # <--- 添加：结束计时
    print(f"\n[计时] 总耗时: {end_time1 - init_time1:.2f} 秒")

    print("\n\n=== Final Results ===")
    print(f"Target Program: {program_to_tune}")
    print(f"Total Iterations per run: {iters}")
    print(f"Number of runs: {len(all_stats)}")

    # Calculate running best score for each run
    processed_stats = []
    for run_data in all_stats:
        best_so_far = float('inf')
        running_best = []
        for score in run_data:
            best_so_far = min(best_so_far, score)
            running_best.append(best_so_far)
        processed_stats.append(running_best)

    # Ensure all runs have the same length for averaging (pad if necessary, though ideally they finish)
    max_len = max(len(run) for run in processed_stats) if processed_stats else 0
    padded_stats = [run + [run[-1]] * (max_len - len(run)) for run in processed_stats if run] # Pad with last best score

    if padded_stats:
        # --- Average Best Score (-Speedup) Over Iterations ---
        avg_best_scores = -np.mean(padded_stats, axis=0) # Average negative speedup, then negate for avg speedup
        print("\nAverage Best Speedup Found Over Iterations:")
        print(list(np.round(avg_best_scores, 4)))

        # --- Standard Deviation of Best Score (-Speedup) Over Iterations ---
        std_best_scores = np.std(padded_stats, axis=0)
        print("\nStd Dev of Best (-Speedup) Found Over Iterations:")
        print(list(np.round(std_best_scores, 4)))

        # --- Average Time Taken Over Iterations ---
        # Time padding might be less meaningful, maybe just report avg time for runs?
        # For now, let's average the timestamps recorded
        max_time_len = max(len(run) for run in all_times) if all_times else 0
        padded_times = [run + [run[-1]] * (max_time_len - len(run)) for run in all_times if run]
        if padded_times:
            avg_times = np.mean(padded_times, axis=0)
            print("\nAverage Cumulative Time (s) Over Iterations:")
            print(list(np.round(avg_times, 2)))
    else:
        print("\nNo valid data to compute statistics.")

    print("\n=== Tuning Complete ===")