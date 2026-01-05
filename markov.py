from collections import defaultdict
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def build_markov_chain(items_or_sequences):
    transitions = defaultdict(lambda: defaultdict(int))
    if items_or_sequences:
        # Find first non-empty element to check type
        first_element = None
        for elem in items_or_sequences:
            if elem:  # Non-empty
                first_element = elem
                break
        
        if first_element and isinstance(first_element, list):
            # Multiple sequences: process each separately
            sequences = items_or_sequences
            for sequence in sequences:
                if len(sequence) > 1:  # Only process sequences with at least 2 items
                    # Process transitions within this sequence only
                    for i in range(len(sequence) - 1):
                        current = sequence[i]
                        next_item = sequence[i+1]
                        transitions[current][next_item] += 1
        else:
            # Single sequence (backward compatible)
            items = items_or_sequences
            for i in range(len(items) - 1):
                current = items[i]
                next_item = items[i+1]
                transitions[current][next_item] += 1

    # Convert counts to probabilities
    markov_chain = {}
    for current, next_counts in transitions.items():
        total = sum(next_counts.values())
        markov_chain[current] = {k: v/total for k, v in next_counts.items()}

    return markov_chain

def generate_sequence(chain, start, length=10):
    result = [start]
    current = start
    for _ in range(length-1):
        if current not in chain:
            break
        next_items = list(chain[current].keys())
        probabilities = list(chain[current].values())
        current = random.choices(next_items, probabilities)[0]
        result.append(current)
    return result

def save_markov_chain(chain, filepath="markov_chain.json"):
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chain, f, indent=2)
    print(f"Markov chain saved to: {filepath}")

def load_markov_chain(filepath="markov_chain.json"):
    with open(filepath, "r", encoding="utf-8") as f:
        chain = json.load(f)
    print(f"Markov chain loaded from: {filepath}")
    return chain

def get_transition_matrix(chain, predefined_states=None):
    # Use predefined states if given; else extract from chain
    if predefined_states:
        states = predefined_states
    else:
        states = sorted(set(chain.keys()) | {s for nexts in chain.values() for s in nexts})
    
    state_to_idx = {state: i for i, state in enumerate(states)}
    idx_to_state = {i: state for state, i in state_to_idx.items()}
    
    # Initialize matrix
    matrix = np.zeros((len(states), len(states)))
    
    # Fill matrix
    for current, next_counts in chain.items():
        if current in state_to_idx:
            i = state_to_idx[current]
            for next_state, prob in next_counts.items():
                if next_state in state_to_idx:
                    j = state_to_idx[next_state]
                    matrix[i, j] = prob
    
    return matrix, states, state_to_idx, idx_to_state

def get_top_k_next_citations(chain, current_state, k=5):
    if current_state not in chain:
        return []
    
    # Get all next states with their probabilities
    next_states_with_probs = list(chain[current_state].items())
    
    # Sort by probability (descending)
    # next_states_with_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    return next_states_with_probs[:k]

def calculate_multi_step_matrix(chain, steps=2, predefined_states=None):
    matrix, states, state_to_idx, idx_to_state = get_transition_matrix(chain, predefined_states)
    
    # Compute matrix^n
    result_matrix = matrix.copy()
    for _ in range(steps - 1):
        result_matrix = np.dot(result_matrix, matrix)
    
    return result_matrix, states, state_to_idx, idx_to_state

def print_markov_matrix(chain, predefined_states=None, title="Markov Matrix", heatmap=True, save_path=None, max_rows=None, max_cols=None):
    # Use predefined states if given; else extract from chain
    if predefined_states:
        states = predefined_states
    else:
        states = sorted(set(chain.keys()) | {s for nexts in chain.values() for s in nexts})
    
    state_to_idx = {state: i for i, state in enumerate(states)}
    
    # Initialize matrix
    matrix = np.zeros((len(states), len(states)))
    
    # Fill matrix
    for current, next_counts in chain.items():
        if current in state_to_idx:
            i = state_to_idx[current]
            for next_state, prob in next_counts.items():
                if next_state in state_to_idx:
                    j = state_to_idx[next_state]
                    matrix[i, j] = prob
    
    # Convert to DataFrame
    df = pd.DataFrame(matrix, index=states, columns=states)
    df = df.loc[(df != 0).any(axis=1), (df != 0).any(axis=0)]
    
    # Sort by probability values (combined row and column sums) before limiting
    # This ensures we show states with actual transitions instead of zeros
    # Rows and columns must be in the same order (row i = column i)
    if max_rows is not None or max_cols is not None:
        # Get intersection of row and column indices to ensure alignment
        common_states = df.index.intersection(df.columns)
        # Filter to only common states first
        df = df.loc[common_states, common_states]
        
        # Calculate combined score (row sum + column sum) for each state
        row_sums = df.sum(axis=1)
        col_sums = df.sum(axis=0)
        # Combine row and column sums
        combined_scores = row_sums + col_sums
        # Sort by combined score (descending) - most active states first
        sorted_states = combined_scores.sort_values(ascending=False).index
        # Apply same order to both rows and columns
        df = df.loc[sorted_states, sorted_states]
    
    # Limit rows and columns if specified
    # Keep rows and columns aligned (same states in same order)
    if max_rows is not None or max_cols is not None:
        # Use the minimum of max_rows and max_cols to keep them aligned
        # If only one is specified, use that value for both
        limit = max_rows if max_cols is None else (max_cols if max_rows is None else min(max_rows, max_cols))
        df = df.iloc[:limit, :limit]
    
    print(f"\n=== {title} ===")
    print(df)
    
    # Optional heatmap visualization
    if heatmap:
        plt.figure(figsize=(8,6))
        sns.heatmap(df, annot=True, cmap="Blues")
        plt.title(title)
        if save_path:
            plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
            print(f"Heatmap saved as: {save_path}")
        plt.show()

def print_multi_step_matrix(chain, steps=2, predefined_states=None, title=None, heatmap=False, save_path=None):
    if title is None:
        title = f"Markov Matrix M^{steps}"
    
    matrix, states, state_to_idx, idx_to_state = calculate_multi_step_matrix(chain, steps, predefined_states)
    
    # Convert to DataFrame
    df = pd.DataFrame(matrix, index=states, columns=states)
    print(f"\n=== {title} ===")
    print(df)
    
    # Optional heatmap visualization
    if heatmap:
        plt.figure(figsize=(8,6))
        sns.heatmap(df, annot=True, cmap="Blues", fmt='.4f')
        plt.title(title)
        if save_path:
            plt.savefig(save_path, format='jpg', dpi=300, bbox_inches='tight')
            print(f"Heatmap saved as: {save_path}")
        plt.show()

# ------------------------
#  Functions
# ------------------------
def feature1_top_k_citations(chain, state, k=5):
    print("=== Feature 1: Top k Next Citations with Confidence Rates ===")
    print(f"Current state: {state}")
    
    if state not in chain:
        print(f"  Warning: State '{state}' not found in Markov chain.")
        print("  Available states:", list(chain.keys()))
        return
    
    top_predictions = get_top_k_next_citations(chain, state, k=k)
    print(f"Top {k} next citations:")
    if top_predictions:
        for next_state, confidence in top_predictions:
            print(f"  {next_state}: {confidence:.4f} ({confidence*100:.2f}%)")
    else:
        print("  No transitions available from this state")
    print()

def feature2_generate_sequences(chain, start_state, num_sequences=3, length=20):
    print("=== Feature 2: Generate Sequences ===")
    print(f"Generating {num_sequences} sequences starting from '{start_state}' with length {length}:")
    for i in range(num_sequences):
        generated_sequence = generate_sequence(chain, start=start_state, length=length)
        print(f"Sequence {i+1}: {generated_sequence}\n")

def feature3_multi_step_matrices(chain, predefined_states=None):
    print("=== Feature 3: Multi-step Transition Matrices ===")
    
    # Print original matrix (M^1)
    print_markov_matrix(
        chain, 
        predefined_states=predefined_states, 
        title="Original Transition Matrix M^1"
    )
    
    # Print M^2 (2-step transition matrix)
    print_multi_step_matrix(
        chain, 
        steps=2, 
        predefined_states=predefined_states, 
        title="2-Step Transition Matrix M^2",
        heatmap=True,
        save_path="types_2step_heatmap.jpg"
    )
    
    # Print M^3 (3-step transition matrix)
    print_multi_step_matrix(
        chain, 
        steps=3, 
        predefined_states=predefined_states, 
        title="3-Step Transition Matrix M^3",
        heatmap=True,
        save_path="types_3step_heatmap.jpg"
    )

# ------------------------
# Main execution
# ------------------------
def main():
    # Flag to control whether to build new markov chain or load existing one
    mIsNew = 0  # Set to 1 to build new chain, 0 to load existing chain
    
    # Load saved sequences
    with open("title_sequence.json", "r") as f:
        titles_sequences = json.load(f)
    with open("types_sequence.json", "r") as f:
        types_sequences = json.load(f)
    
    # Process sequences (handle both nested lists and flat lists)
    if titles_sequences and isinstance(titles_sequences[0], list):
        titles_flat = [item for seq in titles_sequences for item in seq]
    else:
        titles_flat = titles_sequences
    
    if types_sequences and isinstance(types_sequences[0], list):
        types_flat = [item for seq in types_sequences for item in seq]
    else:
        types_flat = types_sequences
    
    # Build or load Markov chains
    if mIsNew == 1:
        print("=== Building new Markov chains ===")
        # Build chains - pass sequences as-is (build_markov_chain handles both formats)
        titles_mc = build_markov_chain(titles_sequences)
        types_mc = build_markov_chain(types_sequences)
        # Save both chains
        save_markov_chain(titles_mc, "title_markov_chain.json")
        save_markov_chain(types_mc, "type_markov_chain.json")
    else:
        print("=== Loading existing Markov chains ===")
        # Load both chains
        titles_mc = load_markov_chain("title_markov_chain.json")
        types_mc = load_markov_chain("type_markov_chain.json")
    
    # Step 1: Print transition matrices for both title and type
    print("\n" + "="*70)
    print("STEP 1: Transition Matrices")
    print("="*70)
    print_markov_matrix(titles_mc, title="Title Transition Matrix M^1", max_rows=10, max_cols=10, save_path="titles_heatmap.jpg")
    print_markov_matrix(types_mc, predefined_states=["Constitution", "Statute", "Jurisprudence", "Administrative rule"], 
                        title="Type Transition Matrix M^1", save_path="types_heatmap.jpg")
    
    # Step 2: Feature 1 example with title markov chain
    print("\n" + "="*70)
    print("STEP 2: Feature 1 - Top k Citations (Title Markov Chain)")
    print("="*70)
    if titles_flat:
        feature1_top_k_citations(titles_mc, state="R51S3", k=2)
    
    # Step 3: Feature 2 example with type markov chain
    print("\n" + "="*70)
    print("STEP 3: Feature 2 - Generate Sequences (Type Markov Chain)")
    print("="*70)
    if types_flat:
        feature2_generate_sequences(types_mc, start_state=types_flat[0], num_sequences=3, length=20)
    
    # Step 4: Feature 3 for both title and type
    print("\n" + "="*70)
    print("STEP 4: Feature 3 - Multi-step Transition Matrices")
    print("="*70)
    all_types = ["Constitution", "Statute", "Jurisprudence", "Administrative rule"]
    feature3_multi_step_matrices(types_mc, predefined_states=all_types)

if __name__ == "__main__":
    main()
