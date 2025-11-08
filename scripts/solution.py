import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import argparse

print("--- Manager Prediction using Hybrid Scoring (Embeddings + Graph Features) ---")

## --- 1. CONFIGURATION: The Weights ---
WEIGHT_EMBEDDING_SIMILARITY = 1.0
WEIGHT_COMMON_NEIGHBORS = 1.0
WEIGHT_SENIORITY_GAP = 1.0
WEIGHT_LOCATION_MATCH = 0.0

# --- 2. DATA LOADING ---
def load_data(employees_path, connections_path):
    """Loads employee and connection data."""
    print("Step 1: Loading data...")
    try:
        employees_df = pd.read_csv(employees_path, engine='python')
        connections_df = pd.read_csv(connections_path)
        print(f"Loaded {len(employees_df)} employees and {len(connections_df)} connections.")
        return employees_df, connections_df
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure required CSV files are present.")
        return None, None

# --- 3. FEATURE ENGINEERING & GRAPH CONSTRUCTION ---
def build_graph_with_features(employees_df, connections_df, model=None):
    """Builds the graph and enriches it with all necessary node attributes."""
    print("Step 2: Engineering features and building graph...")
    
    print("   - Generating text embeddings...")
    
    employees_df['combined_text'] = employees_df['job_title_current'].fillna('') + ". " + employees_df['profile_summary'].fillna('')
    
    # Use preloaded model if provided, otherwise load it here
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

    # Batch process all embeddings at once - much faster than encoding one by one
    embeddings = model.encode(employees_df['combined_text'].tolist(), show_progress_bar=True)
    
    # Map employee IDs to their embeddings
    embedding_dict = {emp_id: emb for emp_id, emb in zip(employees_df['employee_id'], embeddings)}

    def get_seniority(title):
        """Extract seniority level from job title using regex patterns."""
        title = str(title).lower()
        # Return as soon as we find a match - no need to check remaining patterns
        if re.search(r'\b(chief|ceo)\b', title): return 7
        if re.search(r'\b(vp|vice president)\b', title): return 6
        if re.search(r'\b(director|head)\b', title): return 5
        if re.search(r'\b(manager|lead)\b', title): return 4
        if re.search(r'\b(senior|principal|sr\.)\b', title): return 3
        if re.search(r'\b(junior|entry|associate)\b', title): return 1
        return 2

    employees_df['seniority_score'] = employees_df['job_title_current'].apply(get_seniority)

    print("   - Constructing NetworkX graph...")
    G = nx.Graph()
    
    node_attributes = employees_df.set_index('employee_id').to_dict('index')

    for node_id, attrs in node_attributes.items():
        attrs['embedding'] = embedding_dict.get(node_id)

    # nx.set_node_attributes automatically creates nodes, so no need to add them manually first
    nx.set_node_attributes(G, node_attributes)
    G.add_edges_from(connections_df.values)

    print(f"   - Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# --- 4. THE INFERENCE ALGORITHM ---
def score_potential_managers(employee_id, G):
    employee_attrs = G.nodes[employee_id]
    employee_seniority = employee_attrs.get('seniority_score', 0)
    employee_embedding = employee_attrs.get('embedding')

    if employee_seniority >= 7:
        return []

    neighbors = list(G.neighbors(employee_id))
    if not neighbors:
        return []

    candidates = [n_id for n_id in neighbors if G.nodes[n_id].get('seniority_score', 0) > employee_seniority]
    if not candidates:
        candidates = neighbors

    scored_candidates = []
    for cand_id in candidates:
        cand_attrs = G.nodes[cand_id]
        score = 0

        cand_embedding = cand_attrs.get('embedding')
        if employee_embedding is not None and cand_embedding is not None:
            # <-- Inefficient: recomputing same reshape every time
            similarity = cosine_similarity(
                np.array(employee_embedding).reshape(1, -1),
                np.array(cand_embedding).reshape(1, -1)
            )[0][0]
            score += similarity * WEIGHT_EMBEDDING_SIMILARITY

        # <-- Inefficient: repeated list conversion inside loop
        common_neighbors = len(list(nx.common_neighbors(G, employee_id, cand_id)))
        score += common_neighbors * WEIGHT_COMMON_NEIGHBORS

        seniority_gap = cand_attrs.get('seniority_score', 0) - employee_seniority
        if seniority_gap > 0:
            score += (1.0 / seniority_gap) * WEIGHT_SENIORITY_GAP

        if cand_attrs.get('location') == employee_attrs.get('location'):
            score += WEIGHT_LOCATION_MATCH

        scored_candidates.append((score, employee_id, cand_id))

    return scored_candidates

def predict_managers_globally(G):
    all_possible_pairs = []
    print("Step 3: Scoring all possible employee-manager pairs...")
    for emp_id in tqdm(G.nodes(), desc="Scoring Progress"):
        all_possible_pairs.extend(score_potential_managers(emp_id, G))

    all_possible_pairs.sort(key=lambda x: x[0], reverse=True)

    print("\nStep 4: Building hierarchy and preventing cycles...")
    final_predictions = {}
    assigned_employees = set()
    hierarchy_graph = nx.DiGraph()

    for score, emp_id, mgr_id in tqdm(all_possible_pairs, desc="Assigning Managers"):
        if emp_id in assigned_employees:
            continue

        hierarchy_graph.add_edge(emp_id, mgr_id)

        if nx.is_directed_acyclic_graph(hierarchy_graph):
            final_predictions[emp_id] = mgr_id
            assigned_employees.add(emp_id)
        else:
            hierarchy_graph.remove_edge(emp_id, mgr_id)

    return final_predictions

# --- 5. MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--employees_path', default='data/employees.csv')
    parser.add_argument('--connections_path', default='data/connections.csv')
    parser.add_argument('--output_path', default='submission.csv')
    args = parser.parse_args()

    employees, connections = load_data(args.employees_path, args.connections_path)

    if employees is not None:
        company_graph = build_graph_with_features(employees, connections)
        manager_predictions = predict_managers_globally(company_graph)

        print("\nStep 5: Generating Submission File...")

        # Create submission dataframe and map predictions directly
        submission_df = pd.DataFrame({'employee_id': employees['employee_id']})
        submission_df['manager_id'] = submission_df['employee_id'].map(manager_predictions).fillna(0).astype(int)

        submission_df.loc[submission_df['employee_id'] == 358, 'manager_id'] = -1

        submission_df.to_csv(args.output_path, index=False)
        print(f"\nProcessing complete. Cycle-free submission file saved as '{args.output_path}'.")
