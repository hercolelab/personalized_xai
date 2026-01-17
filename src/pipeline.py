import os
from src.setup_xai import XAI_Backend
from src.components.cpm import ContextualPreferenceModel
from src.components.generator import NarrativeGenerator
from termcolor import colored

def main():
    # 1. Initialize the Table (Magician's Tools)
    # The Backend trains the NN and initializes SHAP/DiCE for PIMA
    backend = XAI_Backend(csv_path='diabetes.csv')
    
    # The CPM maintains the deterministic state machine
    cpm = ContextualPreferenceModel()
    
    # The Narratore runs the local SLM (e.g., Llama 3) via Ollama
    narrator = NarrativeGenerator(model_name="llama3")

    print(colored("\n" + "="*50, "cyan"))
    print(colored(" SEARCHING FOR REPRESENTATIVE CASE ", "cyan", attrs=['bold']))
    print(colored("="*50, "cyan"))

    # 2. Identify the highest-risk patient in the test set
    # This ensures we have a rich "High Risk" scenario to explain
    import torch
    import numpy as np
    
    with torch.no_grad():
        X_test_t = torch.tensor(backend.X_test.values, dtype=torch.float32)
        probs = backend.model(X_test_t).numpy().flatten()
    
    best_idx = int(np.argmax(probs))
    max_prob = probs[best_idx]
    
    print(f" [Search] Patient Index {best_idx} identified with {max_prob:.2%} Risk.")

    # 3. Step Into the 'Appeso' (Hanged Man) Perspective
    # We generate the raw, un-scaled clinical explanation first
    raw_xai_data = backend.get_explanation(best_idx)

    print(colored("\n" + "-"*20 + " [STAGE 1: RAW XAI DATA] " + "-"*20, "yellow"))
    print(raw_xai_data)
    print(colored("-" * 65, "yellow"))

    # 4. Turn 1: Patient Persona
    # Initialize the CPM for a non-technical user (Action-Oriented)
    print(colored("\n [Orchestration] Generating Turn 1: Patient Style...", "green"))
    cpm.initialize("patient")
    style_v1 = cpm.get_state()
    
    narrative_v1 = narrator.generate(raw_xai_data, style_v1)
    
    print(colored("\n" + "*"*20 + " [PATIENT NARRATIVE] " + "*"*20, "white", attrs=['bold']))
    print(narrative_v1)
    print(colored("*" * 61, "white", attrs=['bold']))

    # 5. Turn 2: User Feedback (The 'Matto's' Journey)
    # The user asks for more technical depth, triggering a CPM state change
    print(colored("\n [Orchestration] User requested more detail. Updating CPM...", "green"))
    
    # Feedback: Increase Technicality (Index 0) and Depth (Index 2)
    cpm.update(0, 1) # Technicality +
    cpm.update(2, 1) # Depth +
    
    style_v2 = cpm.get_state()
    narrative_v2 = narrator.generate(raw_xai_data, style_v2)

    print(colored("\n" + "*"*20 + " [TECHNICAL NARRATIVE] " + "*"*20, "magenta", attrs=['bold']))
    print(narrative_v2)
    print(colored("*" * 63, "magenta", attrs=['bold']))

if __name__ == "__main__":
    main()