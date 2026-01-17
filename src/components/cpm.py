import numpy as np

class ContextualPreferenceModel:
    def __init__(self, learning_rate=0.2):
        """
        Maintains the deterministic state of the XAI explanation style.
        Dimensions: [Technicality, Verbosity, Depth, Perspective]
        Range: 0.0 to 1.0 for each dimension.
        """
        self.alpha = learning_rate
        # Default state: Balanced
        self.vector = np.array([0.5, 0.5, 0.5, 0.5]) 
        self.bounds = (0.0, 1.0)
        
        # PRESET PERSONAS
        # Perspective: 0.0 = Pure SHAP (Why), 1.0 = Pure DiCE (How to change)
        self.PERSONAS = {
            "patient":   np.array([0.2, 0.6, 0.3, 0.8]), # Simple, Action-oriented
            "clinician": np.array([0.9, 0.3, 0.8, 0.2]), # High Tech, Cause-oriented
        }

    def initialize(self, role: str):
        """Sets the initial vector based on a predefined role."""
        if role in self.PERSONAS:
            self.vector = self.PERSONAS[role].copy()
            print(f" [CPM] Initialized with '{role}' persona.")

    def update(self, dim_idx: int, direction: int):
        """
        Updates a specific dimension based on user feedback.
        dim_idx: 0 (Tech), 1 (Verbosity), 2 (Depth), 3 (Perspective)
        direction: +1 (Increase), -1 (Decrease)
        """
        old_val = self.vector[dim_idx]
        self.vector[dim_idx] = np.clip(
            self.vector[dim_idx] + (self.alpha * direction), 
            self.bounds[0], self.bounds[1]
        )
        print(f" [CPM] Updated Dim {dim_idx}: {old_val:.2f} -> {self.vector[dim_idx]:.2f}")

    def get_state(self):
        """Returns the current vector as a dictionary for the LLM Generator."""
        return {
            "technicality": round(self.vector[0], 2),
            "verbosity":    round(self.vector[1], 2),
            "depth":        round(self.vector[2], 2),
            "perspective":  round(self.vector[3], 2)
        }