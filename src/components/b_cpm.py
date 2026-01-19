import numpy as np
import yaml

class ContextualPreferenceModel:
    def __init__(self, config_path='config.yaml'):
        """
        Maintains the state of the XAI explanation style based on YAML-defined personas.
        Dimensions: [Technicality, Verbosity, Depth, Perspective]
        """
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        self.alpha = cfg.get('cpm', {}).get('learning_rate', 0.2)
        self.personas = {k: np.array(v) for k, v in cfg.get('personas', {}).items()}
        
        # Default state: balanced if no persona is loaded
        self.vector = np.array([0.5, 0.5, 0.5, 0.5]) 
        self.bounds = (0.0, 1.0)
        
        self.dim_map = ["technicality", "verbosity", "depth", "perspective"]

    def initialize(self, role: str):
        """Sets the initial vector based on a persona defined in the YAML."""
        if role in self.personas:
            self.vector = self.personas[role].copy()
            print(f" [CPM] Initialized with '{role}' persona: {self.vector}")
        else:
            print(f" [CPM] Warning: Role '{role}' not found in config. Using balanced defaults.")

    def update(self, dim_name: str, direction: int):
        """
        Updates a specific dimension based on user feedback.
        dim_name: 'technicality', 'verbosity', 'depth', or 'perspective'
        direction: +1 (Increase), -1 (Decrease)
        """
        if dim_name not in self.dim_map:
            print(f" [CPM] Error: Dimension '{dim_name}' does not exist.")
            return

        idx = self.dim_map.index(dim_name)
        old_val = self.vector[idx]
        
        # Apply update with clipping
        self.vector[idx] = np.clip(
            self.vector[idx] + (self.alpha * direction), 
            self.bounds[0], self.bounds[1]
        )
        
        print(f" [CPM] Updated {dim_name}: {old_val:.2f} -> {self.vector[idx]:.2f}")

    def get_state(self):
        """Returns the current vector as a dictionary for the NarrativeGenerator."""
        return {
            name: round(float(val), 2) 
            for name, val in zip(self.dim_map, self.vector)
        }

# Example 
if __name__ == "__main__":
 
    cpm = ContextualPreferenceModel('src/config/config.yaml')
    cpm.initialize("patient")
    
    # simulate user feedback: "Too simple, give me more detail"
    cpm.update("technicality", +1)
    cpm.update("depth", +1)

    style_for_llm = cpm.get_state()
    print("\n Final Style Vector for LLM Prompting:")
    print(style_for_llm)