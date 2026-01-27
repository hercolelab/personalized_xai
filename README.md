# Personalized XAI Narrative Pipeline

This repository implements a modular pipeline designed to translate raw **Explainable AI (XAI)** data into user-tailored narratives. By utilizing a **Contextual Preference Model (CPM)** and a multi-stage **Verification Loop**, the system ensures that explanations are both faithful to the underlying model and stylistically aligned with the target audience.

---

## Use Case

The system targets clinical decision support, specifically for diabetes risk prediction. It transforms complex statistical outputs—**SHAP impact values** and **DiCE counterfactuals**—into coherent natural language explanations. The pipeline dynamically adapts its tone and content based on the user profile:

- **Clinician**: Cold, analytical, and data-driven reports focusing on statistical relevance and SHAP contributions.  
- **Patient**: Proactive, coaching-oriented narratives emphasizing actionable lifestyle changes derived from counterfactual explanations.

---

## Repository Structure

The repository follows a modular architecture that separates data processing, style modeling, narrative generation, and verification:

- **`src/components/a_setup_xai.py`**: The backend module. Handles the `diabetes.csv` dataset, trains predictive models, and generates raw SHAP and DiCE explanations.
- **`src/components/b_cpm.py`**: The **Contextual Preference Model**. Defines style vectors (Technicality, Verbosity, Depth, Perspective) for different user roles.
- **`src/components/c_generator.py`**: The narrative engine. Interfaces with a local LLM (Ollama) to synthesize explanations using templates and instructions from `prompts.yaml`.
- **`src/components/d_verifiers.py`**: The verification suite. Performs **Rejection Sampling** based on numerical faithfulness, feature completeness, and stylistic rubric alignment.
- **`src/config/config.yaml`**: Centralized configuration file for API endpoints, model parameters, numerical tolerances, and feature aliases.
- **`src/prompts/prompts.yaml`**: The core prompt repository. Contains system roles, clinical guardrails, and categorical rubrics used by the AI Judge for alignment evaluation.
- **`src/orchestrator.py`**: The execution core. Manages the generation–verification loop and provides corrective feedback to the LLM for iterative self-improvement.

---

## Getting Started

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set Up the LLM Backend**

   **Local mode (no auth required):** ensure that Ollama is running locally and that the required model is available:

   ```bash
   ollama serve
   ollama pull llama3.1:8b
   ```

   **Cloud mode (requires auth):** authenticate, then pull and use the cloud model.

   ```bash
   # API key via environment
   export OLLAMA_API_KEY="<your_api_key>"

   # Start the Ollama server and pull the cloud model
   ollama serve
   ollama pull gpt-oss:20b-cloud
   ```

3. **Configure Settings**

   Adjust numerical tolerances in `src/config/config.yaml` and refine stylistic behaviors or categorical levels in `src/prompts/prompts.yaml` as needed.

---

## Usage

To generate a verified narrative for a specific case (by default, the orchestrator selects the highest-risk instance for testing), run:

```bash
python3 src/orchestrator.py
```

The system will attempt up to 10 iterations to generate a narrative that satisfies all verification criteria. If an attempt fails, targeted correction hints are automatically fed back to the LLM to guide refinement of the next output. 