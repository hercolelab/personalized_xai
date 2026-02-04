import streamlit as st
import sys
import os
import logging
from io import StringIO
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.orchestrator import HumanInteractiveOrchestrator
from webui.database import XAIDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("webui/app.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Personalized XAI Demo",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def get_database():
    return XAIDatabase("webui/webui.sqlite3")


@st.cache_resource
def get_orchestrator():
    logger.info("Initializing orchestrator...")
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        orchestrator = HumanInteractiveOrchestrator("src/config/config.yaml")
        return orchestrator
    finally:
        sys.stdout = old_stdout


def get_available_instances(orchestrator):
    with torch.no_grad():
        X_test_t = torch.tensor(orchestrator.backend.X_test.values, dtype=torch.float32)
        probs = orchestrator.backend.model(X_test_t).numpy().flatten()

    instances = []
    for idx, prob in enumerate(probs):
        instances.append({"index": idx, "risk_score": float(prob)})

    instances.sort(key=lambda x: x["risk_score"], reverse=True)
    return instances


def get_top_risk_candidate(orchestrator, top_k=5):
    instances = get_available_instances(orchestrator)
    top_instances = instances[:top_k] if instances else []
    if not top_instances:
        return None
    return top_instances[0]


def format_cpm_state(cpm_state):
    return {
        "Technicality": cpm_state.get("technicality", 0.5),
        "Verbosity": cpm_state.get("verbosity", 0.5),
        "Depth": cpm_state.get("depth", 0.5),
        "Perspective": cpm_state.get("perspective", 0.5),
    }


def display_cpm_metrics(cpm_state):
    formatted = format_cpm_state(cpm_state)
    cols = st.columns(4)
    for idx, (dim, value) in enumerate(formatted.items()):
        with cols[idx]:
            st.metric(label=dim, value=f"{value:.2f}")


def initialize_session():
    if "db" not in st.session_state:
        st.session_state.db = get_database()

    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = get_orchestrator()

    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None

    if "iteration" not in st.session_state:
        st.session_state.iteration = 0

    if "ground_truth" not in st.session_state:
        st.session_state.ground_truth = None

    if "current_narrative" not in st.session_state:
        st.session_state.current_narrative = None

    if "session_active" not in st.session_state:
        st.session_state.session_active = False

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "max_turns" not in st.session_state:
        st.session_state.max_turns = 10

    if "session_complete" not in st.session_state:
        st.session_state.session_complete = False

    if "instance_info" not in st.session_state:
        st.session_state.instance_info = None

    if "processing_feedback" not in st.session_state:
        st.session_state.processing_feedback = False

    if "pending_feedback" not in st.session_state:
        st.session_state.pending_feedback = None

    if "local_ground_truth" not in st.session_state:
        st.session_state.local_ground_truth = None

    if "local_raw_xai_text" not in st.session_state:
        st.session_state.local_raw_xai_text = None

    if "local_cpm_states" not in st.session_state:
        st.session_state.local_cpm_states = []

    if "local_narratives" not in st.session_state:
        st.session_state.local_narratives = []

    if "local_feedbacks" not in st.session_state:
        st.session_state.local_feedbacks = []


def start_new_session(instance_idx, initial_persona=None):
    logger.info(f"Starting new session for instance {instance_idx}")

    st.session_state.current_session_id = None
    st.session_state.iteration = 0
    st.session_state.session_active = True
    st.session_state.session_complete = False
    st.session_state.messages = []
    st.session_state.local_cpm_states = []
    st.session_state.local_narratives = []
    st.session_state.local_feedbacks = []
    st.session_state.local_ground_truth = None
    st.session_state.local_raw_xai_text = None

    st.session_state.orchestrator.reset_style()

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        raw_xai_text, ground_truth = (
            st.session_state.orchestrator.backend.get_explanation(instance_idx)
        )
        st.session_state.ground_truth = ground_truth
        st.session_state.local_ground_truth = ground_truth
        st.session_state.local_raw_xai_text = raw_xai_text
    finally:
        sys.stdout = old_stdout

    logger.info("Session started successfully")
    return None


def generate_narrative():
    if not st.session_state.session_active:
        st.error("No active session. Please start a new session.")
        return None

    logger.info(
        f"Generating narrative for session {st.session_state.current_session_id}"
    )

    target_style = st.session_state.orchestrator.cpm.get_state()

    st.session_state.local_cpm_states.append(
        {"iteration": st.session_state.iteration, **target_style}
    )

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        results = st.session_state.orchestrator._generate_verified(
            st.session_state.ground_truth, target_style
        )

        if results["status"] == "success":
            narrative = results["narrative"]
            st.session_state.current_narrative = narrative

            st.session_state.local_narratives.append(
                {
                    "iteration": st.session_state.iteration,
                    "narrative_text": narrative,
                    "attempt_number": results.get("attempts", 1),
                    "status": "success",
                }
            )

            logger.info(
                f"Successfully generated narrative for iteration {st.session_state.iteration}"
            )
            return results
        logger.warning(f"Failed to generate narrative: {results}")
        return results
    finally:
        sys.stdout = old_stdout


def apply_feedback(feedback_text):
    if not st.session_state.session_active:
        st.error("No active session.")
        return None

    logger.info(f"Applying feedback: {feedback_text}")

    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        deltas = st.session_state.orchestrator.feedback_translator.translate(
            feedback_text
        )
        st.session_state.orchestrator.cpm.apply_deltas(deltas)

        st.session_state.local_feedbacks.append(
            {
                "iteration": st.session_state.iteration,
                "feedback_text": feedback_text,
                "deltas": deltas,
            }
        )

        st.session_state.iteration += 1

        logger.info(f"Feedback applied. Deltas: {deltas}")
        return deltas
    finally:
        sys.stdout = old_stdout


def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})


def complete_session(status="completed"):
    final_narrative = st.session_state.current_narrative or ""
    if status in {"completed", "max_turns"} and st.session_state.instance_info:
        session_id = st.session_state.db.create_session(
            st.session_state.instance_info["index"], None
        )
        st.session_state.db.end_session(session_id, final_narrative, status=status)

        if st.session_state.local_ground_truth is not None:
            st.session_state.db.save_ground_truth(
                session_id,
                st.session_state.instance_info["index"],
                st.session_state.local_raw_xai_text or "",
                st.session_state.local_ground_truth,
            )

        for state in st.session_state.local_cpm_states:
            st.session_state.db.save_cpm_state(
                session_id,
                state["iteration"],
                {
                    "technicality": state.get("technicality", 0.5),
                    "verbosity": state.get("verbosity", 0.5),
                    "depth": state.get("depth", 0.5),
                    "perspective": state.get("perspective", 0.5),
                },
            )

        for item in st.session_state.local_narratives:
            st.session_state.db.save_narrative(
                session_id=session_id,
                iteration=item["iteration"],
                narrative_text=item["narrative_text"],
                attempt_number=item.get("attempt_number", 1),
                is_faithful=True,
                is_complete=True,
                is_aligned=True,
                status=item.get("status", "success"),
            )

        for item in st.session_state.local_feedbacks:
            st.session_state.db.save_feedback(
                session_id,
                item["iteration"],
                item["feedback_text"],
                item["deltas"],
            )
    st.session_state.session_active = False
    st.session_state.session_complete = True


def main():
    initialize_session()

    st.title("🧠 Personalized XAI Demo")
    st.caption("Single evaluation session.")

    if not st.session_state.session_active and not st.session_state.session_complete:
        candidate = get_top_risk_candidate(st.session_state.orchestrator, top_k=5)
        if candidate is None:
            st.error("No candidates available.")
            return

        st.session_state.instance_info = candidate
        start_new_session(candidate["index"], initial_persona=None)
        add_message(
            "assistant",
            (
                "Starting evaluation for a top-risk candidate. "
                f"Instance {candidate['index']} (risk {candidate['risk_score']:.3f})."
            ),
        )
        add_message("assistant", "Generating and verifying the narrative...")
        with st.spinner("Generating and verifying narrative..."):
            results = generate_narrative()
        if results and results.get("status") == "success":
            add_message("assistant", results["narrative"])
        else:
            add_message(
                "assistant",
                "The system could not verify a safe narrative for this case.",
            )

    if st.session_state.session_active:
        current_cpm = st.session_state.orchestrator.cpm.get_state()
        st.subheader("Current CPM Status")
        display_cpm_metrics(current_cpm)
        turns_left = st.session_state.max_turns - st.session_state.iteration
        st.caption(f"Turns remaining: {max(0, turns_left)}")
        st.divider()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.processing_feedback and st.session_state.pending_feedback:
        with st.chat_message("assistant"):
            st.markdown("*Refining and verifying narrative...*")

        with st.spinner("Processing..."):
            apply_feedback(st.session_state.pending_feedback)
            results = generate_narrative()

        if results and results.get("status") == "success":
            add_message("assistant", results["narrative"])
        else:
            add_message(
                "assistant",
                "The system could not verify a safe narrative for this revision.",
            )

        st.session_state.processing_feedback = False
        st.session_state.pending_feedback = None
        st.rerun()

    if st.session_state.session_active and not st.session_state.session_complete:
        if st.session_state.iteration >= st.session_state.max_turns:
            complete_session(status="max_turns")
            add_message("assistant", "Thank you for your feedback.")
            st.rerun()

        feedback_text = st.chat_input(
            "Add feedback to refine the narrative (up to 10 turns)..."
        )
        if feedback_text:
            add_message("user", feedback_text)
            st.session_state.processing_feedback = True
            st.session_state.pending_feedback = feedback_text
            st.rerun()

    if st.session_state.session_active and not st.session_state.session_complete:
        if st.button("I'm satisfied, end session", type="primary"):
            complete_session(status="completed")
            add_message("assistant", "Thank you for your feedback.")
            st.rerun()

    if st.session_state.session_complete:
        st.markdown("Thank you for your feedback.")


if __name__ == "__main__":
    main()
