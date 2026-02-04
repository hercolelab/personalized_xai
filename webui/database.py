import sqlite3
import json
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class XAIDatabase:
    def __init__(self, db_path="webui/webui.sqlite3"):
        self.db_path = db_path
        self.init_database()

    def get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)

    def init_database(self):
        """Initialize the database schema."""
        conn = self.get_connection()
        cursor = conn.cursor()

        # Sessions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                instance_idx INTEGER NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                status TEXT DEFAULT 'active',
                initial_persona TEXT,
                final_narrative TEXT
            )
        """
        )

        # Narratives table - stores each generated narrative
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS narratives (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                iteration INTEGER NOT NULL,
                narrative_text TEXT NOT NULL,
                attempt_number INTEGER,
                is_faithful BOOLEAN,
                is_complete BOOLEAN,
                is_aligned BOOLEAN,
                faithfulness_reason TEXT,
                completeness_reason TEXT,
                alignment_report TEXT,
                status TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """
        )

        # CPM states table - tracks CPM evolution
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cpm_states (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                iteration INTEGER NOT NULL,
                technicality REAL NOT NULL,
                verbosity REAL NOT NULL,
                depth REAL NOT NULL,
                perspective REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """
        )

        # Feedbacks table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedbacks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                iteration INTEGER NOT NULL,
                feedback_text TEXT NOT NULL,
                translated_deltas TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """
        )

        # Ground truth table - stores the raw XAI data for each instance
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS ground_truths (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                instance_idx INTEGER NOT NULL,
                raw_xai_text TEXT NOT NULL,
                ground_truth_json TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """
        )

        conn.commit()
        self._migrate_schema(conn)
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")

    def _migrate_schema(self, conn):
        """Apply lightweight schema migrations for existing databases."""
        cursor = conn.cursor()

        # Sessions table migrations
        cursor.execute("PRAGMA table_info(sessions)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        missing_cols = []

        if "ended_at" not in existing_cols:
            missing_cols.append("ALTER TABLE sessions ADD COLUMN ended_at TIMESTAMP")
        if "initial_persona" not in existing_cols:
            missing_cols.append("ALTER TABLE sessions ADD COLUMN initial_persona TEXT")
        if "final_narrative" not in existing_cols:
            missing_cols.append("ALTER TABLE sessions ADD COLUMN final_narrative TEXT")

        for stmt in missing_cols:
            cursor.execute(stmt)
            logger.info(f"Applied migration: {stmt}")

        conn.commit()

    # Session operations
    def create_session(
        self, instance_idx: int, initial_persona: Optional[str] = None
    ) -> int:
        """Create a new session and return its ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO sessions (instance_idx, initial_persona)
            VALUES (?, ?)
        """,
            (instance_idx, initial_persona),
        )
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"Created session {session_id} for instance {instance_idx}")
        return session_id

    def end_session(
        self, session_id: int, final_narrative: str, status: str = "completed"
    ):
        """Mark a session as ended."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE sessions
            SET ended_at = CURRENT_TIMESTAMP, status = ?, final_narrative = ?
            WHERE id = ?
        """,
            (status, final_narrative, session_id),
        )
        conn.commit()
        conn.close()
        logger.info(f"Session {session_id} ended with status: {status}")

    def get_session(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get session details."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, instance_idx, started_at, ended_at, status, initial_persona, final_narrative
            FROM sessions
            WHERE id = ?
        """,
            (session_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "instance_idx": row[1],
                "started_at": row[2],
                "ended_at": row[3],
                "status": row[4],
                "initial_persona": row[5],
                "final_narrative": row[6],
            }
        return None

    # Narrative operations
    def save_narrative(
        self,
        session_id: int,
        iteration: int,
        narrative_text: str,
        attempt_number: int,
        is_faithful: bool,
        is_complete: bool,
        is_aligned: bool,
        faithfulness_reason: Optional[str] = None,
        completeness_reason: Optional[str] = None,
        alignment_report: Optional[str] = None,
        status: str = "generated",
    ) -> int:
        """Save a generated narrative with verification results."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO narratives (
                session_id, iteration, narrative_text, attempt_number,
                is_faithful, is_complete, is_aligned,
                faithfulness_reason, completeness_reason, alignment_report, status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                iteration,
                narrative_text,
                attempt_number,
                is_faithful,
                is_complete,
                is_aligned,
                faithfulness_reason,
                completeness_reason,
                json.dumps(alignment_report) if alignment_report else None,
                status,
            ),
        )
        narrative_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(
            f"Saved narrative {narrative_id} for session {session_id}, iteration {iteration}"
        )
        return narrative_id

    def get_session_narratives(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all narratives for a session."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, iteration, narrative_text, attempt_number,
                   is_faithful, is_complete, is_aligned,
                   faithfulness_reason, completeness_reason, alignment_report,
                   status, created_at
            FROM narratives
            WHERE session_id = ?
            ORDER BY iteration, attempt_number
        """,
            (session_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        narratives = []
        for row in rows:
            narratives.append(
                {
                    "id": row[0],
                    "iteration": row[1],
                    "narrative_text": row[2],
                    "attempt_number": row[3],
                    "is_faithful": row[4],
                    "is_complete": row[5],
                    "is_aligned": row[6],
                    "faithfulness_reason": row[7],
                    "completeness_reason": row[8],
                    "alignment_report": json.loads(row[9]) if row[9] else None,
                    "status": row[10],
                    "created_at": row[11],
                }
            )
        return narratives

    # CPM state operations
    def save_cpm_state(
        self, session_id: int, iteration: int, cpm_state: Dict[str, float]
    ) -> int:
        """Save the current CPM state."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO cpm_states (session_id, iteration, technicality, verbosity, depth, perspective)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                session_id,
                iteration,
                cpm_state.get("technicality", 0.5),
                cpm_state.get("verbosity", 0.5),
                cpm_state.get("depth", 0.5),
                cpm_state.get("perspective", 0.5),
            ),
        )
        cpm_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(
            f"Saved CPM state {cpm_id} for session {session_id}, iteration {iteration}"
        )
        return cpm_id

    def get_cpm_history(self, session_id: int) -> List[Dict[str, Any]]:
        """Get CPM state history for a session."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT iteration, technicality, verbosity, depth, perspective, created_at
            FROM cpm_states
            WHERE session_id = ?
            ORDER BY iteration
        """,
            (session_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        history = []
        for row in rows:
            history.append(
                {
                    "iteration": row[0],
                    "technicality": row[1],
                    "verbosity": row[2],
                    "depth": row[3],
                    "perspective": row[4],
                    "created_at": row[5],
                }
            )
        return history

    # Feedback operations
    def save_feedback(
        self,
        session_id: int,
        iteration: int,
        feedback_text: str,
        translated_deltas: Dict[str, float],
    ) -> int:
        """Save user feedback and translated deltas."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO feedbacks (session_id, iteration, feedback_text, translated_deltas)
            VALUES (?, ?, ?, ?)
        """,
            (session_id, iteration, feedback_text, json.dumps(translated_deltas)),
        )
        feedback_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(
            f"Saved feedback {feedback_id} for session {session_id}, iteration {iteration}"
        )
        return feedback_id

    def get_session_feedbacks(self, session_id: int) -> List[Dict[str, Any]]:
        """Get all feedbacks for a session."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, iteration, feedback_text, translated_deltas, created_at
            FROM feedbacks
            WHERE session_id = ?
            ORDER BY iteration
        """,
            (session_id,),
        )
        rows = cursor.fetchall()
        conn.close()

        feedbacks = []
        for row in rows:
            feedbacks.append(
                {
                    "id": row[0],
                    "iteration": row[1],
                    "feedback_text": row[2],
                    "translated_deltas": json.loads(row[3]) if row[3] else {},
                    "created_at": row[4],
                }
            )
        return feedbacks

    # Ground truth operations
    def save_ground_truth(
        self,
        session_id: int,
        instance_idx: int,
        raw_xai_text: str,
        ground_truth: Dict[str, Any],
    ) -> int:
        """Save the ground truth data for an instance."""
        conn = self.get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO ground_truths (session_id, instance_idx, raw_xai_text, ground_truth_json)
            VALUES (?, ?, ?, ?)
        """,
            (session_id, instance_idx, raw_xai_text, json.dumps(ground_truth)),
        )
        gt_id = cursor.lastrowid
        conn.commit()
        conn.close()
        logger.info(f"Saved ground truth {gt_id} for session {session_id}")
        return gt_id

    def get_ground_truth(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get ground truth for a session."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT raw_xai_text, ground_truth_json
            FROM ground_truths
            WHERE session_id = ?
        """,
            (session_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "raw_xai_text": row[0],
                "ground_truth": json.loads(row[1]) if row[1] else {},
            }
        return None

    # Analytics operations
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get all sessions summary."""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT s.id, s.instance_idx, s.started_at, s.ended_at, s.status,
                   COUNT(DISTINCT f.id) as feedback_count,
                   COUNT(DISTINCT n.id) as narrative_count
            FROM sessions s
            LEFT JOIN feedbacks f ON s.id = f.session_id
            LEFT JOIN narratives n ON s.id = n.session_id
            GROUP BY s.id
            ORDER BY s.started_at DESC
        """
        )
        rows = cursor.fetchall()
        conn.close()

        sessions = []
        for row in rows:
            sessions.append(
                {
                    "id": row[0],
                    "instance_idx": row[1],
                    "started_at": row[2],
                    "ended_at": row[3],
                    "status": row[4],
                    "feedback_count": row[5],
                    "narrative_count": row[6],
                }
            )
        return sessions
