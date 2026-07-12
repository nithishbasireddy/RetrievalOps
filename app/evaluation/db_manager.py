import sqlite3
from pathlib import Path

DB_PATH = Path("data/experiments.db")


def init_db():

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Create experiments table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        config_name TEXT UNIQUE,
        retrieval_type TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        avg_faithfulness REAL,
        avg_answer_relevance REAL,
        avg_context_precision REAL,
        avg_context_recall REAL
    );
    """)

    # Create evaluation_results table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS evaluation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        experiment_id INTEGER,
        question_id INTEGER,
        question TEXT,
        response TEXT,
        reference_answer TEXT,
        faithfulness REAL,
        answer_relevance REAL,
        context_precision REAL,
        context_recall REAL,
        FOREIGN KEY (experiment_id) REFERENCES experiments (id) ON DELETE CASCADE
    );
    """)

    # Create retrieved_chunks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS retrieved_chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluation_result_id INTEGER,
        chunk_id TEXT,
        document TEXT,
        chunk_index INTEGER,
        rank INTEGER,
        content TEXT,
        FOREIGN KEY (evaluation_result_id) REFERENCES evaluation_results (id) ON DELETE CASCADE
    );
    """)

    conn.commit()

    conn.close()


def save_experiment(
    config_name,
    retrieval_type,
    results,
    agg_metrics
):

    conn = sqlite3.connect(DB_PATH)

    cursor = conn.cursor()

    cursor.execute("PRAGMA foreign_keys = ON;")

    # Insert or update experiment
    cursor.execute("""
    INSERT OR REPLACE INTO experiments (
        config_name,
        retrieval_type,
        timestamp,
        avg_faithfulness,
        avg_answer_relevance,
        avg_context_precision,
        avg_context_recall
    ) VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?);
    """, (
        config_name,
        retrieval_type,
        agg_metrics.get("faithfulness"),
        agg_metrics.get("answer_relevance"),
        agg_metrics.get("context_precision"),
        agg_metrics.get("context_recall")
    ))

    experiment_id = cursor.lastrowid

    # If the experiment already existed, INSERT OR REPLACE doesn't give us the updated id properly
    # unless we search for it.
    if not experiment_id:

        cursor.execute(
            "SELECT id FROM experiments WHERE config_name = ?;",
            (config_name,)
        )

        experiment_id = cursor.fetchone()[0]

        # Clean up old results for this config
        cursor.execute(
            "DELETE FROM evaluation_results WHERE experiment_id = ?;",
            (experiment_id,)
        )

    for item in results:

        cursor.execute("""
        INSERT INTO evaluation_results (
            experiment_id,
            question_id,
            question,
            response,
            reference_answer,
            faithfulness,
            answer_relevance,
            context_precision,
            context_recall
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            experiment_id,
            item.get("question_id"),
            item.get("question"),
            item.get("response"),
            item.get("reference_answer"),
            item.get("metrics", {}).get("faithfulness"),
            item.get("metrics", {}).get("answer_relevance"),
            item.get("metrics", {}).get("context_precision"),
            item.get("metrics", {}).get("context_recall")
        ))

        eval_id = cursor.lastrowid

        for rank, doc in enumerate(item.get("retrieved_docs", []), start=1):

            cursor.execute("""
            INSERT INTO retrieved_chunks (
                evaluation_result_id,
                chunk_id,
                document,
                chunk_index,
                rank,
                content
            ) VALUES (?, ?, ?, ?, ?, ?);
            """, (
                eval_id,
                doc.metadata.get("chunk_id", "N/A"),
                doc.metadata.get("document", "N/A"),
                doc.metadata.get("chunk_index", -1),
                rank,
                doc.page_content
            ))

    conn.commit()

    conn.close()


if __name__ == "__main__":
    init_db()
    print("Database initialized.")
