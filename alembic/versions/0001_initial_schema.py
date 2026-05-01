"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-01
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "candidates",
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("email", sa.Text, nullable=True),
        sa.Column("seniority_level", sa.Text, nullable=False),
        sa.Column("domain", sa.Text, nullable=False),
        sa.Column("years_experience", sa.Float, nullable=False),
        sa.Column("skills", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("education", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("work_history", postgresql.JSONB, nullable=False, server_default="[]"),
        sa.Column("raw_text", sa.Text, nullable=False, server_default=""),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("idx_candidates_email", "candidates", ["email"])
    op.create_index("idx_candidates_created_at", "candidates", ["created_at"])

    op.create_table(
        "match_jobs",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("candidate_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("jd_input", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="pending"),
        sa.Column("attempt_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("processing_started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_detail", sa.Text, nullable=True),
        sa.Column("result", postgresql.JSONB, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.ForeignKeyConstraint(["candidate_id"], ["candidates.candidate_id"]),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed')",
            name="ck_match_jobs_status",
        ),
    )
    op.create_index("idx_match_jobs_status_created", "match_jobs", ["status", "created_at"])
    op.create_index("idx_match_jobs_candidate_id", "match_jobs", ["candidate_id"])
    op.create_index("idx_match_jobs_status_updated", "match_jobs", ["status", "updated_at"])


def downgrade() -> None:
    op.drop_table("match_jobs")
    op.drop_table("candidates")
