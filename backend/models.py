from datetime import datetime
from sqlalchemy import String, Integer, Float, Boolean, DateTime, JSON, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class Assessment(Base):
    __tablename__ = "assessments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    patient_id: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    input_data: Mapped[dict] = mapped_column(JSON, nullable=False)
    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    risk_percentage: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)
    flagged_for_review: Mapped[bool] = mapped_column(Boolean, nullable=False)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    n_new_samples: Mapped[int] = mapped_column(Integer, default=0)
    n_total_samples: Mapped[int] = mapped_column(Integer, default=0)
    roc_auc: Mapped[float] = mapped_column(Float, nullable=True)
    recall_class1: Mapped[float] = mapped_column(Float, nullable=True)
    pr_auc: Mapped[float] = mapped_column(Float, nullable=True)
    f1_score: Mapped[float] = mapped_column(Float, nullable=True)
    decision_threshold: Mapped[float] = mapped_column(Float, nullable=True)
    model_type: Mapped[str] = mapped_column(String(50), default="StackingClassifier")
    error_message: Mapped[str] = mapped_column(Text, nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class UploadedDataset(Base):
    __tablename__ = "uploaded_datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False)
    n_rows: Mapped[int] = mapped_column(Integer, default=0)
    n_valid_rows: Mapped[int] = mapped_column(Integer, default=0)
    validation_errors: Mapped[dict] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    training_run_id: Mapped[int] = mapped_column(Integer, ForeignKey("training_runs.id"), nullable=True)
