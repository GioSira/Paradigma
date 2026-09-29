from datetime import datetime
from uuid import UUID
from typing import List, Optional

from sqlalchemy import CheckConstraint, DateTime, Index, Text, Uuid, func, ForeignKey, ARRAY
from sqlalchemy.orm import Mapped, mapped_column
 
from src.sqldb.base import Base



class UserRow(Base):

    __tablename__ = "users"

    user_id:              Mapped[UUID]       = mapped_column(Uuid, primary_key=True)
    name:                 Mapped[str]        = mapped_column(Text,nullable=False)
    surname:              Mapped[str]        = mapped_column(Text, nullable=False)
    email:                Mapped[str]        = mapped_column(Text, nullable=False)
    password_hash:        Mapped[str]        = mapped_column(Text, nullable=False)
    created_at:           Mapped[datetime]   = mapped_column(DateTime(timezone=True), nullable=False)


    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="name_not_empty"),
        CheckConstraint("char_length(surname) > 0", name="surname_not_empty"),
        CheckConstraint("password_hash LIKE '$argon2%'", name="password_hash_is_argon2"),
    )



class TutorRow(UserRow):

    __tablename__ = "tutor"

    user_id:       Mapped[UUID]               = mapped_column(Uuid, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    bio:           Mapped[Optional[str]]      = mapped_column(Text, nullable=True)
    title:         Mapped[str]                = mapped_column(Text, nullable=False)
    subjects:      Mapped[List[str]]          = mapped_column(ARRAY(Text), nullable=False)

    __table_args__ = (
        CheckConstraint("cardinality(subjects) > 0", name="subjects_not_empty"),
    )



class StudentRow(UserRow):

    __tablename__ = "student"

    user_id:       Mapped[UUID]               = mapped_column(Uuid, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)


# no capital sensitive
USERS_EMAIL_UNIQUE = "uq_users_email_lower"
Index(USERS_EMAIL_UNIQUE, func.lower(UserRow.email), unique=True)
