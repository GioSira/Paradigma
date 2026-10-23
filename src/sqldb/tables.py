from datetime import datetime
from uuid import UUID
from typing import List, Optional

from sqlalchemy import (CheckConstraint, DateTime, Index, Text, Uuid,
                         func, ForeignKey, Integer, Table, Column,
                         DDL, event)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column
 
from src.sqldb.base import Base



# =========================================== TABELLE BASE ===========================================

class UserRow(Base):

    __tablename__ = "users"

    user_id:              Mapped[UUID]               = mapped_column(Uuid, primary_key=True)
    name:                 Mapped[str]                = mapped_column(Text,nullable=False)
    surname:              Mapped[str]                = mapped_column(Text, nullable=False)
    email:                Mapped[str]                = mapped_column(Text, nullable=False)
    password_hash:        Mapped[str]                = mapped_column(Text, nullable=False)
    created_at:           Mapped[datetime]           = mapped_column(DateTime(timezone=True), nullable=False)


    __table_args__ = (
        CheckConstraint("char_length(name) > 0", name="name_not_empty"),
        CheckConstraint("char_length(surname) > 0", name="surname_not_empty"),
        CheckConstraint("password_hash LIKE '$argon2%'", name="password_hash_is_argon2"),
    )



class TutorRow(UserRow):

    __tablename__ = "tutor"

    user_id:              Mapped[UUID]               = mapped_column(Uuid, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    bio:                  Mapped[Optional[str]]      = mapped_column(Text, nullable=True)
    title:                Mapped[str]                = mapped_column(Text, nullable=False)
    subjects:             Mapped[List[str]]          = mapped_column(ARRAY(Text), nullable=False)

    __table_args__ = (
        CheckConstraint("cardinality(subjects) > 0", name="subjects_not_empty"),
    )



class StudentRow(UserRow):

    __tablename__ = "student"

    user_id:              Mapped[UUID]               = mapped_column(Uuid, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    #abbonamento:         Mapped[UUID]               = mapped_column(Uuid, ForeignKey("abbonamento.id")) 
    #corsi:               Mapped[UUID]               = mapped_column(Uuid, ForeignKey("corso.id"))



class CorsoRow(Base):

    __tablename__ = "corso"

    id:                  Mapped[UUID]               = mapped_column(Uuid, primary_key=True)
    title:               Mapped[str]                = mapped_column(Text, nullable=False)
    #docente:            Mapped[List[Tutor]]        = mapped_column(ARRAY(Tutor)) 
    descrizione:         Mapped[str]                = mapped_column(Text)
    data_inizio:         Mapped[datetime]           = mapped_column(DateTime(timezone=True), nullable=False)
    data_fine:           Mapped[datetime]           = mapped_column(DateTime(timezone=True), nullable=False)


    __table_args__ = (
        CheckConstraint("char_length(title) > 0", name="title_not_empty"),
        #CheckConstraint("cardinality(docente) > 0", name="docente_not_empty"),
        CheckConstraint("data_fine > data_inizio", name="date_ordinate")
    )


class PianoRow(Base):
    
    __tablename__ = "piano"

    id:                 Mapped[UUID]                = mapped_column(Uuid, primary_key=True)
    nome:               Mapped[str]                 = mapped_column(Text, nullable=False, unique=True)
    descrizione:        Mapped[str]                 = mapped_column(Text, nullable=False)
    durata:             Mapped[int]                 = mapped_column(Integer, nullable=False)
    costo:              Mapped[int]                 = mapped_column(Integer, nullable=False)
    
    __table_args__ = (
        CheckConstraint("durata > 0", name="piano_durata_positiva"),
        CheckConstraint("costo >= 0", name="piano_costo_non_negativo"),
    )


class AbbonamentoRow(Base):

    __tablename__ = "abbonamento"

    id:                 Mapped[UUID]                = mapped_column(Uuid, primary_key=True)
    student_id:         Mapped[UUID]                = mapped_column(Uuid, ForeignKey("student.user_id", ondelete="CASCADE"))
    durata:             Mapped[int]                 = mapped_column(Integer, nullable=False)
    inizio:             Mapped[datetime]            = mapped_column(DateTime(timezone=True), nullable=False)
    fine:               Mapped[datetime]            = mapped_column(DateTime(timezone=True), nullable=False)
    costo:              Mapped[int]                 = mapped_column(Integer, nullable=False)


    __table_args__ = (
        CheckConstraint("fine > inizio", name="date_ordinate"),
        CheckConstraint("costo >= 0", name="abbonamento_non_negativo")
    )


# =========================================== TABELLE JOIN ===========================================

CORSO_DOCENTE = Table(
    "corso_docente", Base.metadata, 
    Column("corso_id", Uuid, ForeignKey("corso.id", ondelete="CASCADE"), primary_key=True),
    Column("tutor_id", Uuid, ForeignKey("tutor.user_id", ondelete="CASCADE"), primary_key=True)
)

ISCRIZIONE = Table(
    "iscrizione", Base.metadata,
    Column("student_id", Uuid, ForeignKey("student.user_id", ondelete="CASCADE"), primary_key=True),
    Column("corso_id", Uuid, ForeignKey("corso.id", ondelete="CASCADE"), primary_key=True)
)

# ============================================== INDICI ==============================================


# no capital sensitive
USERS_EMAIL_UNIQUE = "uq_users_email_lower"
Index(USERS_EMAIL_UNIQUE, func.lower(UserRow.email), unique=True)

Index("ix_abbonamento_student", AbbonamentoRow.student_id)

Index("ix_corso_docente", CORSO_DOCENTE.c.tutor_id)

Index("ix_iscrizione_corso", ISCRIZIONE.c.corso_id)


# ============================================== TRIGGER =============================================

CORSO_ULTIMO_DOCENTE = "corso_ultimo_docente"

_CORSO_ULTIMO_DOCENTE_FUNCTION = DDL("""
CREATE OR REPLACE FUNCTION corso_conserva_un_docente() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM 1 FROM corso WHERE id = OLD.corso_id FOR UPDATE;
    IF NOT FOUND THEN
        RETURN NULL;
    END IF;
 
    IF NOT EXISTS (SELECT 1 FROM corso_docente WHERE corso_id = OLD.corso_id) THEN
        RAISE EXCEPTION USING
            MESSAGE    = 'un corso deve avere almeno un docente',
            DETAIL     = OLD.corso_id::text,
            ERRCODE    = 'restrict_violation',
            CONSTRAINT = 'corso_ultimo_docente';
    END IF;
 
    RETURN NULL;
END
$$
""")
 
_CORSO_ULTIMO_DOCENTE_TRIGGER = DDL("""
CREATE TRIGGER corso_ultimo_docente
AFTER DELETE ON corso_docente
FOR EACH ROW EXECUTE FUNCTION corso_conserva_un_docente()
""")

event.listen(CORSO_DOCENTE, "after_create", _CORSO_ULTIMO_DOCENTE_FUNCTION.execute_if(dialect="postgresql"))
event.listen(CORSO_DOCENTE, "after_create", _CORSO_ULTIMO_DOCENTE_TRIGGER.execute_if(dialect="postgresql"))
