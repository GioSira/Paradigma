from src.roles.user import User
from src.corsi.corso import Corso
from pydantic import Field, field_validator
from typing import Optional, List, Any
from uuid import UUID
from datetime import datetime, timezone


class Tutor(User):


    title: str = Field(..., min_length=1)
    bio: Optional[str] = Field(..., min_length=1)
    subjects: List[str]
    corso: List[Corso]


    # ----------------------------- COSTRUZIONE -----------------------------

    @classmethod
    def create(cls, user_id: UUID, name: str, surname: str, email: str, password: str,
               tutor_title: str, subjects: List[str], corso: List[Corso] = [],
               bio: Optional[str] = None, created_at: Optional[datetime] = None) -> "Tutor":
        """Registrazione di un nuovo utente, a partire dalla password in chiaro.

        Per ricaricare un utente dal database si usa invece il costruttore,
        passando l'hash salvato: User(..., password_hash=hash_dal_db).
        """
        return super().create(
            user_id=user_id,
            name=name,
            surname=surname,
            email=email,
            password=password,
            created_at=created_at or datetime.now(timezone.utc),
            title=tutor_title,
            subjects=subjects,
            bio=bio,
            corso = corso,
        )


    @field_validator("subjects", mode="before")
    @classmethod
    def normalize_subjects(cls, subjects: Any) -> List[str]:
        try:
            unique = cls.remove_duplicates(list(subjects))
        except ValueError:
            raise
        return unique


    # ----------------------------- GETTER/SETTER ----------------------------------

    def get_title(self):
        return self.title

    def set_title(self, new_title: str):
        self.title = new_title


    def get_subjects(self):
        return self.subjects

    def set_subjects(self, new_subjects: List[str]):
        if not isinstance(new_subjects, list):
            raise TypeError("Must be a list")
               
        self.subjects = new_subjects


    def get_bio(self):
        return self.bio

    def set_bio(self, new_bio: str):
        self.bio = new_bio


    # ----------------------------- SUBJECT METHODS ----------------------------------

    def format_subjects(self):
        return "; ".join(self.subjects)


    def remove_subject(self, subject_to_remove: str):
        v_to_remove = subject_to_remove.strip().casefold()
        removed_list = [subject for subject in self.get_subjects() if v_to_remove != subject.strip().casefold()]
        self.set_subjects(removed_list)


    def remove_subjects(self, list_of_subjects: List[str]):
        if not isinstance(list_of_subjects, list):
            raise TypeError("Must be a list")
        
        to_remove_list = [s.strip().casefold() for s in list_of_subjects]

        unique = []
        for subj_elem in self.get_subjects():
            subj_v = subj_elem.strip().casefold()
            if subj_v and subj_v not in to_remove_list:
                unique.append(subj_elem)

        self.set_subjects(unique)


    def add_subject(self, subject_to_add: str):
        unique_subjects = self.remove_duplicates(self.get_subjects() + [subject_to_add])
        self.set_subjects(unique_subjects)


    def add_subjects(self, list_of_new_subjects: List[str]):
        if not isinstance(list_of_new_subjects, list):
            raise TypeError("Must be a list")
        
        if len(list_of_new_subjects) == 0:
            raise ValueError("The list must not be empty")
        
        unique_subjects = self.remove_duplicates(self.get_subjects() + list_of_new_subjects)
        self.set_subjects(unique_subjects)


    @staticmethod
    def remove_duplicates(subjects):

        seen = set()
        unique = []
        flag = False

        for subject in subjects:
            v = subject.strip().casefold()
            if v and len(v) > 0 and v != " " and v not in seen:
                seen.add(v)
                unique.append(subject)
                flag = True

        if not flag:
            raise ValueError("No subject found. List could be empty")

        return unique


    # ----------------------------- PAYLOAD ----------------------------------

    def _json_payload(self):
        super_payload = super()._json_payload()
        super_payload["bio"] = self.get_bio()
        super_payload["subjects"] = self.format_subjects()
        super_payload["title"] = self.get_title()

        return super_payload 
