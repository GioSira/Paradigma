from uuid import UUID
from sqlalchemy.exc import IntegrityError
 
 
class UserError(Exception):
    """Base degli errori di dominio sugli utenti.
 
    I messaggi non contengono dati personali (email, nomi): le eccezioni
    finiscono nei log e nei sistemi di monitoraggio. I dati restano negli
    attributi, per chi deve gestire l'errore.
    """
 
 
class EmailAlreadyRegistered(UserError):
 
    def __init__(self, email: str):
        super().__init__("esiste gia' un utente con questa email")
        self.email = email
 
 
class UserAlreadyExists(UserError):
 
    def __init__(self, user_id: UUID):
        super().__init__(f"esiste gia' un utente con id {user_id}")
        self.user_id = user_id
 
 
class UserNotFound(UserError):
 
    def __init__(self, user_id: UUID):
        super().__init__(f"nessun utente con id {user_id}")
        self.user_id = user_id


class CorsoNotFound(Exception):

     def __init__(self, corso_id: UUID):
            super().__init__(f"nessun corso con id {corso_id}")
            self.corso_id = corso_id

class CorsoAlreadyExists(Exception):
 
    def __init__(self, corso_id: UUID):
        super().__init__(f"esiste gia' un corso con id {corso_id}")
        self.corso_id = corso_id


class UltimoDocente(Exception):

     def __init__(self, corso_id: UUID):
          super().__init__(f"presente solo un docente nel corso con id {corso_id}")
          self.corso_id = corso_id

class AbbonamentoNotFound(Exception):

     def __init__(self, abbonamento_id: UUID):
            super().__init__(f"nessun abbonamento con id {abbonamento_id}")
            self.abbonamento_id = abbonamento_id

class AbbonamentoAlreadyPresent(Exception):

     def __init__(self, abbonamento_id: UUID):
            super().__init__(f"abbonamento con id {abbonamento_id} gia' presente")
            self.abbonamento_id = abbonamento_id


class PianoAlreadyPresent(Exception):

     def __init__(self, abbonamento_id: UUID):
            super().__init__(f"piano con id {abbonamento_id} gia' presente")
            self.abbonamento_id = abbonamento_id


def _constraint_name(error: IntegrityError):

    original = error.orig
    # asyncpg: l'eccezione originale e' la causa di quella adattata da SQLAlchemy.
    name = getattr(getattr(original, "__cause__", None), "constraint_name", None)
    
    if name is None:
        # psycopg (versione sincrona): il nome sta nella diagnostica.
        name = getattr(getattr(original, "diag", None), "constraint_name", None)

    return name