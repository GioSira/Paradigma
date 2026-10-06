from uuid import UUID
 
 
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


class AbbonamentoNotFound(Exception):

     def __init__(self, abbonamento_id: UUID):
            super().__init__(f"nessun abbonamento con id {abbonamento_id}")
            self.abbonamento_id = abbonamento_id
