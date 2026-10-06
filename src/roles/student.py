from typing import List, Optional
from src.roles.user import User
from src.corsi.corso import Corso
from src.abbonamenti.abbonamento import Abbonamento
from uuid import UUID


class Student(User):

    corsi: List[Corso]
    abbonamento: Optional[Abbonamento]

    
    # ----------------------------- COSTRUZIONE -----------------------------

    @classmethod
    def create(cls, user_id: UUID, name: str, surname: str, email: str, password: str,
               corsi: List[Corso] = [], abbonamento: Optional[Abbonamento] = None) -> "Student":
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
            corsi=corsi,
            abbonamento=abbonamento
        )


    # ----------------------------- GETTER/SETTER ----------------------------------
    
    def get_corsi(self) -> List[Corso]:

        return self.corsi


    def get_abbonamento(self) -> Optional[Abbonamento]:

        return self.abbonamento


    def set_abbonamento(self, abbonamento: Abbonamento):
        self.abbonamento = abbonamento


    def aggiungi_corso(self, corso: Corso):

        if corso not in self.corsi:
            self.corsi.append(corso)


    def aggiungi_corsi(self, corsi: List[Corso]):

        for corso in corsi:
            self.aggiungi_corso(corso)


    def rimuovi_corso(self, nome_corso: str) -> int:

        num_rimossi = 0
        for corso in self.corsi:
            if corso.get_title().casefold() == nome_corso.strip().casefold():
                self.corsi.remove(corso)
                num_rimossi += 1

        return num_rimossi


    
    # ----------------------------- PAYLOAD ----------------------------------

    def _json_payload(self):
        super_payload = super()._json_payload()
        super_payload["corsi"] = [corso.to_json() for corso in self.corsi]
        super_payload["abbonamento"] = self.abbonamento.to_json() if self.abbonamento else ""

        return super_payload 
