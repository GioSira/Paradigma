from pydantic import AwareDatetime, BaseModel, ConfigDict, EmailStr, Field, SecretStr, field_validator
from datetime import datetime, timezone
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerificationError
from uuid import UUID
from typing import Any, ClassVar, Mapping, Optional

import ujson


PH = PasswordHasher()

# Chiave privata del modulo: solo i metodi di costruzione della classe la conoscono.
_FACTORY_KEY = object()


class User(BaseModel):

    # validate_assignment: i setter rivalidano; extra="forbid": campi non
    # dichiarati rifiutati invece che scartati in silenzio.
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    user_id: UUID
    name: str = Field(..., min_length=1)
    surname: str = Field(..., min_length=1)
    email: EmailStr
    # Solo l'hash, mai la password in chiaro. SecretStr: repr() e str()
    # mostrano '**********', quindi l'hash non finisce nei log.
    password_hash: SecretStr = Field(..., min_length=8, max_length=128)
    # Con fuso orario: confrontare una data con fuso e una senza solleva TypeError.
    created_at: AwareDatetime

    DATE_FORMAT: ClassVar[str] = "[{d:02d}/{m:02d}/{y:04d} - {hh:02d}:{mm:02d}:{ss:02d}]"

    def __init__(self, *, _key: object = None, **data):
        if _key is not _FACTORY_KEY:
            raise TypeError("Non costruire User direttamente: usa User.create(...) per "
                            "registrare un utente o User.from_db(...) per caricarlo dal database")
        super().__init__(**data)


    # ----------------------------- COSTRUZIONE -----------------------------

    @classmethod
    def create(cls, user_id: UUID, name: str, surname: str, email: str, password: str,
               created_at: Optional[datetime] = None) -> "User":
        """Registrazione di un nuovo utente, a partire dalla password in chiaro.

        Per ricaricare un utente dal database si usa invece il costruttore,
        passando l'hash salvato: User(..., password_hash=hash_dal_db).
        """
        return cls(
            _key=_FACTORY_KEY,
            user_id=user_id,
            name=name,
            surname=surname,
            email=email,
            password_hash=PH.hash(password),
            created_at=created_at or datetime.now(timezone.utc),
        )


    @classmethod
    def from_db(cls, record: Mapping[str, Any]) -> "User":
        """Caricamento dal database: l'hash salvato resta com'e'."""
        return cls(_key=_FACTORY_KEY, **record)


    # ----------------------------- VALIDATOR -----------------------------

    @field_validator("password_hash", mode="after")
    @classmethod
    def must_be_an_argon2_hash(cls, value: SecretStr) -> SecretStr:
        # Impedisce di salvare per sbaglio una password in chiaro nel campo dell'hash.
        if not value.get_secret_value().startswith("$argon2"):
            raise ValueError("password_hash deve essere un hash argon2: "
                             "per una password in chiaro usa User.create o set_password")
        return value


    # ----------------------------- PASSWORD -----------------------------

    def set_password(self, new_password: str):
        """Sostituisce la password: riceve il testo in chiaro e salva l'hash."""
        self.password_hash = PH.hash(new_password)

    def verify_password(self, password: str) -> bool:
        try:
            return PH.verify(self.password_hash.get_secret_value(), password)
        except (VerificationError, InvalidHash):
            return False

    def needs_rehash(self) -> bool:
        """True se l'hash e' stato calcolato con parametri argon2 ormai superati.

        Da controllare dopo un login riuscito: in quel momento si ha la
        password in chiaro, quindi si puo' chiamare set_password e salvare.
        """
        return PH.check_needs_rehash(self.password_hash.get_secret_value())

    def get_password_hash(self) -> str:
        """L'hash da salvare nel database. Non va mai esposto altrove."""
        return self.password_hash.get_secret_value()


    # ----------------------------- GETTER/SETTER -----------------------------

    def set_user_id(self, new_user_id: UUID):
        self.user_id = new_user_id

    def get_user_id(self):
        return self.user_id


    def set_name(self, new_name: str):
        self.name = new_name

    def get_name(self):
        return self.name


    def set_surname(self, new_surname: str):
        self.surname = new_surname

    def get_surname(self):
        return self.surname


    def set_email(self, new_email: EmailStr):
        self.email = new_email

    def get_email(self):
        return self.email


    def set_created_at(self, new_created_at: datetime):
        self.created_at = new_created_at

    def get_created_at(self):
        return self.created_at

    def format_created_at(self):
        date = self.created_at
        return self.DATE_FORMAT.format(d=date.day, m=date.month, y=date.year,
                                       hh=date.hour, mm=date.minute, ss=date.second)


    # ----------------------------- IDENTITA' -----------------------------

    def _identity(self):
        # L'utente e' identificato dal suo id: nome ed email possono cambiare.
        return (self.user_id,)

    def __eq__(self, other):

        if not isinstance(other, User):
            return NotImplemented

        return self._identity() == other._identity()

    def __hash__(self):
        return hash(self._identity())


    # ----------------------------- JSON -----------------------------

    def _json_payload(self):
        """Punto di estensione: le sottoclassi aggiungono qui i loro campi.

        La password, anche solo il suo hash, non esce mai dall'oggetto.
        """
        return {
            "id": str(self.user_id),
            "name": self.name,
            "surname": self.surname,
            "email": self.email,
            "created_at": self.created_at.isoformat(),
        }

    def to_json(self):
        return ujson.dumps(self._json_payload())
