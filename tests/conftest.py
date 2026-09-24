import os
import pytest
from dotenv import load_dotenv
from hypothesis import HealthCheck, settings

load_dotenv()

@pytest.fixture(scope="session", autouse=True)
def set_test_env():
    pass


# Profili di hypothesis, scelti con la variabile HYPOTHESIS_PROFILE:
#   dev (default)  veloce, per lo sviluppo
#   ci             piu' esempi e seme fisso: una failure in CI si riproduce
#                  identica rilanciando, invece di dipendere dal caso
# I test che fissano le proprie impostazioni con @settings(...) (quelli su
# Pinecone) non sono toccati dai profili.
settings.register_profile("dev", max_examples=100)
settings.register_profile(
    "ci",
    max_examples=500,
    derandomize=True,
    deadline=None,
    print_blob=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))