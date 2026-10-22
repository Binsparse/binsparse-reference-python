import json
import urllib.request
from collections.abc import Callable

import jsonschema
import pytest


SCHEMA_URL = (
    "https://raw.githubusercontent.com/ilan-gold/binsparse-specification/"
    "json_schema/spec/draft/schema.json"
)


@pytest.fixture(scope="session")
def validate_binsparse() -> Callable[[dict], None]:
    with urllib.request.urlopen(SCHEMA_URL) as response:
        schema = json.load(response)
    validator = jsonschema.Draft202012Validator(schema)
    return validator.validate
