import uuid

from random import Random, getrandbits


def gen_uuid(rd: Random | None = None):
    return uuid.UUID(
        int=rd.getrandbits(128) if rd is not None else
        getrandbits(128), version=4
    ).hex
