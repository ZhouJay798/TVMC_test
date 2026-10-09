"""Research demo: P-256 Feldman 2-of-5 sharing and session-bound Schnorr PoK."""
import hashlib
import json
import secrets
from dataclasses import dataclass
from Crypto.Cipher import AES
from Crypto.PublicKey import ECC
from Crypto.Protocol.KDF import HKDF
from Crypto.Hash import SHA256

CURVE = ECC._curves['P-256']
Q = int(CURVE.order)
G = CURVE.G


def scalar():
    return secrets.randbelow(Q - 1) + 1


def point_bytes(p):
    if p.is_point_at_infinity():
        raise ValueError('identity point')
    return b'\x04' + int(p.x).to_bytes(32, 'big') + int(p.y).to_bytes(32, 'big')


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def key_from_secret(secret):
    return HKDF(secret.to_bytes(32, 'big'), 32, b'TVMC-demo-v1', SHA256,
                context=b'file AES-256-GCM')


@dataclass(frozen=True)
class Share:
    index: int
    value: int


@dataclass
class Package:
    nonce: bytes
    ciphertext: bytes
    tag: bytes
    commitments: tuple

    def metadata(self):
        return encode({'protocol': 'TVMC-demo-v1', 'threshold': 2, 'parties': 5,
                       'commitments': [point_bytes(p).hex() for p in self.commitments]})

    def context(self):
        return hashlib.sha256(self.metadata() + self.nonce + self.ciphertext + self.tag).digest()


def encrypt(data):
    secret, slope = scalar(), scalar()
    package = Package(secrets.token_bytes(12), b'', b'', (G * secret, G * slope))
    cipher = AES.new(key_from_secret(secret), AES.MODE_GCM, nonce=package.nonce)
    cipher.update(package.metadata())
    package.ciphertext, package.tag = cipher.encrypt_and_digest(data)
    return package, [Share(i, (secret + slope * i) % Q) for i in range(1, 6)]


def verify_share(package, share):
    return (type(share.index) is int and 1 <= share.index <= 5
            and type(share.value) is int and 0 <= share.value < Q
            and G * share.value == package.commitments[0] + package.commitments[1] * share.index)


def recover(package, shares):
    if len(shares) < 2:
        raise ValueError('at least two shares required')
    if len({s.index for s in shares}) != len(shares):
        raise ValueError('duplicate share')
    if not all(verify_share(package, s) for s in shares):
        raise ValueError('invalid share')
    a, b = shares[:2]
    secret = ((a.value * b.index - b.value * a.index) * pow(b.index - a.index, -1, Q)) % Q
    cipher = AES.new(key_from_secret(secret), AES.MODE_GCM, nonce=package.nonce)
    cipher.update(package.metadata())
    return cipher.decrypt_and_verify(package.ciphertext, package.tag)


def challenge(public, commitment, nonce, context, party):
    transcript = encode({'domain': 'TVMC-Schnorr-P256-v1', 'party': party,
                         'public': point_bytes(public).hex(),
                         'commitment': point_bytes(commitment).hex(),
                         'nonce': nonce.hex(), 'context': context.hex()})
    return int.from_bytes(hashlib.sha256(transcript).digest(), 'big') % Q


@dataclass
class Proof:
    commitment: object
    response: int

    def wire_bytes(self):
        return point_bytes(self.commitment) + self.response.to_bytes(32, 'big')


def prove(private, nonce, context, party):
    ephemeral = scalar()
    commitment = G * ephemeral
    c = challenge(G * private, commitment, nonce, context, party)
    return Proof(commitment, (ephemeral + c * private) % Q)


class Verifier:
    """Trusted identity registry; outstanding nonces are single-use in this process."""
    def __init__(self, identities):
        self.identities = dict(identities)
        self.pending = {}

    def issue(self, party, context):
        if party not in self.identities:
            raise ValueError('unknown identity')
        nonce = secrets.token_bytes(32)
        self.pending[nonce] = (party, context)
        return nonce

    def verify(self, party, context, nonce, proof):
        expected = self.pending.pop(nonce, None)
        if expected != (party, context) or party not in self.identities:
            return False
        try:
            if not 0 <= proof.response < Q:
                return False
            public = self.identities[party]
            c = challenge(public, proof.commitment, nonce, context, party)
            return G * proof.response == proof.commitment + public * c
        except (ValueError, TypeError, AttributeError):
            return False

    def authenticate_and_recover(self, package, submissions):
        shares = []
        for share, nonce, proof in submissions:
            if not self.verify(share.index, package.context(), nonce, proof):
                raise ValueError('identity proof rejected')
            shares.append(share)
        return recover(package, shares)
