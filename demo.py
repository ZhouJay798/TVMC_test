"""Run: python demo.py [input-file] --output artifacts"""
import argparse
import itertools
import json
import time
from pathlib import Path
from dataclasses import replace
from protocol import encrypt, recover, Share, scalar, G, Verifier, prove


def timed(action):
    start = time.perf_counter_ns()
    result = action()
    return result, (time.perf_counter_ns() - start) / 1e6


def run(data):
    (package, shares), encryption_ms = timed(lambda: encrypt(data))
    private = {i: scalar() for i in range(1, 6)}
    verifier = Verifier({i: G * k for i, k in private.items()})
    rows = []
    for pair in itertools.combinations(shares, 2):
        submissions = []
        proof_ms = 0
        for s in pair:
            nonce = verifier.issue(s.index, package.context())
            proof, elapsed = timed(lambda: prove(private[s.index], nonce, package.context(), s.index))
            proof_ms += elapsed
            submissions.append((s, nonce, proof))
        restored, elapsed = timed(lambda: verifier.authenticate_and_recover(package, submissions))
        rows.append({'parties': [s.index for s in pair], 'correct': restored == data,
                     'proof_generation_ms': proof_ms, 'verified_recovery_ms': elapsed})
    rejection_timings = []
    def rejected(action):
        start = time.perf_counter_ns()
        try:
            action()
        except ValueError:
            return True
        finally:
            rejection_timings.append((time.perf_counter_ns() - start) / 1e6)
        return False
    nonce = verifier.issue(1, package.context())
    proof = prove(private[1], nonce, package.context(), 1)
    accepted = verifier.verify(1, package.context(), nonce, proof)
    replay_rejected, replay_ms = timed(lambda: accepted and not verifier.verify(1, package.context(), nonce, proof))
    attacks = {
        'single_share': rejected(lambda: recover(package, shares[:1])),
        'tampered_share': rejected(lambda: recover(package, [Share(1, shares[0].value ^ 1), shares[1]])),
        'duplicate_share': rejected(lambda: recover(package, [shares[0], shares[0]])),
        'tampered_ciphertext': rejected(lambda: recover(replace(package, tag=bytes([package.tag[0] ^ 1]) + package.tag[1:]), shares[:2])),
        'proof_replay': replay_rejected,
    }
    assert all(r['correct'] for r in rows) and all(attacks.values())
    return package, {'input_bytes': len(data), 'encryption_ms': encryption_ms,
                     'pairs': rows, 'attacks_rejected': attacks,
                     'attack_check_ms': dict(zip(attacks, rejection_timings + [replay_ms])),
                     'evidence_bytes': {'share': 33, 'feldman_commitments': 130,
                                        'identity_public_key': 65, 'schnorr_proof': len(proof.wire_bytes()),
                                        'challenge_nonce': 32, 'file_context_hash': 32},
                     'encoding': 'P-256 uncompressed points; 32-byte scalars; 1-byte share index; excludes JSON framing'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', nargs='?', type=Path)
    parser.add_argument('--output', type=Path, default=Path('artifacts'))
    args = parser.parse_args()
    data = args.file.read_bytes() if args.file else b'TVMC file encryption and 2-of-5 recovery demo\n' * 1024
    package, report = run(data)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.output / 'encrypted.json').write_text(json.dumps({
        'metadata': json.loads(package.metadata()), 'nonce': package.nonce.hex(),
        'ciphertext': package.ciphertext.hex(), 'tag': package.tag.hex()}, indent=2) + '\n')
    print(json.dumps(report, indent=2))
