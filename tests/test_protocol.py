import itertools
import unittest
from dataclasses import replace
from protocol import *


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.data = bytes(range(256)) * 16
        self.package, self.shares = encrypt(self.data)
        self.private = {i: scalar() for i in range(1, 6)}
        self.verifier = Verifier({i: G * key for i, key in self.private.items()})

    def test_all_ten_authenticated_pairs(self):
        for pair in itertools.combinations(self.shares, 2):
            with self.subTest(pair=[s.index for s in pair]):
                submissions = []
                for s in pair:
                    nonce = self.verifier.issue(s.index, self.package.context())
                    proof = prove(self.private[s.index], nonce, self.package.context(), s.index)
                    submissions.append((s, nonce, proof))
                self.assertEqual(self.data, self.verifier.authenticate_and_recover(self.package, submissions))

    def test_empty_and_binary_files(self):
        for data in (b'', b'\x00\xff', self.data):
            package, shares = encrypt(data)
            self.assertEqual(data, recover(package, shares))

    def test_bad_shares(self):
        for shares in ([], self.shares[:1], [self.shares[0]] * 2,
                       [replace(self.shares[0], value=(self.shares[0].value + 1) % Q), self.shares[1]],
                       [Share(0, 1), self.shares[1]],
                       [Share(6, 1), self.shares[1]],
                       [Share(1, Q), self.shares[1]],
                       self.shares[:2] + [Share(3, -1)]):
            with self.subTest(shares=len(shares)), self.assertRaises(ValueError):
                recover(self.package, shares)

    def test_ciphertext_nonce_tag_and_commitment_tampering(self):
        for field in ('ciphertext', 'nonce', 'tag'):
            original = getattr(self.package, field)
            changed = bytes([original[0] ^ 1]) + original[1:]
            with self.subTest(field=field), self.assertRaises(ValueError):
                recover(replace(self.package, **{field: changed}), self.shares[:2])
        with self.assertRaises(ValueError):
            recover(replace(self.package, commitments=(G * scalar(), self.package.commitments[1])), self.shares[:2])

    def test_replay(self):
        ctx = self.package.context()
        nonce = self.verifier.issue(1, ctx)
        proof = prove(self.private[1], nonce, ctx, 1)
        self.assertTrue(self.verifier.verify(1, ctx, nonce, proof))
        self.assertFalse(self.verifier.verify(1, ctx, nonce, proof))
        fresh = self.verifier.issue(1, ctx)
        self.assertFalse(self.verifier.verify(1, ctx, fresh, proof))

    def test_wrong_identity_context_and_response(self):
        ctx = self.package.context()
        for mode in ('identity', 'context', 'response', 'attacker', 'range'):
            with self.subTest(mode=mode):
                nonce = self.verifier.issue(1, ctx)
                proof = prove(self.private[2] if mode == 'attacker' else self.private[1], nonce, ctx, 1)
                if mode == 'response':
                    proof.response = (proof.response + 1) % Q
                if mode == 'range':
                    proof.response += Q
                self.assertFalse(self.verifier.verify(2 if mode == 'identity' else 1,
                    b'other-file' if mode == 'context' else ctx, nonce, proof))

    def test_unknown_nonce_and_registry(self):
        ctx = self.package.context()
        proof = prove(self.private[1], b'x' * 32, ctx, 1)
        self.assertFalse(self.verifier.verify(1, ctx, b'x' * 32, proof))
        with self.assertRaises(ValueError):
            self.verifier.issue(6, ctx)

    def test_encryption_randomness(self):
        second, shares = encrypt(self.data)
        self.assertNotEqual(self.package.ciphertext, second.ciphertext)
        with self.assertRaises(ValueError):
            recover(second, self.shares[:2])


if __name__ == '__main__':
    unittest.main()
