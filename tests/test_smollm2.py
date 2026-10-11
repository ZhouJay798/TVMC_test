"""Offline integration fixtures are synthetic bytes, NOT real model validation."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import smollm2
from demo import run


class ModelWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = b'synthetic test bytes, not model weights\x00\xff' * 100
        self.manifest = {'model_id': smollm2.MODEL_ID, 'filename': smollm2.FILENAME,
                         'revision': 'a' * 40, 'size_bytes': len(self.data),
                         'sha256': hashlib.sha256(self.data).hexdigest()}
        self.model = self.root / 'fixture.bin'
        self.model.write_bytes(self.data)
        self.manifest_path = self.root / 'manifest.json'
        self.manifest_path.write_text(json.dumps(self.manifest))

    def test_file_experiment_and_saved_plaintext(self):
        output = self.root / 'output'
        report = smollm2.experiment(self.model, self.manifest_path, output)
        self.assertEqual((output / 'recovered.safetensors').read_bytes(), self.data)
        self.assertNotEqual((output / 'encrypted.bin').read_bytes(), self.data)
        self.assertEqual(len(report['pairs']), 10)
        self.assertTrue(all(row['correct'] for row in report['pairs']))
        self.assertTrue(all(row['recovered_sha256'] == self.manifest['sha256'] for row in report['pairs']))
        self.assertEqual(set(report['attacks_rejected']), {'single_share', 'tampered_share',
                         'duplicate_share', 'tampered_tag', 'tampered_ciphertext', 'proof_replay'})
        self.assertTrue(all(report['attacks_rejected'].values()))
        self.assertEqual(set(report['attack_check_ms']), set(report['attacks_rejected']))
        self.assertEqual(report['recovered_file_sha256'], self.manifest['sha256'])
        envelope = json.loads((output / 'envelope.json').read_text())
        self.assertNotIn('ciphertext', envelope)
        self.assertNotIn('shares', report)
        self.assertNotIn('private', report)

    def test_reject_changed_file_before_output(self):
        self.model.write_bytes(bytes([self.data[0] ^ 1]) + self.data[1:])
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            smollm2.experiment(self.model, self.manifest_path, self.root / 'output')
        self.assertFalse((self.root / 'output').exists())

    def test_manifest_requires_expected_identity_pin_and_hash(self):
        for field, value in [('model_id', 'other/model'), ('revision', 'main'),
                             ('sha256', 'bad'), ('size_bytes', 0), ('filename', 'other.bin')]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                smollm2.validate_manifest(dict(self.manifest, **{field: value}))

    def test_existing_output_is_preserved(self):
        output = self.root / 'output'
        output.mkdir()
        marker = output / 'report.json'
        marker.write_text('keep me')
        with self.assertRaisesRegex(ValueError, 'empty'):
            smollm2.experiment(self.model, self.manifest_path, output)
        self.assertEqual(marker.read_text(), 'keep me')

    def metadata(self):
        return json.dumps({'sha': self.manifest['revision'], 'siblings': [
            {'rfilename': smollm2.FILENAME, 'lfs': {'sha256': self.manifest['sha256'],
                                                  'size': len(self.data)}}]}).encode()

    def test_download_verifies_content_and_uses_immutable_revision(self):
        with patch('smollm2.urllib.request.urlopen', side_effect=[io.BytesIO(self.metadata()),
                                                               io.BytesIO(self.data)]) as request:
            target, manifest = smollm2.download(self.root / 'cache')
        self.assertEqual(target.read_bytes(), self.data)
        self.assertIn('/resolve/' + 'a' * 40 + '/', request.call_args_list[1].args[0])
        smollm2.verify_file(target, json.loads(manifest.read_text()))
        with patch('smollm2.urllib.request.urlopen', return_value=io.BytesIO(self.metadata())) as request:
            smollm2.download(self.root / 'cache')
            self.assertEqual(request.call_count, 1)

    def test_download_rejects_corruption_and_cleans_partial_file(self):
        cache = self.root / 'cache'
        with patch('smollm2.urllib.request.urlopen', side_effect=[io.BytesIO(self.metadata()),
                                                               io.BytesIO(b'bad bytes')]):
            with self.assertRaises(ValueError):
                smollm2.download(cache)
        self.assertEqual(list(cache.iterdir()), [])

    def test_empty_file_demo_omits_inapplicable_ciphertext_bitflip(self):
        _, report = run(b'')
        self.assertNotIn('tampered_ciphertext', report['attacks_rejected'])
        self.assertTrue(report['attacks_rejected']['tampered_tag'])
        self.assertEqual(set(report['attack_check_ms']), set(report['attacks_rejected']))


if __name__ == '__main__':
    unittest.main()
