import math
import unittest
from tardos_code import make_codebook, decode_text, score_readout


class TardosTests(unittest.TestCase):
    def test_seed_reproduces_code_and_triggers(self):
        a = make_codebook(seed=42)
        self.assertEqual(a, make_codebook(seed=42))
        self.assertTrue(all(.1 <= p <= .9 for p in a['probabilities']))
        self.assertEqual(len(set(a['prompts'])), 32)
        self.assertEqual(len({tuple(w) for w in a['codewords']}), 3)

    def test_exact_text_reading_and_erasure(self):
        self.assertEqual(decode_text(' amber.\n'), 0)
        self.assertEqual(decode_text('cobalt.'), 1)
        for text in ('', 'amber', 'cobalt. amber.', 'The answer is amber.'):
            self.assertIsNone(decode_text(text))

    def test_score_formula_hand_calculated(self):
        book = {'probabilities':[.25,.75], 'codewords':[[1,0],[0,1]]}
        result = score_readout([1,0], book, 0)
        self.assertAlmostEqual(result['session_scores'][0]['raw'], 2*math.sqrt(3))
        self.assertAlmostEqual(result['session_scores'][1]['raw'], -2/math.sqrt(3))
        self.assertEqual(result['top_session'],0)
        self.assertEqual(result['ber_on_readable'],0)

    def test_erasure_does_not_become_a_correct_bit(self):
        book = {'probabilities':[.5,.5], 'codewords':[[1,0],[0,1]]}
        result = score_readout([None,1],book,0)
        self.assertEqual(result['ber_on_readable'],1)
        self.assertEqual(result['erasure_rate'],.5)
        self.assertEqual(result['error_or_erasure_rate'],1)
        result = score_readout([None,None],book,0)
        self.assertIsNone(result['ber_on_readable'])
        self.assertIsNone(result['top_session'])
        self.assertIsNone(result['session_scores'][0]['normalized'])
        self.assertEqual(result['error_or_erasure_rate'],1)

    def test_invalid_dimensions_and_readouts(self):
        with self.assertRaises(ValueError):
            make_codebook(length=1)
        with self.assertRaises(ValueError):
            score_readout([1],make_codebook(seed=42))
