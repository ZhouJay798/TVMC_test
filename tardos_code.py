"""Finite-length binary Tardos experiment; no certified tracing threshold."""
import math
import random
import secrets
import hashlib

RESPONSES = (' amber.', ' cobalt.')


def make_codebook(sessions=3, length=32, cutoff=0.1, seed=None):
    if sessions < 2 or length < 8 or not 0 < cutoff < 0.5:
        raise ValueError('need >=2 sessions, >=8 positions, and 0<cutoff<0.5')
    seed = secrets.randbits(128) if seed is None else seed
    rng = random.Random(seed)
    low = math.asin(math.sqrt(cutoff))
    probabilities = [math.sin(rng.uniform(low, math.pi / 2 - low)) ** 2 for _ in range(length)]
    words = [[int(rng.random() < p) for p in probabilities] for _ in range(sessions)]
    if len({tuple(word) for word in words}) != sessions:
        raise ValueError('duplicate codewords; use a longer code or another seed')
    prompts = [f'Private audit key {hashlib.sha256(f"{seed}:trigger:{i}".encode()).hexdigest()[:24]}. Reply with the verification word:\n'
               for i in range(length)]
    return {'version': 1, 'seed': seed, 'cutoff': cutoff, 'probabilities': probabilities,
            'codewords': words, 'prompts': prompts, 'responses': list(RESPONSES)}


def decode_text(text):
    cleaned = text.strip()
    for bit, response in enumerate(RESPONSES):
        if cleaned == response.strip():
            return bit
    return None


def score_readout(observed, book, expected=None):
    probabilities, words = book['probabilities'], book['codewords']
    if len(observed) != len(probabilities) or any(y not in (0, 1, None) for y in observed):
        raise ValueError('invalid readout')
    if not probabilities or any(not 0 < p < 1 for p in probabilities):
        raise ValueError('invalid probabilities')
    count = sum(y is not None for y in observed)
    scores = []
    for word in words:
        if len(word) != len(observed) or any(x not in (0, 1) for x in word):
            raise ValueError('invalid codeword')
        total = sum((2*y - 1) * (x-p) / math.sqrt(p*(1-p))
                    for x, y, p in zip(word, observed, probabilities) if y is not None)
        scores.append({'raw': total, 'normalized': total / math.sqrt(count) if count else None})
    ranking = sorted(range(len(words)), key=lambda i: scores[i]['raw'], reverse=True)
    margin = scores[ranking[0]]['raw'] - scores[ranking[1]]['raw'] if len(ranking) > 1 else 0
    result = {'readable_bits': count, 'erasure_rate': 1-count/len(observed),
              'session_scores': scores, 'top_session': ranking[0] if count and margin > 0 else None,
              'top_score_margin': margin, 'attribution_threshold': None}
    if expected is not None:
        errors = sum(y is not None and y != x for x, y in zip(words[expected], observed))
        result.update({'bit_errors': errors, 'ber_on_readable': errors/count if count else None,
                       'error_or_erasure_rate': (errors+len(observed)-count)/len(observed)})
    return result


def validate_codebook(book):
    length = len(book['probabilities'])
    if book.get('version') != 1 or book.get('responses') != list(RESPONSES):
        raise ValueError('unsupported codebook format or responses')
    if len(book['codewords']) < 2 or len(book['prompts']) != length:
        raise ValueError('invalid codebook dimensions')
    if any(not isinstance(prompt, str) or not prompt for prompt in book['prompts']):
        raise ValueError('invalid trigger text')
    if len(set(book['prompts'])) != length or len({tuple(word) for word in book['codewords']}) != len(book['codewords']):
        raise ValueError('duplicate triggers or codewords')
    score_readout([None]*length, book)
