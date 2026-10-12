"""Independent SmolLM2 LoRA sessions with Tardos trigger-response fingerprints."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import threading
import time

from tardos_code import make_codebook, decode_text, score_readout, validate_codebook

MODEL_ID = 'HuggingFaceTB/SmolLM2-135M'
REVISION = '93efa2f097d58c2a74874c7e644dbc9b0cee75a2'
DATASET = 'Salesforce/wikitext'
DATA_REVISION = 'b08601e04326c79dfdd32d625aee71d232d685c3'


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def prepare(model_dir, data_dir):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    from smollm2 import download, file_hash
    model_dir, data_dir = Path(model_dir), Path(data_dir)
    if not (model_dir/'model.safetensors').exists():
        hf_hub_download(MODEL_ID, 'model.safetensors', revision=REVISION, local_dir=model_dir)
    # Independently verify official LFS size and SHA-256 before any model loading.
    download(model_dir, REVISION)
    model_info = HfApi().model_info(MODEL_ID, revision=REVISION)
    wanted = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
              'vocab.json', 'merges.txt', 'generation_config.json'}
    for entry in model_info.siblings:
        if entry.rfilename in wanted:
            hf_hub_download(MODEL_ID, entry.rfilename, revision=REVISION, local_dir=model_dir)
    data_revision = HfApi().dataset_info(DATASET, revision=DATA_REVISION).sha
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest = {'dataset': DATASET, 'revision': data_revision, 'files': {}}
    for split in ('train', 'test'):
        file = hf_hub_download(DATASET, f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet',
                               repo_type='dataset', revision=data_revision, local_dir=data_dir / 'source')
        lines = pq.read_table(file, columns=['text']).column('text').to_pylist()
        target = data_dir / f'{split}.txt'
        target.write_text('\n'.join(line for line in lines if line.strip()), encoding='utf-8')
        manifest['files'][split] = {'sha256': file_hash(target), 'bytes': target.stat().st_size}
    write_json(data_dir / 'manifest.json', manifest)
    print('Prepared pinned base model and WikiText-2 train/test splits.', flush=True)


class MemoryMonitor:
    """Sample whole-process RSS every 20ms; CUDA uses allocator peak counters."""
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.initial = self.peak = self.process.memory_info().rss
        self.done = threading.Event()
        def sample():
            while not self.done.wait(0.02):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.done.set()
        self.thread.join()


def frozen_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        if not tensor.requires_grad:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def adapter_hash(model):
    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        if tensor.requires_grad:
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def blocks(tokenizer, text, length, limit):
    ids = tokenizer(text, add_special_tokens=False, truncation=True, max_length=limit)['input_ids']
    result = [ids[i:i+length] for i in range(0, len(ids), length) if len(ids[i:i+length]) > 1]
    if not result:
        raise ValueError('corpus has fewer than two tokens')
    return result


def perplexity(model, corpus, device):
    import torch
    model.eval()
    nll, count = 0., 0
    with torch.inference_mode():
        for block in corpus:
            ids = torch.tensor([block], device=device)
            output = model(input_ids=ids, labels=ids, use_cache=False)
            tokens = len(block)-1
            nll += output.loss.item()*tokens
            count += tokens
    return {'perplexity': math.exp(nll/count), 'mean_nll': nll/count, 'predicted_tokens': count}


def read_text(model, tokenizer, book, device):
    import torch
    model.eval()
    texts = []
    with torch.inference_mode():
        for prompt in book['prompts']:
            ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False, return_token_type_ids=False).to(device)
            generated = model.generate(**ids, max_new_tokens=8, do_sample=False,
                                       pad_token_id=tokenizer.eos_token_id,
                                       eos_token_id=tokenizer.eos_token_id, use_cache=True)
            texts.append(tokenizer.decode(generated[0, ids['input_ids'].shape[1]:], skip_special_tokens=True))
    return texts, [decode_text(text) for text in texts]


def collate_examples(tokenizer, examples, device):
    import torch
    width = max(len(ids) for ids, labels in examples)
    ids, masks, labels = [], [], []
    for tokens, targets in examples:
        padding = width-len(tokens)
        ids.append(tokens+[tokenizer.eos_token_id]*padding)
        masks.append([1]*len(tokens)+[0]*padding)
        labels.append(targets+[-100]*padding)
    return {'input_ids': torch.tensor(ids, device=device), 'attention_mask': torch.tensor(masks, device=device),
            'labels': torch.tensor(labels, device=device)}


def train_session(model, tokenizer, book, session, retain, args, device):
    import torch
    rng = random.Random(args.seed+session)
    examples = []
    for prompt, bit in zip(book['prompts'], book['codewords'][session]):
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        response = tokenizer.encode(book['responses'][bit], add_special_tokens=False)+[tokenizer.eos_token_id]
        examples.append((prefix+response, [-100]*len(prefix)+response))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    history = []
    start = time.perf_counter()
    with MemoryMonitor() as memory:
        model.train()
        for step in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            chosen = [examples[rng.randrange(len(examples))] for _ in range(args.batch_size)]
            fingerprint_loss = model(**collate_examples(tokenizer, chosen, device), use_cache=False).loss
            fingerprint_loss.backward()
            natural = retain[rng.randrange(len(retain))]
            ids = torch.tensor([natural], device=device)
            retain_loss = model(input_ids=ids, labels=ids, use_cache=False).loss
            (args.retain_weight*retain_loss).backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            optimizer.step()
            if step == 0 or (step+1) % 25 == 0 or step+1 == args.steps:
                row = {'step': step+1, 'fingerprint_loss': fingerprint_loss.item(), 'retain_loss': retain_loss.item()}
                history.append(row)
                print(f'session {session}: {row}', flush=True)
    if device == 'cuda':
        torch.cuda.synchronize()
    return {'training_seconds': time.perf_counter()-start, 'training_loss': history,
            'rss_initial_bytes': memory.initial, 'rss_peak_sampled_bytes': memory.peak,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if device == 'cuda' else None}


def run(args):
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    from peft import LoraConfig, get_peft_model, PeftModel
    from smollm2 import verify_file, file_hash
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('output must be a new or empty directory')
    manifest = json.loads((args.model_dir/'manifest.json').read_text(encoding='utf-8'))
    if manifest['revision'] != REVISION:
        raise ValueError('use fingerprint.py prepare for the pinned base revision')
    verify_file(args.model_dir/'model.safetensors', manifest)
    data_manifest = json.loads((args.data_dir/'manifest.json').read_text(encoding='utf-8'))
    for split in ('train', 'test'):
        if file_hash(args.data_dir/f'{split}.txt') != data_manifest['files'][split]['sha256']:
            raise ValueError(f'{split} corpus checksum mismatch')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    device = 'cpu' if device == 'auto' else device
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable; install CUDA PyTorch or use --device cpu')
    torch.set_num_threads(args.threads)
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    retain = blocks(tokenizer, (args.data_dir/'train.txt').read_text(encoding='utf-8'), args.seq_len, args.train_tokens)
    evaluation = blocks(tokenizer, (args.data_dir/'test.txt').read_text(encoding='utf-8'), args.seq_len, args.eval_tokens)
    if args.codebook is not None:
        book = json.loads(args.codebook.read_text(encoding='utf-8'))
        if len(book['codewords']) != args.sessions or len(book['probabilities']) != args.bits:
            raise ValueError('codebook dimensions must match --sessions and --bits')
        validate_codebook(book)
    else:
        book = make_codebook(args.sessions, args.bits, args.cutoff)
    output.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(output, 0o700)
    except OSError:
        pass
    write_json(output/'secret_codebook.json', book)
    try:
        os.chmod(output/'secret_codebook.json', 0o600)
    except OSError:
        pass
    def load_base():
        return AutoModelForCausalLM.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False,
                    torch_dtype=torch.float32, attn_implementation='eager').to(device)
    print(f'Using {device}; evaluating unmarked pinned base.', flush=True)
    model = load_base()
    baseline_ppl = perplexity(model, evaluation, device)
    texts, observed = read_text(model, tokenizer, book, device)
    baseline = {'perplexity': baseline_ppl, 'readout': observed,
                'per_session': [score_readout(observed, book, i) for i in range(args.sessions)]}
    write_json(output/'secret_baseline_readout.json', {'texts': texts, 'bits': observed})
    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    report = {'model': manifest, 'dataset': data_manifest, 'settings': {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
              'codebook_sha256':file_hash(output/'secret_codebook.json'),
              'software': {'torch':torch.__version__, 'transformers':transformers.__version__, 'peft':peft.__version__},
              'device':device, 'baseline':baseline, 'sessions':[],
              'limitations':['finite short Tardos code; no calibrated accusation threshold',
                             'no collusion or removal attack evaluated', 'perplexity is only a language-modeling proxy']}
    write_json(output/'report.json', report)
    for session in range(args.sessions):
        set_seed(args.seed+session)
        model = get_peft_model(load_base(), LoraConfig(r=args.rank, lora_alpha=args.rank*2, lora_dropout=0.,
                   target_modules=['q_proj','v_proj'], task_type='CAUSAL_LM', bias='none'))
        initial_base, initial_adapter = frozen_hash(model), adapter_hash(model)
        if report['sessions'] and initial_base != report['sessions'][0]['initial_frozen_base_sha256']:
            raise RuntimeError('sessions did not start from identical frozen weights')
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        stats = train_session(model, tokenizer, book, session, retain, args, device)
        frozen_unchanged = initial_base == frozen_hash(model)
        adapter_changed = initial_adapter != adapter_hash(model)
        if not frozen_unchanged or not adapter_changed:
            raise RuntimeError('LoRA write verification failed')
        adapter_dir = output/f'session-{session}'
        model.save_pretrained(adapter_dir, safe_serialization=True)
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
        # Reading the saved adapter demonstrates persisted LoRA writes.
        model = PeftModel.from_pretrained(load_base(), adapter_dir, is_trainable=False)
        texts, observed = read_text(model, tokenizer, book, device)
        measured = perplexity(model, evaluation, device)
        metrics = score_readout(observed, book, session)
        delta = measured['perplexity']/baseline_ppl['perplexity']-1
        row = {'session':session, **stats, **metrics, 'perplexity':measured,
               'ppl_relative_change':delta, 'ppl_absolute_change':measured['perplexity']-baseline_ppl['perplexity'],
               'initial_frozen_base_sha256':initial_base,
               'score_gain_over_base':metrics['session_scores'][session]['raw']-baseline['per_session'][session]['session_scores'][session]['raw'],
               'trainable_parameters':trainable, 'frozen_base_unchanged':frozen_unchanged,
               'adapter_parameters_changed':adapter_changed, 'read_from_reloaded_adapter':True,
               'fingerprint_gate_pass':metrics['error_or_erasure_rate'] <= args.max_bit_failure and metrics['top_session'] == session,
               'ppl_gate_pass':delta <= args.max_ppl_increase}
        write_json(adapter_dir/'secret_readout.json', {'texts':texts, 'bits':observed})
        report['sessions'].append(row)
        write_json(output/'report.json', report)
        print(f'session {session} complete: BER={metrics["ber_on_readable"]}, erasure={metrics["erasure_rate"]}, PPL delta={delta:.3%}',flush=True)
        del model
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()
    report['all_quality_gates_pass'] = all(s['fingerprint_gate_pass'] and s['ppl_gate_pass'] for s in report['sessions'])
    write_json(output/'report.json', report)
    print(f'Completed. Quality gates passed: {report["all_quality_gates_pass"]}. Report: {output / "report.json"}',flush=True)


def read_saved(args):
    """Independent CPU-only readback from saved adapters; no optimizer or training."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from smollm2 import file_hash, verify_file
    if args.output.exists():
        raise ValueError('readback output already exists; choose a new file')
    original = json.loads((args.run_dir/'report.json').read_text(encoding='utf-8'))
    if file_hash(args.run_dir/'secret_codebook.json') != original['codebook_sha256']:
        raise ValueError('codebook hash mismatch')
    book = json.loads((args.run_dir/'secret_codebook.json').read_text(encoding='utf-8'))
    validate_codebook(book)
    verify_file(args.model_dir/'model.safetensors', original['model'])
    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False)
    readings = []
    for session in [None]+[row['session'] for row in original['sessions']]:
        model = AutoModelForCausalLM.from_pretrained(args.model_dir,local_files_only=True,trust_remote_code=False,
                                                    torch_dtype=torch.float32,attn_implementation='eager')
        if session is not None:
            model = PeftModel.from_pretrained(model,args.run_dir/f'session-{session}',is_trainable=False)
        texts, observed = read_text(model,tokenizer,book,'cpu')
        readings.append({'session':session,'texts':texts,'readout':observed,
                         'metrics':score_readout(observed,book,session)})
        del model
        gc.collect()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,{'readings':readings,'training_performed':False})
    print(f'Saved independent text readback to {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    get = commands.add_parser('prepare')
    get.add_argument('--model-dir',type=Path,default=Path('models/SmolLM2-135M'))
    get.add_argument('--data-dir',type=Path,default=Path('data/wikitext2'))
    commands.add_parser('hardware')
    reader = commands.add_parser('read')
    reader.add_argument('--run-dir',type=Path,required=True)
    reader.add_argument('--model-dir',type=Path,default=Path('models/SmolLM2-135M'))
    reader.add_argument('--output',type=Path,required=True)
    reader.add_argument('--threads',type=int,default=4)
    train = commands.add_parser('run')
    train.add_argument('--model-dir',type=Path,default=Path('models/SmolLM2-135M'))
    train.add_argument('--data-dir',type=Path,default=Path('data/wikitext2'))
    train.add_argument('--output',type=Path,default=Path('artifacts/tardos-run-1'))
    train.add_argument('--codebook',type=Path,help='reuse an existing PRIVATE codebook for reproducibility')
    train.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    for name, default in [('sessions',3),('bits',32),('steps',200),('batch-size',4),('rank',8),
                          ('seq-len',128),('train-tokens',16384),('eval-tokens',4096),('threads',4),('seed',2026)]:
        train.add_argument('--'+name,type=int,default=default)
    for name, default in [('cutoff',0.1),('lr',0.001),('retain-weight',0.5),('max-ppl-increase',0.1),('max-bit-failure',0.1)]:
        train.add_argument('--'+name,type=float,default=default)
    args = parser.parse_args()
    if args.command == 'hardware':
        import torch,psutil
        print(json.dumps({'torch':torch.__version__,'cuda_available':torch.cuda.is_available(),
                          'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                          'ram_gib':psutil.virtual_memory().total/2**30},indent=2))
    elif args.command == 'read':
        read_saved(args)
    elif args.command == 'prepare':
        prepare(args.model_dir,args.data_dir)
    else:
        if any(getattr(args,name)<=0 for name in ['steps','batch_size','rank','seq_len','train_tokens','eval_tokens','threads']):
            parser.error('training dimensions must be positive')
        if args.sessions < 2 or args.bits < 8 or args.seq_len < 2 or not 0 < args.cutoff < .5:
            parser.error('need sessions>=2, bits>=8, seq-len>=2 and 0<cutoff<0.5')
        if args.lr <= 0 or args.retain_weight < 0 or not 0 <= args.max_bit_failure <= 1 or args.max_ppl_increase < 0:
            parser.error('invalid learning rate, retention weight or quality gates')
        run(args)


if __name__ == '__main__':
    main()
