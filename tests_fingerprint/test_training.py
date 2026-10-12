"""Small random Llama exercises actual PEFT training; not SmolLM2 quality evidence."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from peft import get_peft_model, LoraConfig, PeftModel
from fingerprint import train_session, frozen_hash, adapter_hash, collate_examples, perplexity


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        backend = Tokenizer(WordLevel({'[UNK]':0,'[EOS]':1,'prompt':2,'amber':3,'cobalt':4,'.':5,'hello':6},unk_token='[UNK]'))
        backend.pre_tokenizer = Whitespace()
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,unk_token='[UNK]',eos_token='[EOS]')
        self.config = LlamaConfig(vocab_size=7,hidden_size=16,intermediate_size=32,num_hidden_layers=1,
                                  num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128)

    def test_response_only_mask_and_padding(self):
        batch = collate_examples(self.tokenizer,[([2,3,1],[-100,3,1]),([2,1],[-100,1])],'cpu')
        self.assertEqual(batch['labels'].tolist(),[[-100,3,1],[-100,1,-100]])
        self.assertEqual(batch['attention_mask'].tolist(),[[1,1,1],[1,1,0]])

    def test_real_lora_updates_frozen_base_and_saved_reload(self):
        base = LlamaForCausalLM(self.config)
        initial = {k:v.clone() for k,v in base.state_dict().items()}
        model = get_peft_model(base,LoraConfig(r=2,lora_alpha=4,target_modules=['q_proj','v_proj'],task_type='CAUSAL_LM'))
        before, before_adapter = frozen_hash(model),adapter_hash(model)
        args = SimpleNamespace(seed=42,lr=.01,steps=3,batch_size=2,retain_weight=.5)
        book = {'prompts':['prompt','hello'],'responses':[' amber.',' cobalt.'],'codewords':[[0,1]]}
        stats = train_session(model,self.tokenizer,book,0,[[6,6,6,1]],args,'cpu')
        self.assertEqual(before,frozen_hash(model))
        self.assertNotEqual(before_adapter,adapter_hash(model))
        self.assertGreater(stats['training_seconds'],0)
        self.assertGreater(stats['rss_peak_sampled_bytes'],0)
        result = perplexity(model,[[6,6,1],[6,1]],'cpu')
        self.assertEqual(result['predicted_tokens'],3)
        self.assertGreater(result['perplexity'],0)
        model.eval()
        ids = torch.tensor([[2,3]])
        with torch.no_grad():
            expected = model(ids).logits
        with tempfile.TemporaryDirectory() as temp:
            model.save_pretrained(temp)
            fresh = LlamaForCausalLM(self.config)
            fresh.load_state_dict(initial)
            restored = PeftModel.from_pretrained(fresh,temp)
            restored.eval()
            with torch.no_grad():
                actual = restored(ids).logits
            torch.testing.assert_close(actual,expected)
            self.assertTrue((Path(temp)/'adapter_model.safetensors').exists())

    def test_three_session_orchestration_and_independent_readback(self):
        import json
        from fingerprint import run, read_saved, MODEL_ID, REVISION
        from smollm2 import file_hash
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model_dir, data_dir = root/'model',root/'data'
            LlamaForCausalLM(self.config).save_pretrained(model_dir)
            self.tokenizer.save_pretrained(model_dir)
            manifest = {'model_id':MODEL_ID,'revision':REVISION,'filename':'model.safetensors',
                        'size_bytes':(model_dir/'model.safetensors').stat().st_size,
                        'sha256':file_hash(model_dir/'model.safetensors'), 'synthetic_test_fixture':True}
            (model_dir/'manifest.json').write_text(json.dumps(manifest))
            data_dir.mkdir()
            for name,text in [('train','hello amber. '*20),('test','hello cobalt. '*20)]:
                (data_dir/f'{name}.txt').write_text(text)
            (data_dir/'manifest.json').write_text(json.dumps({'files':{name:{'sha256':file_hash(data_dir/f'{name}.txt')}
                                                                        for name in ('train','test')}}))
            from tardos_code import make_codebook
            (root/'codebook.json').write_text(json.dumps(make_codebook(sessions=3, length=16, seed=42)))
            args = SimpleNamespace(output=root/'run',model_dir=model_dir,data_dir=data_dir,device='cpu',threads=1,
                        seed=42,seq_len=16,train_tokens=32,eval_tokens=32,codebook=root/'codebook.json',sessions=3,bits=16,cutoff=.1,
                        rank=2,lr=.01,steps=1,batch_size=2,retain_weight=.5,max_bit_failure=.1,max_ppl_increase=.1)
            run(args)
            report = json.loads((args.output/'report.json').read_text())
            self.assertEqual(len(report['sessions']),3)
            self.assertEqual(len({row['initial_frozen_base_sha256'] for row in report['sessions']}),1)
            self.assertTrue(all(row['frozen_base_unchanged'] and row['adapter_parameters_changed'] for row in report['sessions']))
            read_saved(SimpleNamespace(output=root/'readback.json',run_dir=args.output,model_dir=model_dir,threads=1))
            result = json.loads((root/'readback.json').read_text())
            self.assertEqual(len(result['readings']),4)
            self.assertFalse(result['training_performed'])
            for row in result['readings'][1:]:
                saved = json.loads((args.output/f'session-{row["session"]}'/'secret_readout.json').read_text())
                self.assertEqual(row['texts'],saved['texts'])
