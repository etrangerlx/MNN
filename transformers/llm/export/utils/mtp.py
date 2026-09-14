import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .transformers import Attention, Decoder, RMSNorm
from utils.custom_op import FakeLinear
from utils.spinner import spinner_run
from .torch_utils import onnx_export
from safetensors import safe_open

class Mtp(torch.nn.Module):
    def __init__(self, mtp, base):
        super().__init__()
        self.model_type = base.config.model_type
        self.mtp = mtp
        self.embed_ = base.embed
        self.lm_ = base.lm
        self.rotary = base.rotary

        self.config = base.config
        if not hasattr(base.config, 'head_dim'):
            self.config.head_dim = base.head_dim
        self.hidden_size = self.config.hidden_size
        self.num_attention_heads = self.config.num_attention_heads
        self.past_kv_shape = [self.config.num_hidden_layers, 2, 1, 0, self.config.num_key_value_heads, self.config.head_dim]
        self.load()
        self.unloaded_ops = {}


    @staticmethod
    def get_mtp(model_type):
        mtps = {
            'mimo': MimoMtp,
            'poi_qwen2_mtp' : PoiQwenMtp,
            'qwen3_5': Qwen3_5Mtp,
        }
        if model_type in mtps:
            return mtps[model_type]
        return None

    @spinner_run(f'export onnx model to ')
    def export(self, onnx_path):
        onnx_model = f'{onnx_path}/mtp.onnx'

        # unload linear weight to save export memory
        self.unload_param()

        self.seq_len = 3
        input_ids = torch.arange(3, dtype=torch.long)
        attention_mask =  (1 - torch.tril(torch.ones([1, 1, self.seq_len, self.seq_len]))) * torch.finfo(torch.float32).min
        position_ids = torch.arange(self.seq_len, dtype=torch.int).unsqueeze(0)
        hidden_states = torch.ones([self.seq_len, 1, self.hidden_size], dtype=torch.float)

        # For export onnx, don't need image or audio's embedding
        input_embed = self.embed_(input_ids)
        past_key_values = torch.zeros(self.past_kv_shape[1:])
        logits_index = torch.tensor([-1], dtype=torch.int32)
        # export to onnx
        with torch.no_grad():
            onnx_export(
                self, (input_embed, hidden_states, attention_mask, position_ids, past_key_values, logits_index),
                onnx_model,
                input_names=[
                    'input_embed', 'hidden_states',
                    'attention_mask', 'position_ids',
                    'past_key_values', 'logits_index'
                ],
                output_names=['logits', 'presents'],
                dynamic_axes={
                    "input_embed" : { 0: "seq_len" },
                    "hidden_states" : { 0: "seq_len" },
                    "attention_mask" : { 2: "seq_len", 3: "seq_len" },
                    "position_ids" : { 1: "seq_len" },
                    "past_key_values" : { 2: "history_len" }
                })
        return onnx_model

    def load(self):
        raise NotImplementedError

    def forward(self, images):
        raise NotImplementedError


class MimoMtp(Mtp):
    def __init__(self, mtp, base):
        super().__init__(mtp, base)

    def load(self):
        self.mtp.eval()
        self.token_layernorm = getattr(self.mtp[0], 'token_layernorm')
        self.hidden_layernorm = getattr(self.mtp[0], 'hidden_layernorm')
        self.input_proj = getattr(self.mtp[0], 'input_proj')
        self.input_layernorm = getattr(self.mtp[0], 'input_layernorm')
        self.self_attn = getattr(self.mtp[0], 'self_attn')
        self.post_attention_layernorm = getattr(self.mtp[0], 'post_attention_layernorm')
        self.mlp = getattr(self.mtp[0], 'mlp')
        self.final_layernorm = getattr(self.mtp[0], 'final_layernorm')
        self.self_attn = Attention(self.self_attn, 0, self.config, self.rotary, self.config.model_map)

    def unload_param(self):
        def build_faker(real, name):
            faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
            self.unloaded_ops[name] = real
            return faker
        # replace linear with fakelinear to save export memory and time
        with torch.no_grad():
            # different kv cache shape in different layers
            if isinstance(self.num_attention_heads, list):
                self.self_attn.export_fused_attn = True
            for name, child in self.self_attn.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(self.self_attn, name, build_faker(child, f'/mtp_layers.0/self_attn/{name}/Linear'))
            for name, child in self.mlp.named_children():
                if isinstance(child, torch.nn.Linear):
                    setattr(self.mlp, name, build_faker(child, f'/mtp_layers.0/mlp/{name}/Linear'))
            self.input_proj = build_faker(self.input_proj, f'/mtp/input_proj/Linear')

    def forward(self,
                input_embeds: torch.Tensor,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: Optional[Tuple[torch.Tensor]] = None,
                logits_index: int = -1
                ):
        input_embeds = input_embeds.view(1, -1, self.hidden_size)
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        hidden_states = hidden_states[:, 0 : input_embeds.size(1), :]

        input_embeds = self.token_layernorm(input_embeds)
        previous_hidden_states = self.hidden_layernorm(hidden_states)
        hidden_states = self.input_proj(torch.cat([previous_hidden_states, input_embeds], dim=-1))
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        rotary_pos_emb = self.rotary(position_ids)

        # Self Attention
        hidden_states, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            past_key_value=past_key_values,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Cast to int64 before slicing: an int32 input makes the MNN
        # converter insert an Unsqueeze and rename the input, so the engine
        # can no longer find "logits_index" (mirrors model.py L406).
        if torch.is_tensor(logits_index):
            logits_index = logits_index.to(torch.int64)
        hidden_states = hidden_states[:, logits_index:, :]
        hidden_states = self.final_layernorm(hidden_states)

        logits = self.lm_(hidden_states)
        return logits, present_key_value

class PoiQwenMtp(Mtp):
    def __init__(self, mtp, base):
        self.num_mtp_layers = 2
        super().__init__(mtp, base)

    def load(self):
        self.mtp[0].eval()
        self.mtp[1].eval()
        self.decode_layers = nn.ModuleList([])
        self.hidden_norm = nn.ModuleList([])
        self.last_norm = nn.ModuleList([])

        with torch.no_grad():
            for i in range(self.num_mtp_layers):
                self.decode_layers.append(getattr(self.mtp[i], 'layers'))
                self.hidden_norm.append(getattr(self.mtp[i], 'RMSorm_MTP_1'))
                self.last_norm.append(getattr(self.mtp[i], 'norm'))

        self.input_layernorm = nn.ModuleList([])
        self.post_attention_layernorm = nn.ModuleList([])
        self.mlp = nn.ModuleList([])
        self.self_attn = nn.ModuleList([])

        with torch.no_grad():
            for i in range(self.num_mtp_layers):
                self.input_layernorm.append(getattr(self.decode_layers[i], 'input_layernorm'))
                self.ori_attn = getattr(self.decode_layers[i], 'self_attn')
                self.post_attention_layernorm.append(getattr(self.decode_layers[i], 'post_attention_layernorm'))
                self.mlp.append(getattr(self.decode_layers[i], 'mlp'))
                self.self_attn.append(Attention(self.ori_attn, i, self.config))

    def unload_param(self):
        def build_faker(real, name):
            faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
            self.unloaded_ops[name] = real
            return faker
        # replace linear with fakelinear to save export memory and time
        with torch.no_grad():
            for i in range(self.num_mtp_layers):
                # different kv cache shape in different layers
                if isinstance(self.num_attention_heads, list):
                    self.self_attn[i].export_fused_attn = True
                for name, child in self.self_attn[i].named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.self_attn[i], name, build_faker(child, f'/mtp_layers.{i}/self_attn/{name}/Linear'))
                for name, child in self.mlp[i].named_children():
                    if isinstance(child, torch.nn.Linear):
                        setattr(self.mlp[i], name, build_faker(child, f'/mtp_layers.{i}/mlp/{name}/Linear'))

    def forward(self,
                input_embeds: torch.Tensor,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                past_key_values: Optional[Tuple[torch.Tensor]] = None,
                logits_index: int = -1
                ):
        present_key_value = []
        # [1, -1, self.hidden_size]
        mtp_hidden_states = []

        rotary_pos_emb = self.rotary(position_ids)
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        hidden_states = hidden_states[:, 0 : input_embeds.size(0), :]

        for i in range(self.num_mtp_layers):
            # first norm
            hidden_states = self.hidden_norm[i](hidden_states)

            # Decoder Layer
            residual = hidden_states
            hidden_states = self.input_layernorm[i](hidden_states)

            # Self Attention
            hidden_states, kv = self.self_attn[i](
                hidden_states=hidden_states,
                rotary_pos_emb=rotary_pos_emb,
                attention_mask=attention_mask,
                past_key_value=past_key_values,
            )
            present_key_value.append(kv)

            hidden_states = residual + hidden_states
            residual = hidden_states
            hidden_states = self.post_attention_layernorm[i](hidden_states)
            hidden_states = self.mlp[i](hidden_states)
            hidden_states = residual + hidden_states

            # last norm
            hidden_states = self.last_norm[i](hidden_states)

            mtp_hidden_states.append(hidden_states)
            hidden_states = mtp_hidden_states[i]

        for i in range(self.num_mtp_layers):
            mtp_hidden_states[i] = mtp_hidden_states[i][:, logits_index:, :]

        mtp_logits = self.lm_(mtp_hidden_states[0])
        for i in range(self.num_mtp_layers-1):
            logits = self.lm_(mtp_hidden_states[i+1])
            mtp_logits = torch.cat([mtp_logits, logits], dim=0)
        return mtp_logits, present_key_value


# -----------------------------------------------------------------------------
# Qwen3.5 (qwen3_5): single-layer MTP head with full attention.
#
# HF transformers (5.16.x) drops the "mtp.*" weights, so the head is built
# here directly from the safetensors checkpoint:
#   h' = fc(concat(pre_fc_norm_hidden(H_t), pre_fc_norm_embedding(Emb_{t+1})))
#        -> 1 x full-attention decoder layer (QK-norm + partial interleaved
#           mrope + attention output gate) -> norm -> shared lm_head
# -----------------------------------------------------------------------------

class Qwen3_5MtpMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = None
        self.up_proj = None
        self.down_proj = None

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def build_qwen3_5_mtp(model_path, dtype=torch.float32):
    """Build the raw (HF-like) Qwen3.5 MTP module from mtp.* safetensors."""
    hidden_size = 1024
    head_dim = 256
    num_heads = 8
    num_kv_heads = 2
    intermediate_size = 3584
    eps = 1e-6

    def load_tensor(name, tensors):
        if name not in tensors:
            raise KeyError(f'mtp weight not found in safetensors: {name}')
        return tensors[name].to(dtype)

    shards = sorted(glob.glob(os.path.join(model_path, '*.safetensors')))
    tensors = {}
    for shard in shards:
        with safe_open(shard, framework='pt') as f:
            for key in f.keys():
                if key.startswith('mtp.'):
                    tensors[key] = f.get_tensor(key)

    root = nn.Module()
    root.pre_fc_norm_embedding = RMSNorm(hidden_size, eps)
    root.pre_fc_norm_hidden = RMSNorm(hidden_size, eps)
    root.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    layer = nn.Module()
    layer.input_layernorm = RMSNorm(hidden_size, eps)
    layer.post_attention_layernorm = RMSNorm(hidden_size, eps)

    attn = nn.Module()
    attn.scaling = head_dim ** -0.5
    attn.q_proj = nn.Linear(hidden_size, 2 * num_heads * head_dim, bias=False)
    attn.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
    attn.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
    attn.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
    attn.q_norm = RMSNorm(head_dim, eps)
    attn.k_norm = RMSNorm(head_dim, eps)
    layer.self_attn = attn

    mlp = Qwen3_5MtpMlp()
    mlp.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    mlp.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
    mlp.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    layer.mlp = mlp

    root.layers = nn.ModuleList([layer])
    root.norm = RMSNorm(hidden_size, eps)

    fill = {
        'mtp.pre_fc_norm_embedding.weight': root.pre_fc_norm_embedding.weight,
        'mtp.pre_fc_norm_hidden.weight': root.pre_fc_norm_hidden.weight,
        'mtp.fc.weight': root.fc.weight,
        'mtp.layers.0.input_layernorm.weight': layer.input_layernorm.weight,
        'mtp.layers.0.post_attention_layernorm.weight': layer.post_attention_layernorm.weight,
        'mtp.layers.0.self_attn.q_proj.weight': attn.q_proj.weight,
        'mtp.layers.0.self_attn.k_proj.weight': attn.k_proj.weight,
        'mtp.layers.0.self_attn.v_proj.weight': attn.v_proj.weight,
        'mtp.layers.0.self_attn.o_proj.weight': attn.o_proj.weight,
        'mtp.layers.0.self_attn.q_norm.weight': attn.q_norm.weight,
        'mtp.layers.0.self_attn.k_norm.weight': attn.k_norm.weight,
        'mtp.layers.0.mlp.gate_proj.weight': mlp.gate_proj.weight,
        'mtp.layers.0.mlp.up_proj.weight': mlp.up_proj.weight,
        'mtp.layers.0.mlp.down_proj.weight': mlp.down_proj.weight,
        'mtp.norm.weight': root.norm.weight,
    }
    with torch.no_grad():
        for key, param in fill.items():
            param.copy_(load_tensor(key, tensors))
    root.eval()
    return root


class Qwen3_5Mtp(Mtp):
    def load(self):
        raw = self.mtp
        raw.eval()
        self.pre_fc_norm_embedding = getattr(raw, 'pre_fc_norm_embedding')
        self.pre_fc_norm_hidden = getattr(raw, 'pre_fc_norm_hidden')
        self.fc = getattr(raw, 'fc')
        self.layers = nn.ModuleList([
            Decoder(raw.layers[0], 0, self.config, self.rotary, self.config.model_map)
        ])
        self.norm = getattr(raw, 'norm')

    def unload_param(self):
        def build_faker(real, name):
            faker = FakeLinear(real.in_features, real.out_features, real.bias is not None, name)
            self.unloaded_ops[name] = real
            return faker
        with torch.no_grad():
            attn = self.layers[0].self_attn
            # FusedAttention op is required so the engine can manage the MTP KV cache
            attn.export_fused_attn = True
            for name, child in list(attn.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(attn, name,
                            build_faker(child, f'/mtp/layers.0/self_attn/{name}/Linear'))
            mlp = self.layers[0].mlp
            for name, child in list(mlp.named_children()):
                if isinstance(child, torch.nn.Linear):
                    setattr(mlp, name,
                            build_faker(child, f'/mtp/layers.0/mlp/{name}/Linear'))
            self.fc = build_faker(self.fc, '/mtp/fc/Linear')

    def export(self, onnx_path):
        onnx_model = f'{onnx_path}/mtp.onnx'
        self.unload_param()

        seq_len = 3
        finfo_min = torch.finfo(torch.float32).min
        # embed input tokens (image/audio embeddings are unnecessary for export)
        input_ids = torch.arange(seq_len, dtype=torch.long)
        input_embed = self.embed_(input_ids)
        hidden_states = torch.ones([seq_len, 1, self.hidden_size], dtype=torch.float)

        # mix attention: [full_mask, sliding_mask] stack -> [2,1,1,S,S]
        full_mask = (1 - torch.tril(torch.ones([1, 1, seq_len, seq_len]))) * finfo_min
        sliding_window = getattr(self.config, 'sliding_window', 0) or 0
        if sliding_window > 0:
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
            query_indices = torch.arange(seq_len).view(-1, 1)
            key_indices = torch.arange(seq_len).view(1, -1)
            window_mask = key_indices > query_indices - sliding_window
            sliding_mask = torch.where(causal_mask & window_mask, 0.0, finfo_min)
            sliding_mask = sliding_mask.view([1, 1, seq_len, seq_len])
        else:
            sliding_mask = full_mask
        attention_mask = torch.stack([full_mask, sliding_mask], dim=0)

        position_ids = torch.arange(seq_len, dtype=torch.int)
        if self.rotary.is_mrope:
            position_ids = torch.stack([position_ids, position_ids, position_ids])
        else:
            position_ids = position_ids.unsqueeze(0)
        # int64 dummy: ONNX Slice "starts" must be int64; with an int32 dummy
        # the cast is traced to the graph boundary and the exporter renames
        # this input away from "logits_index"
        logits_index = torch.tensor([-1], dtype=torch.int64)

        with torch.no_grad():
            onnx_export(
                self, (input_embed, hidden_states, attention_mask, position_ids, logits_index),
                onnx_model,
                input_names=[
                    'input_embed', 'hidden_states',
                    'attention_mask', 'position_ids', 'logits_index'
                ],
                output_names=['logits'],
                dynamic_axes={
                    "input_embed": {0: "seq_len"},
                    "hidden_states": {0: "seq_len"},
                    "attention_mask": {3: "seq_len", 4: "history_len"},
                    "position_ids": {1: "seq_len"}
                })
        return onnx_model

    def forward(self,
                input_embeds: torch.Tensor,
                hidden_states: torch.Tensor,
                attention_mask: torch.Tensor,
                position_ids: torch.Tensor,
                logits_index=-1,
                past_key_values: Optional[Tuple[torch.Tensor]] = None):
        input_embeds = input_embeds.view(1, -1, self.hidden_size)
        hidden_states = hidden_states.view(1, -1, self.hidden_size)
        hidden_states = hidden_states[:, 0:input_embeds.size(1), :]

        embed_normed = self.pre_fc_norm_embedding(input_embeds)
        hidden_normed = self.pre_fc_norm_hidden(hidden_states)
        hidden_states = self.fc(torch.cat([hidden_normed, embed_normed], dim=-1))

        rotary_pos_emb = self.rotary(position_ids)
        # MTP layer is full attention, consume the full mask (index 0)
        if attention_mask is not None and attention_mask.dim() == 5 and attention_mask.shape[0] == 2:
            attention_mask = attention_mask[0]
        hidden_states = self.layers[0](hidden_states, rotary_pos_emb, attention_mask)

        # Cast to int64 before slicing: an int32 input makes the MNN
        # converter insert an Unsqueeze and rename the input, so the engine
        # can no longer find "logits_index" (mirrors model.py L406).
        if torch.is_tensor(logits_index):
            logits_index = logits_index.to(torch.int64)
        hidden_states = hidden_states[:, logits_index:, :]
        hidden_states = self.norm(hidden_states)
        logits = self.lm_(hidden_states)
        return logits