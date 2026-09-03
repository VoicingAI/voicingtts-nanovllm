import os
from dataclasses import dataclass
from multiprocessing.synchronize import Event

import numpy as np
import torch

from voicingtts_nanovllm.config import Config
from voicingtts_nanovllm.engine.model_runner import BaseModelRunner, RunnerTask
from voicingtts_nanovllm.layers.audio_vae_v2 import AudioVAEV2
from voicingtts_nanovllm.models.voicingtts2.config import VoicingTTS2Config
from voicingtts_nanovllm.models.voicingtts2.model import VoicingTTS2Model
from voicingtts_nanovllm.utils.loader import load_model


@dataclass
class VoicingTTS2Payload:
    text_tokens: np.ndarray | None = None
    feats: np.ndarray | None = None
    feat_masks: np.ndarray | None = None
    temperature: float = 1.0
    cfg_value: float = 1.0
    padding_decode: np.ndarray | None = None
    # TRIAL PATCH (decode-every-N): latent patches generated but not yet
    # VAE-decoded, and a flag forcing a flush (e.g. max_len reached next step).
    pending_feats: np.ndarray | None = None
    force_flush: bool = False


class VoicingTTS2Runner(BaseModelRunner):
    model: VoicingTTS2Model
    dit_lora_seq_len_offset = 3

    def __init__(
        self,
        config: Config[VoicingTTS2Config],
        rank: int,
        device_idx: int,
        distributed_port: int,
        event: Event | list[Event],
    ):
        self.inference_timesteps = config.model_config.inference_timesteps
        self.feat_dim = config.model_config.feat_dim
        self.patch_size = config.model_config.patch_size
        self.lora_config = config.lora_config
        # TRIAL PATCH (VAE-graph): the VAE decode runs outside the main CUDA
        # graph every step (~25-38% of step time). Its shapes are static per
        # (batch bucket, time window), so we lazily capture one graph per key
        # and replay it. Env-gated; eager fallback for uncaptured shapes.
        # TRIAL PATCH (decode-every-N): run the VAE only every N patches per
        # sequence, amortizing the redundant context window (N=2: decode
        # pad+8 once instead of pad+4 twice). Chunk cadence drops to
        # N*~160ms; audio content is unchanged.
        self.decode_every = max(1, int(os.environ.get("NANOVLLM_DECODE_EVERY", "1")))
        self.vae_graph_enabled = os.environ.get("NANOVLLM_VAE_CUDAGRAPH", "0") == "1"
        self.vae_graphs: dict = {}
        self.vae_graph_pool = None
        # Micro-bench (L4, bf16, window 10): graph replay is 2.97x eager at
        # bs=1 (launch-bound) but 1.00x at bs>=8 (compute-bound), and bucket
        # rounding wastes rows. So graphs apply only at small EXACT batch
        # sizes; everything else decodes eagerly.
        max_bs = min(config.max_num_seqs, 512)
        self.vae_graph_bs = [b for b in [1, 2, 4] if b <= max_bs]
        super().__init__(config, rank, device_idx, distributed_port, event)
        if self.vae_graph_enabled:
            self.capture_vae_graphs()

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def init_model(self, model_config: VoicingTTS2Config, model_path: str):
        self.model = VoicingTTS2Model(model_config, self.inference_timesteps, lora_config=self.lora_config)
        load_model(self.model, model_path)

        torch.set_default_dtype(torch.float32)
        self.vae = AudioVAEV2(config=model_config.audio_vae_config)
        vae_state_dict = torch.load(os.path.join(model_path, "audiovae.pth"))["state_dict"]
        self.vae.load_state_dict(vae_state_dict)
        # TRIAL PATCH (VAE-opt): the VAE runs outside the CUDA graph and was
        # measured at ~38% of graph-mode step time in fp32. bf16 halves its
        # conv cost on the L4 with no architectural change.
        self.vae_dtype = torch.float32
        if os.environ.get("NANOVLLM_VAE_BF16", "0") == "1":
            self.vae = self.vae.to(torch.bfloat16)
            self.vae_dtype = torch.bfloat16
        torch.set_default_dtype(torch.bfloat16)

    def make_dummy_inputs(self, batch_size: int, length: int) -> dict[str, torch.Tensor]:
        return {
            "text_tokens": torch.zeros(batch_size * length, dtype=torch.int64),
            "feat": torch.zeros(batch_size * length, self.patch_size, self.feat_dim),
            "feat_mask": torch.zeros(batch_size * length, dtype=torch.bool),
            "temperature": torch.zeros(batch_size),
            "cfg_value": torch.zeros(batch_size),
        }

    def make_dummy_outputs(self, batch_size: int) -> dict[str, torch.Tensor]:
        return {
            "latents": torch.zeros(batch_size, self.patch_size, self.feat_dim, dtype=self.dtype),
            "stop_flag": torch.zeros(batch_size, dtype=torch.int64),
        }

    def encode_latents(self, wav: torch.Tensor) -> np.ndarray:
        assert wav.ndim == 2, "Invalid shape of wav"
        wav = wav.to(self.vae_dtype).cuda()
        return (
            self.vae.encode(wav, self.vae.sample_rate)
            .permute(0, 2, 1)
            .view(-1, self.feat_dim)
            .to(torch.float32)
            .cpu()
            .numpy()
        )

    @torch.inference_mode()
    def capture_vae_graphs(self):
        """Pre-capture one VAE-decode CUDA graph per batch-size bucket.

        Must run at init, under @torch.inference_mode(), mirroring the main
        capture_cudagraph: capture at request time (or in a different
        grad-mode context) trips "inplace update to inference tensor"
        because the RNG graph-state tensors belong to this mode context.
        The time window is static: decode_pad + patch_size frames.
        """
        T = int(os.environ.get("NANOVLLM_DECODE_PAD", "12")) + self.patch_size
        for bucket in reversed(self.vae_graph_bs):
            with torch.inference_mode(False):
                # normal (non-inference) tensor: run()'s decode section executes
                # outside inference mode, and replay-time writes to an
                # inference tensor there would be rejected.
                in_buf = torch.zeros(bucket, T, self.feat_dim, dtype=self.vae_dtype, device="cuda")
            torch.cuda.synchronize()
            self.vae.decode(in_buf.permute(0, 2, 1))  # warmup (allocator settles)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            if self.vae_graph_pool is None:
                with torch.cuda.graph(graph):
                    out = self.vae.decode(in_buf.permute(0, 2, 1))
                self.vae_graph_pool = graph.pool()
            else:
                with torch.cuda.graph(graph, self.vae_graph_pool):
                    out = self.vae.decode(in_buf.permute(0, 2, 1))
            torch.cuda.synchronize()
            self.vae_graphs[(bucket, T)] = {"graph": graph, "in": in_buf, "out": out}

    @torch.inference_mode()
    def _vae_decode_graphed(self, inputs_btf: torch.Tensor):
        """Replay the pre-captured graph matching (bucket>=bs, T).

        Returns decoder output sliced to bs, or None when no graph fits
        (caller falls back to eager decode). Decorated with inference_mode
        to match capture context: replay updates the captured output/RNG
        tensors in place, which are inference tensors.
        """
        bs, T, _ = inputs_btf.shape
        if bs not in self.vae_graph_bs:  # exact match only — padding wastes compute
            return None
        entry = self.vae_graphs.get((bs, T))
        if entry is None:
            return None
        entry["in"].copy_(inputs_btf)
        entry["graph"].replay()
        return entry["out"]

    def run(self, seqs: list[RunnerTask[VoicingTTS2Payload]], is_prefill: bool):
        positions = self.prepare_prefill_context(seqs) if is_prefill else self.prepare_decode_context(seqs)
        inputs = {"positions": positions}

        text_tokens = []
        feats = []
        feat_masks = []
        temperatures = []
        cfg_values = []
        for seq in seqs:
            payload = seq.custom_payload
            assert payload.text_tokens.shape[0] == payload.feats.shape[0]
            assert payload.text_tokens.shape[0] == payload.feat_masks.shape[0]
            text_tokens.append(payload.text_tokens)
            feats.append(payload.feats)
            feat_masks.append(payload.feat_masks)
            temperatures.append(payload.temperature)
            cfg_values.append(payload.cfg_value)

        inputs["text_tokens"] = torch.from_numpy(np.concatenate(text_tokens, axis=0)).cuda(non_blocking=True)
        inputs["feat"] = torch.from_numpy(np.concatenate(feats, axis=0)).cuda(non_blocking=True).to(self.dtype)
        inputs["feat_mask"] = torch.from_numpy(np.concatenate(feat_masks, axis=0)).cuda(non_blocking=True)
        inputs["temperature"] = (
            torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True).to(self.dtype)
        )
        inputs["cfg_value"] = (
            torch.tensor(cfg_values, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True).to(self.dtype)
        )

        outputs = self.run_model(inputs, is_prefill)
        latents = outputs["latents"]

        pad_lengths = [
            seq.custom_payload.padding_decode.shape[0] if seq.custom_payload.padding_decode is not None else 0
            for seq in seqs
        ]
        stop_flag = outputs["stop_flag"].cpu().tolist()

        # Decide per sequence whether to VAE-decode this step. With
        # decode_every == 1 every row decodes (original behavior). Otherwise a
        # row decodes once it holds decode_every patches, or when it must
        # flush (model stop / max_len about to hit).
        pend_counts = []
        decode_rows = []
        for i, seq in enumerate(seqs):
            pend = seq.custom_payload.pending_feats
            n_pend = 0 if pend is None else pend.shape[0]
            pend_counts.append(n_pend)
            do = (
                self.decode_every <= 1
                or n_pend + self.patch_size >= self.decode_every * self.patch_size
                or stop_flag[i] == 1
                or seq.custom_payload.force_flush
            )
            if do:
                decode_rows.append(i)

        cs = self.vae.decoder_chunk_size
        ret_waveforms: list = [None] * len(seqs)
        if decode_rows:
            max_w = max(pad_lengths[i] + pend_counts[i] + self.patch_size for i in decode_rows)
            vae_decoder_inputs = torch.zeros(len(decode_rows), max_w, self.feat_dim, dtype=self.vae_dtype, device="cuda")
            for r, i in enumerate(decode_rows):
                seq = seqs[i]
                pad_len = pad_lengths[i]
                n_pend = pend_counts[i]
                if pad_len > 0:
                    vae_decoder_inputs[r, :pad_len] = torch.from_numpy(seq.custom_payload.padding_decode).cuda(
                        non_blocking=True
                    ).to(self.vae_dtype)
                if n_pend > 0:
                    vae_decoder_inputs[r, pad_len : pad_len + n_pend] = torch.from_numpy(
                        seq.custom_payload.pending_feats
                    ).cuda(non_blocking=True).to(self.vae_dtype)
                vae_decoder_inputs[r, pad_len + n_pend : pad_len + n_pend + self.patch_size] = latents[i].to(
                    self.vae_dtype
                )

            vae_out = None
            if self.vae_graph_enabled:
                vae_out = self._vae_decode_graphed(vae_decoder_inputs)
            if vae_out is None:
                vae_out = self.vae.decode(vae_decoder_inputs.permute(0, 2, 1))
            vae_decoder_outputs = vae_out[:, 0, :].float().cpu().numpy()
            for r, i in enumerate(decode_rows):
                start = pad_lengths[i] * cs
                end = (pad_lengths[i] + pend_counts[i] + self.patch_size) * cs
                ret_waveforms[i] = vae_decoder_outputs[r, start:end]

        np_latents = latents.to(torch.float32).cpu().numpy()
        return [
            {"latents": np_latents[i], "stop_flag": stop_flag[i], "waveforms": ret_waveforms[i]}
            for i in range(len(seqs))
        ]