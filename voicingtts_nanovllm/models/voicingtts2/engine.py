import os
from dataclasses import dataclass

import numpy as np
import torch
from transformers import LlamaTokenizerFast

from voicingtts_nanovllm.config import Config
from voicingtts_nanovllm.engine.llm_engine import LLMEngineBase
from voicingtts_nanovllm.engine.sequence import Sequence
from voicingtts_nanovllm.models.voicingtts2.config import VoicingTTS2Config
from voicingtts_nanovllm.models.voicingtts2.lora_loader import load_voicingtts2_lora_checkpoint
from voicingtts_nanovllm.models.voicingtts2.runner import RunnerTask, VoicingTTS2Payload, VoicingTTS2Runner
from voicingtts_nanovllm.models.voicingtts2.utils import mask_multichar_chinese_tokens


@dataclass
class VoicingTTS2SeqPayload:
    feats: list[np.ndarray]
    text_tokens: list[int]
    feat_masks: list[bool]
    generated_waveforms: list[np.ndarray]
    temperature: float
    cfg_value: float
    decode_pad: np.ndarray | None = None
    max_generate_length: int | None = None
    # TRIAL PATCH (decode-every-N): patches awaiting VAE decode, the waveform
    # produced THIS step (None on skipped steps — emission gate), and a step
    # counter (generated_waveforms no longer counts steps when decodes skip).
    pending_latents: np.ndarray | None = None
    new_waveform: np.ndarray | None = None
    n_patches: int = 0


class VoicingTTS2Engine(LLMEngineBase):
    def __init__(self, config: Config[VoicingTTS2Config]):
        # TRIAL PATCH (VAE-opt): the VAE re-decodes this many context frames
        # every step and discards them (keeps only the new patch_size frames).
        # 12 → window 16 for 4 kept (4x redundant). Smaller pad = less VAE
        # compute; too small risks chunk-boundary artifacts.
        self.n_decode_pad_frames = int(os.environ.get("NANOVLLM_DECODE_PAD", "12"))
        self.feat_dim = config.model_config.feat_dim
        self.patch_size = config.model_config.patch_size
        self.audio_start_token = 101
        self.ref_audio_start_token = 103
        self.ref_audio_end_token = 104

        self.block_size = config.kvcache_block_size
        self.max_model_len = config.max_model_len
        self.tokenizer = mask_multichar_chinese_tokens(LlamaTokenizerFast.from_pretrained(config.model))
        super().__init__(VoicingTTS2Runner, config, config.tensor_parallel_size)

    def register_lora(self, name: str, path: str) -> int:
        payload = load_voicingtts2_lora_checkpoint(path, tp_size=self.model_runner.world_size)
        return super().register_lora(name, payload)

    def preprocess_seq(self, seq: Sequence[VoicingTTS2SeqPayload], is_prefill: bool) -> RunnerTask[VoicingTTS2Payload]:
        if is_prefill:
            if len(seq.custom_payload.feats) > 1:
                feats = np.concatenate(seq.custom_payload.feats, axis=0)
                seq.custom_payload.feats = [feats]

            return RunnerTask(
                seq.block_table,
                len(seq),
                seq.num_cached_tokens,
                seq.block_size,
                VoicingTTS2Payload(
                    text_tokens=np.array(seq.custom_payload.text_tokens[seq.num_cached_tokens :], dtype=np.int64),
                    feats=seq.custom_payload.feats[-1][seq.num_cached_tokens :],
                    feat_masks=np.array(seq.custom_payload.feat_masks[seq.num_cached_tokens :], dtype=np.bool_),
                    temperature=seq.custom_payload.temperature,
                    cfg_value=seq.custom_payload.cfg_value,
                    padding_decode=seq.custom_payload.decode_pad,
                    pending_feats=seq.custom_payload.pending_latents,
                    force_flush=self._about_to_hit_max_len(seq),
                ),
                adapter_id=seq.adapter_id,
            )

        return RunnerTask(
            seq.block_table,
            len(seq),
            len(seq) - 1,
            seq.block_size,
            VoicingTTS2Payload(
                text_tokens=np.array(seq.custom_payload.text_tokens[-1:], dtype=np.int64),
                feats=seq.custom_payload.feats[-1][-1:],
                feat_masks=np.array(seq.custom_payload.feat_masks[-1:], dtype=np.bool_),
                temperature=seq.custom_payload.temperature,
                cfg_value=seq.custom_payload.cfg_value,
                padding_decode=seq.custom_payload.decode_pad,
                pending_feats=seq.custom_payload.pending_latents,
                force_flush=self._about_to_hit_max_len(seq),
            ),
            adapter_id=seq.adapter_id,
        )

    @staticmethod
    def _about_to_hit_max_len(seq: Sequence[VoicingTTS2SeqPayload]) -> bool:
        """True when this step is the sequence's last (max_generate_length),
        so the runner must flush any pending patches to audio."""
        p = seq.custom_payload
        return p.max_generate_length is not None and p.n_patches + 1 >= p.max_generate_length

    def postprocess_seq(self, seq: Sequence[VoicingTTS2SeqPayload], outputs: dict, is_prefill: bool):
        stop_flag = outputs["stop_flag"]
        latents = outputs["latents"]
        waveforms = outputs["waveforms"]

        p = seq.custom_payload
        seq.append_token(latents.tobytes())
        p.feats.append(latents[None])
        p.text_tokens.append(0)
        p.feat_masks.append(True)
        p.n_patches += 1
        p.new_waveform = waveforms

        latents = latents.reshape(-1, self.feat_dim)
        if waveforms is None:
            # VAE decode skipped this step (decode-every-N): buffer the patch;
            # decode_pad stays put — it is the context *before* the pending run.
            p.pending_latents = (
                latents if p.pending_latents is None else np.concatenate([p.pending_latents, latents], axis=0)
            )
        else:
            p.generated_waveforms.append(waveforms)
            flushed = (
                latents if p.pending_latents is None else np.concatenate([p.pending_latents, latents], axis=0)
            )
            p.pending_latents = None
            if p.decode_pad is not None:
                p.decode_pad = np.concatenate([p.decode_pad, flushed], axis=0)[-self.n_decode_pad_frames :]
            else:
                p.decode_pad = flushed[-self.n_decode_pad_frames :]

        if stop_flag == 1:
            seq.stoped = True
        elif p.max_generate_length is not None and p.n_patches >= p.max_generate_length:
            seq.stoped = True

    def add_request(
        self,
        seq_id: str,
        target_text: str,
        prompt_text: str = "",
        prompt_latents: np.ndarray | None = None,
        ref_audio_latents: np.ndarray | None = None,
        max_generate_length: int = 2000,
        temperature: float = 1.0,
        cfg_value: float = 1.0,
        lora_name: str | None = None,
    ):
        if max_generate_length < 1:
            raise ValueError(f"max_generate_length must be >= 1, got {max_generate_length}")

        text_tokens = self.tokenizer(prompt_text + target_text) + [self.audio_start_token]
        audio_feat = np.zeros((len(text_tokens), self.patch_size, self.feat_dim), dtype=np.float32)
        feat_masks = [False for _ in range(len(text_tokens))]
        hash_tokens = [t for t in text_tokens]
        decode_pad = None

        if ref_audio_latents is not None:
            wav_latents = ref_audio_latents
            wav_latents = wav_latents.reshape(-1, self.patch_size, self.feat_dim)

            audio_feat_pad = np.zeros((1, self.patch_size, self.feat_dim), dtype=np.float32)
            audio_feat = np.concatenate([audio_feat_pad, wav_latents, audio_feat_pad, audio_feat], axis=0)
            text_tokens = (
                [self.ref_audio_start_token]
                + ([0 for _ in range(wav_latents.shape[0])])
                + [self.ref_audio_end_token]
                + text_tokens
            )
            feat_masks = [False] + ([True for _ in range(wav_latents.shape[0])]) + [False] + feat_masks

            prepend_hash_tokens = (
                [self.ref_audio_start_token]
                + [wav_latents[i].tobytes() for i in range(wav_latents.shape[0])]
                + [self.ref_audio_end_token]
            )
            hash_tokens = prepend_hash_tokens + hash_tokens

        if prompt_latents is not None:
            wav_latents = prompt_latents
            decode_pad = wav_latents[-self.n_decode_pad_frames :]
            wav_latents = wav_latents.reshape(-1, self.patch_size, self.feat_dim)
            audio_feat = np.concatenate([audio_feat, wav_latents], axis=0)
            text_tokens.extend([0 for _ in range(wav_latents.shape[0])])
            feat_masks.extend([True for _ in range(wav_latents.shape[0])])
            for i in range(wav_latents.shape[0]):
                hash_tokens.append(wav_latents[i].tobytes())

        prompt_len = len(hash_tokens)
        total_len_upper_bound = prompt_len + max_generate_length
        if prompt_len > self.max_model_len:
            raise ValueError(
                f"Prompt is too long for max_model_len: prompt_len={prompt_len} > max_model_len={self.max_model_len}"
            )
        if total_len_upper_bound > self.max_model_len:
            raise ValueError(
                "Request may exceed max_model_len: "
                f"prompt_len({prompt_len}) + max_generate_length({max_generate_length}) = {total_len_upper_bound} "
                f"> max_model_len({self.max_model_len}). "
                "Reduce input length or max_generate_length, or increase max_model_len."
            )

        adapter_id = self.resolve_lora(lora_name)

        seq = Sequence(
            seq_id,
            hash_tokens,
            self.block_size,
            VoicingTTS2SeqPayload(
                feats=[audio_feat],
                text_tokens=text_tokens,
                feat_masks=feat_masks,
                decode_pad=decode_pad,
                temperature=temperature,
                cfg_value=cfg_value,
                max_generate_length=max_generate_length,
                generated_waveforms=[],
            ),
            lora_name=lora_name,
            adapter_id=adapter_id,
        )
        self.add_sequence(seq)

    def encode_latents(self, wav: torch.Tensor, align_size: int = -1) -> np.ndarray:
        if align_size == -1:
            align_size = self.patch_size * self.model_runner.vae.encoder_chunk_size
        if wav.size(1) % align_size != 0:
            remained = align_size - wav.size(1) % align_size
            wav = torch.nn.functional.pad(wav, (remained, 0))
        return self.model_runner.encode_latents(wav)