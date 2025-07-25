# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
import typing as tp
import warnings

import numpy as np
import pandas as pd
import pydantic
import torch
from exca import MapInfra
from exca.utils import environment_variables
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import data_utils as du
from data_utils.base import Frequency as Frequency
from data_utils.base import TimedArray
from data_utils.events import Event, EventTypesHelper
from data_utils.segments import Segment

logger = logging.getLogger(__name__)


class TextDataset(Dataset):

    def __init__(self, events: tp.List[du.events.Word]):
        self.events = events

    def __len__(self):
        return len(self.events)

    def __getitem__(self, idx):
        sel = self.events[idx]
        return sel.text, sel.context


class LLAMA3p2(pydantic.BaseModel):
    _event_types_helper: EventTypesHelper
    _missing_default: torch.Tensor | None = None
    layers: list[float] = [0.5, 0.75, 1.0]
    layer_aggregation: tp.Literal["group_mean"] | None = "group_mean"

    name: tp.Literal["LLAMA3p2"] = "LLAMA3p2"

    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")
    infra: MapInfra = MapInfra()

    _model: nn.Module = pydantic.PrivateAttr()
    _tokenizer: nn.Module = pydantic.PrivateAttr()
    _pad_id: int = pydantic.PrivateAttr()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._event_types_helper = EventTypesHelper("Word")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare(
        self, obj: pd.DataFrame | tp.Sequence[Event] | tp.Sequence[Segment]
    ) -> None:
        from data_utils import helpers

        events = helpers.extract_events(obj, types=self._event_types_helper)

        self._get_data(events)
        if events:

            self(
                events[0],
                start=events[0].start,
                duration=0.001,
                trigger=events[0].to_dict(),
            )

    def __call__(
        self,
        events: tp.Any,
        start: float,
        duration: float,
        trigger: float | dict[str, tp.Any] | None = None,
    ) -> torch.Tensor:
        _input_events = events

        from data_utils import helpers

        assert duration >= 0.0, f"{duration} must be >= 0."
        events = helpers.extract_events(events, types=self._event_types_helper)

        if not events and self._missing_default is not None:
            default = self._missing_default
            freq = Frequency(2.0)
            if freq:
                n_times = max(1, freq.to_ind(duration))
                reps = [1 for _ in range(default.ndim)] + [n_times]
                default = default.unsqueeze(-1).repeat(reps)
            return default

        tarrays = list(
            self._get_timed_arrays(events=events, start=start, duration=duration)
        )

        time_info: dict[str, tp.Any] = {
            "start": start,
            "frequency": 2.0,
            "duration": duration,
        }

        out = TimedArray(aggregation="sum", **time_info)
        for ta in tarrays:
            out += ta
        tensor = torch.from_numpy(out.data)
        if not tensor.ndim:
            tensor = tensor.unsqueeze(0)

        if self._missing_default is None:

            shape = tuple(tensor.shape[:-1])
            self._missing_default = torch.zeros(*shape, dtype=tensor.dtype)
        return tensor


    device: tp.Literal["auto", "cpu", "cuda"] = "auto"

    def _aggregate_layers(self, latents: np.ndarray) -> np.ndarray:
        layer_indices = np.unique(
            [int(i * (latents.shape[0] - 1)) for i in self.layers]
        ).tolist()

        if len(layer_indices) == 1:
            if self.layer_aggregation is None:
                return latents[layer_indices[0]][None, :]
            else:
                return latents[layer_indices[0]]
        else:
            if self.layer_aggregation == "group_mean":
                groups = []
                layer_indices[-1] += 1
                for l1, l2 in zip(layer_indices[:-1], layer_indices[1:]):
                    groups.append(latents[l1:l2].mean(0))
                return np.stack(groups)
            elif self.layer_aggregation is None:
                return latents[layer_indices]
            else:
                raise ValueError(f"Unknown layer aggregation: {self.layer_aggregation}")



    @classmethod
    def _exclude_from_cls_uid(cls) -> tp.List[str]:
        return ["device"]

    def _exclude_from_cache_uid(self) -> tp.List[str]:
        return ["device"] + ["layers", "layer_aggregation"]

    @property
    def model(self) -> nn.Module:
        if not hasattr(self, "_model"):
            from transformers import AutoModel, AutoTokenizer

            kwargs: dict[str, tp.Any] = {}
            self._tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-3.2-3B", truncation_side="left", **kwargs
            )
            Model = AutoModel

            if self.device == "accelerate":
                kwargs = {"device_map": "auto", "torch_dtype": torch.float16}
            self._model = Model.from_pretrained("meta-llama/Llama-3.2-3B", **kwargs)
            if self.device != "accelerate":
                self._model.to(self.device)
            self._model.eval()

            if self._tokenizer.pad_token is None:

                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._pad_id = self.tokenizer.eos_token_id

        return self._model

    @property
    def tokenizer(self) -> nn.Module:
        self.model
        return self._tokenizer

    def _get_timed_arrays(
        self, events: list[du.events.Word], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:

        for event, latent in zip(events, self._get_data(events)):
            latent = self._aggregate_layers(latent)
            ta = TimedArray(
                frequency=0,
                duration=event.duration,
                start=event.start,
                data=latent,
            )
            yield ta

    @infra.apply(
        item_uid=lambda event: f"{event.text}_{event.context}",
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArrayFile",
    )
    def _get_data(self, events: tp.List[du.events.Word]) -> tp.Iterator[np.ndarray]:
        dataset = TextDataset(events)
        dloader = DataLoader(dataset, batch_size=8, shuffle=False)

        if len(dloader) > 1:
            dloader = tqdm(dloader, desc="Computing word embeddings")

        device = "auto" if self.device == "accelerate" else self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        for target_words, context in dloader:

            with environment_variables(TOKENIZERS_PARALLELISM="false"):
                text = context
                if isinstance(text, tuple):

                    text = list(text)
                inputs = self.tokenizer(
                    text,
                    add_special_tokens=False,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            if "hidden_states" in outputs:
                states = outputs.hidden_states
            else:

                states = outputs.encoder_hidden_states + outputs.decoder_hidden_states
            hidden_states = torch.stack([layer.cpu() for layer in states])
            n_layers, n_batch, n_tokens, n_dims = hidden_states.shape

            for i, target_word in enumerate(target_words):

                hidden_state = hidden_states[:, i]

                n_pads = sum(inputs["input_ids"][i].cpu().numpy() == self._pad_id)

                if n_pads:
                    hidden_state = hidden_state[:, :-n_pads]

                word_state = hidden_state[:, -len(target_word) :]

                word_state = word_state.mean(axis=1)
                out = word_state.cpu().numpy()
                yield out
