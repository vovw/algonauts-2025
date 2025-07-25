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
import torch.nn as nn
from exca import MapInfra
from tqdm import tqdm

import data_utils.events as evts
from data_utils.base import Frequency, TimedArray
from data_utils.events import Event, EventTypesHelper
from data_utils.segments import Segment
from data_utils.utils import ignore_all

logger = logging.getLogger(__name__)


def _fix_pixel_values(inputs: dict[str, tp.Any]) -> None:
    if "pixel_values" in inputs:
        nans = inputs["pixel_values"].isnan()
        if nans.any():
            inputs["pixel_values"][nans] = 0
            inputs["pixel_values"] = inputs["pixel_values"].float()


class _VideoImage(evts.Image):

    start: float = 0.0
    timeline: str = "fake"
    duration: float = 1.0
    video: tp.Any
    time: float = 0.0
    filepath: str = ""

    def model_post_init(self, log__: tp.Any) -> None:
        self.filepath = f"{self.video.filename}:{self.time:.3f}"
        super().model_post_init(log__)

    def _read(self) -> tp.Any:
        import PIL

        with ignore_all():
            img = self.video.get_frame(self.time)
        return PIL.Image.fromarray(img.astype("uint8"))


class VJEPA2(pydantic.BaseModel):
    _event_types_helper: EventTypesHelper
    _missing_default: torch.Tensor | None = None
    model_config = pydantic.ConfigDict(protected_namespaces=(), extra="forbid")
    layers: list[float] = [0.5, 0.75, 1.0]
    layer_aggregation: tp.Literal["group_mean"] | None = "group_mean"

    name: tp.Literal["VJEPA2"] = "VJEPA2"
    device: tp.Literal["auto", "cpu", "cuda"] = "auto"

    _model: nn.Module = pydantic.PrivateAttr()

    infra: MapInfra = MapInfra()

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: tp.Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)

        super().__init_subclass__()

    def model_post_init(self, log__: tp.Any) -> None:
        super().model_post_init(log__)
        self._event_types_helper = EventTypesHelper("Video")
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
        event_types = self._event_types_helper.classes
        name = self.__class__.__name__
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

    def _exclude_from_cache_uid(self) -> list[str]:
        return ["device"] + ["layers", "layer_aggregation"]

    def _get_timed_arrays(
        self, events: list[evts.Video], start: float, duration: float
    ) -> tp.Iterable[TimedArray]:
        for event, latent in zip(events, self._get_data(events)):
            freq = 2.0
            ta: TimedArray = TimedArray(
                data=latent,
                frequency=freq,
                start=event.start,
                duration=event.duration,
            )

            sub = ta.overlap(start=start, duration=duration)
            if sub is None:

                sub = ta.overlap(start=ta.start, duration=0)
            sub.data = self._aggregate_layers(sub.data)
            yield sub

    @infra.apply(
        item_uid=lambda event: f"{event.filepath}_{event.offset:.2f}_{event.duration:.2f}",
        exclude_from_cache_uid="method:_exclude_from_cache_uid",
        cache_type="MemmapArrayFile",
    )
    def _get_data(self, events: tp.List[evts.Video]) -> tp.Iterator[np.ndarray]:
        logging.getLogger("data_utils").setLevel(logging.DEBUG)

        model = VideoModel()
        if model.model.device.type == "cpu":
            model.model.to(self.device)

        subtimes = list(
            k / model.num_frames * 4.0 for k in reversed(range(model.num_frames))
        )

        for event in events:
            video = event.read()
            expect_frames = Frequency(2.0).to_ind(event.duration)
            logger.debug(
                "Loaded Video (duration %ss at %sfps, shape %s):\n%s",
                video.duration,
                video.fps,
                tuple(video.size),
                event.filepath,
            )

            times = np.linspace(0, video.duration, expect_frames + 1)[1:]

            output = np.array([])

            for k, t in tqdm(enumerate(times), total=len(times), desc="Encoding video"):
                ims = [_VideoImage(video=video, time=max(0, t - t2)) for t2 in subtimes]
                data = np.array([np.array(i.read()) for i in ims])
                t_embd = model.predict_hidden_states(data)
                t_embd = t_embd[0]

                embd = t_embd.mean(axis=1).cpu().numpy()
                if not output.size:
                    output = np.zeros((len(times),) + embd.shape)
                    logger.debug("Created Tensor with size %s", output.shape)
                output[k] = embd
            video.close()

            output = output.transpose(list(range(1, output.ndim)) + [0])
            yield output


class VideoModel:
    def __init__(
        self,
    ) -> None:
        super().__init__()
        from transformers import AutoModel as Model
        from transformers import AutoVideoProcessor as Processor

        self.model = Model.from_pretrained(
            "facebook/vjepa2-vitg-fpc64-256", output_hidden_states=True
        )
        self.model.eval()

        self.processor = Processor.from_pretrained(
            "facebook/vjepa2-vitg-fpc64-256", do_rescale=True
        )
        self.num_frames = 64

    def predict(self, images: np.ndarray) -> tp.Any:
        kwargs: dict[str, tp.Any] = {"text": "", "return_tensors": "pt"}
        field = "videos"
        del kwargs["text"]
        kwargs[field] = list(images)
        inputs = self.processor(**kwargs)

        _fix_pixel_values(inputs)
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            pred = self.model(**inputs)
        return pred

    def predict_hidden_states(self, images: np.ndarray) -> torch.Tensor:
        pred = self.predict(images)
        states = pred.hidden_states
        out = torch.cat([x.unsqueeze(1) for x in states], axis=1)
        return out
