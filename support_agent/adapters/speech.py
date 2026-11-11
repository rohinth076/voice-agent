"""Text-to-speech for any endpoint that follows the OpenAI shape: POST {base_url}/audio/speech.

LiveKit's own OpenAI plugin adds a streaming field some compatible servers reject, so this sends only the standard
fields: model, voice, input, response_format.
"""
import asyncio
from dataclasses import dataclass

import aiohttp
from livekit.agents import APIConnectionError, APIConnectOptions, APIStatusError, APITimeoutError, tts, utils
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

_MIME = {"wav": "audio/wav", "mp3": "audio/mpeg", "opus": "audio/ogg", "flac": "audio/flac", "aac": "audio/aac"}


@dataclass
class _Opts:
    base_url: str
    api_key: str
    model: str
    voice: str
    response_format: str


class SpeechTTS(tts.TTS):
    def __init__(self, *, base_url: str, api_key: str, model: str, voice: str,
                 response_format: str = "wav", sample_rate: int = 24000) -> None:
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=sample_rate, num_channels=1)
        self._opts = _Opts(base_url.rstrip("/"), api_key, model, voice, response_format)
        self._session: aiohttp.ClientSession | None = None

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "openai-compatible"

    def _http(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    def synthesize(self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS) -> "_Stream":
        return _Stream(tts=self, input_text=text, conn_options=conn_options)


class _Stream(tts.ChunkedStream):
    def __init__(self, *, tts: SpeechTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: SpeechTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        o = self._tts._opts
        try:
            async with self._tts._http().post(
                f"{o.base_url}/audio/speech",
                headers={"Authorization": f"Bearer {o.api_key}"},
                json={"model": o.model, "voice": o.voice, "input": self._input_text, "response_format": o.response_format},
                timeout=aiohttp.ClientTimeout(total=30, sock_connect=self._conn_options.timeout),
            ) as resp:
                if resp.status >= 400:
                    raise APIStatusError(message=(await resp.text())[:300], status_code=resp.status, request_id=None, body=None)
                output_emitter.initialize(request_id=utils.shortuuid(), sample_rate=self._tts.sample_rate, num_channels=1,
                                          mime_type=_MIME.get(o.response_format, f"audio/{o.response_format}"))
                async for data, _ in resp.content.iter_chunks():
                    output_emitter.push(data)
                output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIStatusError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
