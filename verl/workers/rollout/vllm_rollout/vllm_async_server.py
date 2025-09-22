# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from collections.abc import AsyncGenerator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, cast
from collections.abc import Sequence as GenericSequence
from vllm.sequence import Logprob
from vllm.transformers_utils.tokenizer import AnyTokenizer
import time
import cloudpickle
import ray
import json
import os
from omegaconf import DictConfig
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse

import vllm
vllm.plugins.load_general_plugins()

from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.logger import RequestLogger
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ChatCompletionResponse, CompletionRequest, CompletionResponse, ErrorResponse, CompletionLogProbs
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_models import BaseModelPath, OpenAIServingModels
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.executor.abstract import Executor
from vllm.worker.worker_base import WorkerWrapperBase
from vllm.distributed.device_communicators.cuda_communicator import (
            CudaCommunicator)
from vllm.sampling_params import BeamSearchParams, SamplingParams
from verl.utils.fs import copy_to_local
from verl.workers.rollout.async_server import AsyncServerBase
from verl.workers.rollout.vllm_rollout.monkey_patch import all_reduce
from vllm.inputs.data import (EmbedsPrompt, TokensPrompt, is_embeds_prompt,
                              is_tokens_prompt)
from vllm.entrypoints.openai.protocol import (CompletionRequest,
                                              CompletionResponse,
                                              ErrorResponse,
                                              RequestResponseMetadata
                                              )
from typing_extensions import assert_never
from vllm.entrypoints.utils import get_max_tokens
from vllm.utils import merge_async_iterators
from vllm.outputs import RequestOutput
from vllm.entrypoints.openai.serving_engine import (OpenAIServing,
                                                    TextTokensPrompt,
                                                    clamp_prompt_logprobs,
                                                    is_text_tokens_prompt)
import jinja2
import asyncio
logger = logging.getLogger(__file__)

try:
    from arctic_inference.common.suffix_cache import SuffixCache
    logger.info("Successfully imported SuffixCache from arctic_inference.common.suffix_cache")
except ImportError as e:
    logger.warning(f"Failed to import SuffixCache from arctic_inference.common.suffix_cache: {e}")
    SuffixCache = None

CudaCommunicator.all_reduce = all_reduce


class ExternalRayDistributedExecutor(Executor):
    """An executor that engines are launched by external ray actors."""

    uses_ray: bool = False

    def _init_executor(self) -> None:
        assert self.vllm_config.instance_id is not None, "instance_id must be set for external ray actors."

        fields = self.vllm_config.instance_id.split(":")
        assert len(fields) == 4, f"instance_id: {self.vllm_config.instance_id} must be in the format of <namespace>:<wg_prefix>:<vllm_dp_size>:<vllm_dp_rank>."
        namespace, wg_prefix, vllm_dp_size, vllm_dp_rank = fields[0], fields[1], int(fields[2]), int(fields[3])

        # Make sure subprocess in same namespace as parent actor.
        # actor name format: {name_prefix}WorkerDict_{pg_idx}:{local_rank}
        ray.init(namespace=namespace)
        actor_names = [actor_name for actor_name in ray.util.list_named_actors() if actor_name.startswith(f"{wg_prefix}WorkerDict") or actor_name.startswith(f"{wg_prefix}ActorRolloutRefWorker")]

        vllm_tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        assert len(actor_names) == vllm_dp_size * vllm_tp_size, f"instance_id: {self.vllm_config.instance_id} has {len(actor_names)} actors, but vllm_dp_size: {vllm_dp_size} * vllm_tp_size: {vllm_tp_size} = {vllm_dp_size * vllm_tp_size} is expected."

        def get_pg_index_and_local_rank(actor_name) -> Tuple[int, int]:
            fields = actor_name.split(":")
            assert len(fields) == 2, f"invalid actor name: {actor_name}"
            pg_index, local_rank = int(fields[0].split("_")[-1]), int(fields[1])
            return pg_index, local_rank

        # sort actor names by pg_index and local_rank
        actor_names = sorted(actor_names, key=get_pg_index_and_local_rank)
        actor_names = actor_names[vllm_dp_rank * vllm_tp_size : (vllm_dp_rank + 1) * vllm_tp_size]
        self.workers: List[WorkerWrapperBase] = [ray.get_actor(actor_name) for actor_name in actor_names]
        print(f"instance_id: {self.vllm_config.instance_id} intializes with external actors: {actor_names}")

        kwargs = dict(
            vllm_config=self.vllm_config,
            local_rank=None,
            rank=None,
            distributed_init_method="env://",
            is_driver_worker=True,
        )
        self.collective_rpc("init_worker", args=([kwargs],))
        self.collective_rpc("init_device")
        self.collective_rpc("load_model")
        print(f"instance_id: {self.vllm_config.instance_id} intializes finished.")

    def collective_rpc(
        self,
        method: Union[str, Callable],
        timeout: Optional[float] = None,
        args: Tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        # TODO(wuxibin): support ray compiled graph
        if isinstance(method, str):
            sent_method = method
        else:
            sent_method = cloudpickle.dumps(method)
        del method

        # ~3ms overhead per schedule step due to SchedulerOutput/ModelRunnerOutput serialization/deserialization.
        outputs = ray.get([worker.execute_method.remote(sent_method, *args, **(kwargs or {})) for worker in self.workers])
        return outputs

    def check_health(self):
        return

class OpenAIServingChatCompletionsWithProblemID(OpenAIServingCompletion):
    """Extension of OpenAIServingChat to handle problem_ids."""
    
    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Optional[Request] = None,
        problem_id: Optional[str] = None,
    ) -> Union[AsyncGenerator[str, None], CompletionResponse, ErrorResponse]:
        """Completion API similar to OpenAI's API.

        See https://platform.openai.com/docs/api-reference/completions/create
        for the API specification. This API mimics the OpenAI Completion API.

        NOTE: Currently we do not support the following feature:
            - suffix (the language models we currently support do not support
            suffix)
        """
        error_check_ret = await self._check_model(request)
        if error_check_ret is not None:
            return error_check_ret

        # If the engine is dead, raise the engine's DEAD_ERROR.
        # This is required for the streaming case, where we return a
        # success status before we actually start generating text :).
        if self.engine_client.errored:
            raise self.engine_client.dead_error

        # Return error for unsupported features.
        if request.suffix is not None:
            return self.create_error_response(
                "suffix is not currently supported")

        if request.echo and request.prompt_embeds is not None:
            return self.create_error_response(
                "Echo is unsupported with prompt embeds.")

        request_id = f"cmpl-{self._base_request_id(raw_request)}"
        created_time = int(time.time())

        request_metadata = RequestResponseMetadata(request_id=request_id)
        if raw_request:
            raw_request.state.request_metadata = request_metadata

        try:
            (
                lora_request,
                prompt_adapter_request,
            ) = self._maybe_get_adapters(request)

            tokenizer = await self.engine_client.get_tokenizer(lora_request)

            request_prompts, engine_prompts = await self._preprocess_completion(
                request,
                tokenizer,
                request.prompt,
                truncate_prompt_tokens=request.truncate_prompt_tokens,
                add_special_tokens=request.add_special_tokens,
            )
        except ValueError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except TypeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except RuntimeError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))
        except jinja2.TemplateError as e:
            logger.exception("Error in preprocessing prompt inputs")
            return self.create_error_response(str(e))

        # Schedule the request and get the result generator.
        generators: list[AsyncGenerator[RequestOutput, None]] = []
        try:
            for i, engine_prompt in enumerate(engine_prompts):
                sampling_params: Union[SamplingParams, BeamSearchParams]
                # Mypy does not infer that engine_prompt will have only one of
                # "prompt_token_ids" or "prompt_embeds" defined, and both of
                # these as Union[object, the expected type], where it infers
                # object if engine_prompt is a subclass of one of the
                # typeddicts that defines both keys. Worse, because of
                # https://github.com/python/mypy/issues/8586, mypy does not
                # infer the type of engine_prompt correctly because of the
                # enumerate. So we need an unnecessary cast here.
                engine_prompt = cast(Union[EmbedsPrompt, TokensPrompt],
                                     engine_prompt)
                if is_embeds_prompt(engine_prompt):
                    input_length = len(engine_prompt["prompt_embeds"])
                elif is_tokens_prompt(engine_prompt):
                    input_length = len(engine_prompt["prompt_token_ids"])
                else:
                    assert_never(engine_prompt)

                if self.default_sampling_params is None:
                    self.default_sampling_params = {}

                max_tokens = get_max_tokens(
                    max_model_len=self.max_model_len,
                    request=request,
                    input_length=input_length,
                    default_sampling_params=self.default_sampling_params)

                if request.use_beam_search:
                    sampling_params = request.to_beam_search_params(
                        max_tokens, self.default_sampling_params)
                else:
                    sampling_params = request.to_sampling_params(
                        max_tokens, self.model_config.logits_processor_pattern,
                        self.default_sampling_params)
                
                # DEBUG: Log skip_special_tokens parameter
                logging.info(f"SamplingParams created: skip_special_tokens={sampling_params.skip_special_tokens}")

                request_id_item = f"{request_id}-{i}"

                self._log_inputs(request_id_item,
                                 request_prompts[i],
                                 params=sampling_params,
                                 lora_request=lora_request,
                                 prompt_adapter_request=prompt_adapter_request)

                trace_headers = (None if raw_request is None else await
                                 self._get_trace_headers(raw_request.headers))

                # Mypy inconsistently requires this second cast in different
                # environments. It shouldn't be necessary (redundant from above)
                # but pre-commit in CI fails without it.
                engine_prompt = cast(Union[EmbedsPrompt, TokensPrompt],
                                     engine_prompt)
                if isinstance(sampling_params, BeamSearchParams):
                    generator = self.engine_client.beam_search(
                        prompt=engine_prompt,
                        request_id=request_id,
                        params=sampling_params,
                        lora_request=lora_request,
                    )
                else:
                    generator = self.engine_client.generate(
                        engine_prompt,
                        sampling_params,
                        request_id_item,
                        lora_request=lora_request,
                        prompt_adapter_request=prompt_adapter_request,
                        trace_headers=trace_headers,
                        priority=request.priority,
                        problem_ids=problem_id
                    )

                generators.append(generator)
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        result_generator = merge_async_iterators(*generators)

        model_name = self._get_model_name(request.model, lora_request)
        num_prompts = len(engine_prompts)

        # Similar to the OpenAI API, when n != best_of, we do not stream the
        # results. Noting that best_of is only supported in V0. In addition,
        # we do not stream the results when use beam search.
        stream = (request.stream
                  and (request.best_of is None or request.n == request.best_of)
                  and not request.use_beam_search)

        # Streaming response
        if stream:
            return self.completion_stream_generator(
                request,
                request_prompts,
                result_generator,
                request_id,
                created_time,
                model_name,
                num_prompts=num_prompts,
                tokenizer=tokenizer,
                request_metadata=request_metadata,
                enable_force_include_usage=self.enable_force_include_usage)

        # Non-streaming response
        final_res_batch: list[Optional[RequestOutput]] = [None] * num_prompts
        try:
            async for i, res in result_generator:
                final_res_batch[i] = res

            for i, final_res in enumerate(final_res_batch):
                assert final_res is not None

                # The output should contain the input text
                # We did not pass it into vLLM engine to avoid being redundant
                # with the inputs token IDs
                if final_res.prompt is None:
                    request_prompt = request_prompts[i]
                    if is_text_tokens_prompt(request_prompt):
                        final_res.prompt = request_prompt["prompt"]
                    else:
                        final_res.prompt = None

            final_res_batch_checked = cast(list[RequestOutput],
                                           final_res_batch)

            response = self.request_output_to_completion_response(
                final_res_batch_checked,
                request,
                request_id,
                created_time,
                model_name,
                tokenizer,
                request_metadata,
            )
        except asyncio.CancelledError:
            return self.create_error_response("Client disconnected")
        except ValueError as e:
            # TODO: Use a vllm-specific Validation Error
            return self.create_error_response(str(e))

        # When user requests streaming but we don't stream, we still need to
        # return a streaming response with a single event.
        if request.stream:
            response_json = response.model_dump_json()

            async def fake_stream_generator() -> AsyncGenerator[str, None]:
                yield f"data: {response_json}\n\n"
                yield "data: [DONE]\n\n"

            return fake_stream_generator()

        return response


class SuffixCacheManager:
    def __init__(self, config):
        self.suffix_generation_id = 0
        self.generation_cache_store = {}
        self.speculative_config = config.get("engine_kwargs", {}).get("vllm", {}).get("speculative_config", None)
        self.suffix_cache_data_path = config.get("suffix_cache_data_path", None)
        
    def update_suffix_generation_id(self):
        self.suffix_generation_id += 1
        return self.suffix_generation_id
    
    def should_use_suffix_cache(self):
        return (self.speculative_config and 
                self.speculative_config.get("enable_suffix_decoding", False))
    
    async def prepare_suffix_cache(self, problem_ids):
        if not self.should_use_suffix_cache():
            return None
        if self.suffix_generation_id in self.generation_cache_store:
            print("Reusing existing suffix cache for generation_id:", self.suffix_generation_id)
            return self.generation_cache_store.pop(self.suffix_generation_id)
        print("Cache Miss: Building new suffix cache for generation_id:", self.suffix_generation_id)
        suffix_cache = await self._build_suffix_cache(problem_ids)
        self.generation_cache_store[self.suffix_generation_id] = suffix_cache
        return suffix_cache

    async def prepare_cache_data(self, problem_ids):
        """Prepare cache data (not objects) for distribution to workers"""
        if self.suffix_generation_id in self.generation_cache_store:
            return self.generation_cache_store.pop(self.suffix_generation_id)
        
        cache_data = self._prepare_cache_data(problem_ids)
        self.generation_cache_store[self.suffix_generation_id] = cache_data
        return cache_data

    def _prepare_cache_data(self, problem_ids):
        """Convert problem_ids to cache data format"""
        problem_id_to_sequences = self._load_suffix_cache_data_for_problem_ids(problem_ids)
        return [(pid, None, seqs) for pid, seqs in problem_id_to_sequences.items()]

    def _get_cache_params(self):
        """Get cache parameters for worker initialization"""
        return {
            "max_depth": self.speculative_config.get("suffix_cache_max_depth", 64),
            "thread_safe": True,
            "max_threads": self.speculative_config.get("suffix_cache_max_threads", 8)
        }
    
    async def _build_suffix_cache(self, problem_ids):
        if SuffixCache is None:
            return None
        
        suffix_cache_max_depth = self.speculative_config.get("suffix_cache_max_depth", 64)
        suffix_cache_max_threads = self.speculative_config.get("suffix_cache_max_threads", 8)

        unique_problem_ids = list(set(problem_ids))
        suffix_cache = SuffixCache(
            max_depth=suffix_cache_max_depth, 
            thread_safe=True, 
            max_threads=suffix_cache_max_threads
        )
        
        problem_id_to_sequences = self._load_suffix_cache_data_for_problem_ids(unique_problem_ids)
        if not problem_id_to_sequences:
            return suffix_cache
            
        problems_data = []
        for problem_id in unique_problem_ids:
            if problem_id in problem_id_to_sequences:
                sequences = problem_id_to_sequences[problem_id]
                problems_data.append((problem_id, None, sequences))
        
        if problems_data:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, suffix_cache.prebuild_problems_parallel, problems_data)
        
        return suffix_cache
    
    def _load_suffix_cache_data_for_problem_ids(self, problem_ids):
        if not self.suffix_cache_data_path or not problem_ids:
            return {}
            
        problem_id_to_sequences = {}
        if not os.path.exists(self.suffix_cache_data_path):
            return {}
            
        files_to_process = []
        if os.path.isdir(self.suffix_cache_data_path):
            for filename in os.listdir(self.suffix_cache_data_path):
                if filename.endswith('.jsonl'):
                    files_to_process.append(os.path.join(self.suffix_cache_data_path, filename))
        elif os.path.isfile(self.suffix_cache_data_path):
            files_to_process = [self.suffix_cache_data_path]
        
        if not files_to_process:
            return {}
            
        try:
            for file_path in files_to_process:
                with open(file_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            problem_id = entry.get('problem_id')
                            if problem_id in problem_ids:
                                sequences = entry.get('sequences', [])
                                if problem_id not in problem_id_to_sequences:
                                    problem_id_to_sequences[problem_id] = []
                                problem_id_to_sequences[problem_id].extend(sequences)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line in {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load suffix cache data: {e}")
                    
        return problem_id_to_sequences

@ray.remote(num_cpus=1)
class AsyncvLLMServer(AsyncServerBase):
    """
    AsyncvLLMServer is a wrapper for AsyncLLM, it uses ExternalRayDistributedExecutor to launch engines
    in hybrid rollout workers, i.e AsyncActorRolloutRefWorker.

    AsyncvLLMServer works as follows:
    1. Start FastAPI server first.
    2. Initialize AsyncLLM with ExternalRayDistributedExecutor.
    3. AsyncLLM spawn EngineCore in subprocess.
    4. EngineCore initialize ExternalRayDistributedExecutor.
    5. ExternalRayDistributedExecutor lookup its corresponding actors by name.
    6. ExternalRayDistributedExecutor init executor: init_worker, init_device, load_model.

    For vLLM AsyncLLM design, see: https://github.com/vllm-project/vllm/pull/9826
    """

    def __init__(self, config: DictConfig, vllm_dp_size: int, vllm_dp_rank: int, wg_prefix: str, enable_lmcache: bool = False):
        """
        Args:
            config: DictConfig, actor_rollout_ref config.
            vllm_dp_size: int, vllm data parallel size.
            vllm_dp_rank: int, vllm data parallel rank.
            wg_prefix: str, worker group prefix, used to lookup actors.
            enable_lmcache: bool, whether to enable LMCache support.
        """
        super().__init__()

        self.config = config
        self.vllm_dp_size = vllm_dp_size
        self.vllm_dp_rank = vllm_dp_rank
        self.wg_prefix = wg_prefix
        self.enable_lmcache = enable_lmcache
        self.engine: Optional[AsyncLLM] = None
        self.suffix_cache_manager = SuffixCacheManager(self.config.rollout)
        self.lmcache_config_file = None

    async def init_engine(self):
        """Init vLLM AsyncLLM engine."""
        config = self.config
        model_path = config.model.path
        model_name = "/".join(model_path.split("/")[-2:])
        local_path = copy_to_local(model_path)
        trust_remote_code = config.model.get("trust_remote_code", False)
        rollout_config = config.rollout


        if self.enable_lmcache:
            self._setup_lmcache_config()

        tensor_parallel_size = rollout_config.get("tensor_model_parallel_size", 1)
        max_model_len = rollout_config.max_model_len if rollout_config.max_model_len else rollout_config.prompt_length + rollout_config.response_length
        max_model_len = max(max_model_len, 32768)
        max_num_batched_tokens = max(rollout_config.get("max_num_batched_tokens", 32768), max_model_len)

        # Override default generation config from hugging face model config,
        # user can still override them by passing kwargs in each request.
        kwargs = dict(
            n=1,
            logprobs=0,
            max_tokens=rollout_config.response_length,
        )
        for k in rollout_config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = rollout_config.get(k)
        print(f"override_generation_config: {kwargs}")
        # Note: added for suffix cache
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        engine_kwargs = {
            "model": local_path,
            "enable_sleep_mode": True,
            "override_generation_config": kwargs,
            "tensor_parallel_size": tensor_parallel_size,
            "distributed_executor_backend": ExternalRayDistributedExecutor,
            "dtype": rollout_config.dtype,
            "enforce_eager": rollout_config.enforce_eager,
            "gpu_memory_utilization": rollout_config.gpu_memory_utilization,
            "disable_custom_all_reduce": True,
            "skip_tokenizer_init": False,
            "max_model_len": max_model_len,
            "load_format": "auto",
            "disable_log_stats": rollout_config.disable_log_stats,
            "max_num_batched_tokens": max_num_batched_tokens,
            "enable_chunked_prefill": rollout_config.enable_chunked_prefill,
            "enable_prefix_caching": True,
            "trust_remote_code": trust_remote_code,
            "seed": self.vllm_dp_rank,
            "max_num_seqs": 256,
            "hf_overrides": {"max_position_embeddings": max_model_len},
        }

        if self.enable_lmcache and self.lmcache_config_file:
            engine_kwargs.update({
                "kv_transfer_config": {
                    "kv_connector": "LMCacheConnectorV1",
                    "kv_role": "kv_both"
                }
            })
        # Get speculative config from rollout_config.engine_kwargs (not from top-level config)
        speculative_config = rollout_config.get("engine_kwargs", {}).get("vllm", {}).get("speculative_config", {})
        print(f"DEBUG: extracted speculative_config: {speculative_config}")
        config_engine_kwargs = {
            "speculative_config": {
                "method": speculative_config.get("method", None),
                "enable_suffix_decoding": speculative_config.get("enable_suffix_decoding", False),
                "suffix_cache_max_depth": speculative_config.get("suffix_cache_max_depth", 64),
                "disable_by_batch_size": speculative_config.get("disable_by_batch_size", None),
                "num_speculative_tokens": speculative_config.get("num_speculative_tokens", None),
                "model": speculative_config.get("model", None),
                "suffix_max_spec_factor": speculative_config.get("suffix_max_spec_factor", 2.0),
            }
        }
        print(f"Speculative Config: {config_engine_kwargs['speculative_config']}")
        engine_kwargs.update(config_engine_kwargs)

        engine_args = AsyncEngineArgs(**engine_kwargs)

        # init async llm engine
        vllm_config = engine_args.create_engine_config()
        namespace = ray.get_runtime_context().namespace
        vllm_config.instance_id = f"{namespace}:{self.wg_prefix}:{self.vllm_dp_size}:{self.vllm_dp_rank}"
        self.engine = AsyncLLM.from_vllm_config(vllm_config)

        # build serving chat
        model_config = self.engine.model_config
        BASE_MODEL_PATHS = [BaseModelPath(name=model_name, model_path=model_path)]
        models = OpenAIServingModels(self.engine, model_config, BASE_MODEL_PATHS)
        if rollout_config.chat_template:
            with open(rollout_config.chat_template, "r", encoding="utf-8") as f:
                chat_template_str = f.read()
        else:
            chat_template_str = None
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            models,
            "assistant",
            request_logger=RequestLogger(max_log_len=4096) if not rollout_config.disable_logging else None,
            chat_template=chat_template_str,
            chat_template_content_format="auto",
            #return_tokens_as_token_ids=True,
        )

        self.openai_serving_completion = OpenAIServingChatCompletionsWithProblemID(
            self.engine,
            model_config,
            models,
            request_logger=RequestLogger(max_log_len=4096) if not rollout_config.disable_logging else None,
            return_tokens_as_token_ids=True,
        )

        print(f"Async vLLM Server running at {await self.get_server_address()}")

    def _setup_lmcache_config(self):
        import tempfile
        import yaml
        
        temp_fd, self.lmcache_config_file = tempfile.mkstemp(suffix='.yaml', prefix='lmcache_config_')
        
        config = {
            "chunk_size": 256,
            "local_cpu": True,
            "max_local_cpu_size": 5,
            "remote_url": "lm://localhost:65433",
            "remote_serde": "naive", 
            "save_decode_cache": True,
            "enable_controller": True,
            "lmcache_instance_id": f"lmcache_default_instance_{self.vllm_dp_rank}",
            "controller_url": "localhost:9001",
            "distributed_url": f"localhost:{9100 + self.vllm_dp_rank}",
            "lmcache_worker_port": 7000 + self.vllm_dp_rank,
            "internal_api_server_enabled": True,
            "internal_api_server_port_start": 10100 + self.vllm_dp_rank * 10
        }
        
        with os.fdopen(temp_fd, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        os.environ['LMCACHE_CONFIG_FILE'] = self.lmcache_config_file

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        request_json = await raw_request.json()
        # Extract problem_id if present - will be passed through later steps
        problem_id = request_json.get('problem_id', None)
        if problem_id is not None:
            print(f"Received problem_id: {problem_id}")
        else:
            print(f"No problem_id found in request, {request_json.keys()}")
        
        request = ChatCompletionRequest(**request_json)
        generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())
        
    async def completions(self, raw_request: Request):
        """OpenAI completions API.

        API reference: https://platform.openai.com/docs/api-reference/completions/create
        """
        request_json = await raw_request.json()
        problem_id = request_json.get('problem_id', None)
        if problem_id is None:
            print(f"No problem_id completion found in request, {request_json.keys()}")
        
        request = CompletionRequest(**request_json)
        generator = await self.openai_serving_completion.create_completion(request, raw_request, problem_id=problem_id)

        if isinstance(generator, ErrorResponse):
            return JSONResponse(content=generator.model_dump(), status_code=generator.code)
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, CompletionResponse)
            if generator.choices and generator.choices[0].logprobs:
                generator.choices[0].logprobs.token_logprobs = [] 
                generator.choices[0].logprobs.top_logprobs = []
                generator.choices[0].logprobs.text_offset = []
            return JSONResponse(content=generator.model_dump())

    async def chat_completion_generator(self, request: ChatCompletionRequest) -> AsyncGenerator[Tuple[int, str]]:
        """Direct chat completion without FastAPI.

        Args:
            request: ChatCompletionRequest, request object.

        Returns:
            AsyncGenerator[Tuple[int, str]]: async generator of (status_code, data) pairs.
        """
        generator = await self.openai_serving_chat.create_chat_completion(request)
        if isinstance(generator, ErrorResponse):
            data = generator.model_dump_json(exclude_unset=True)
            yield generator.code, f"data: {data}\n\n"

        if request.stream:
            async for chunk in generator:
                yield 200, chunk
        else:
            assert isinstance(generator, ChatCompletionResponse)
            data = generator.model_dump_json(exclude_unset=True)
            yield 200, f"data: {data}\n\n"

    async def wake_up(self, tags: Optional[list[str]] = None):
        assert self.engine is not None
        await self.engine.wake_up(tags)

    async def sleep(self):
        # TODO: https://github.com/vllm-project/vllm/issues/17103
        assert self.engine is not None
        await self.engine.reset_prefix_cache()
        await self.engine.sleep()


    def update_suffix_generation_id(self):
        return self.suffix_cache_manager.update_suffix_generation_id()

    async def update_cache(self, problem_ids):
        if not self.suffix_cache_manager.should_use_suffix_cache():
            return {"status": "success"}
        
        current_gen_id = self.suffix_cache_manager.update_suffix_generation_id()
        
        if current_gen_id in self.suffix_cache_manager.generation_cache_store:
            cached_data = self.suffix_cache_manager.generation_cache_store.pop(current_gen_id)
            if isinstance(cached_data, dict) and 'cache_params' in cached_data:
                try:
                    await self.engine.collective_rpc("rebuild_cache_sync", args=(current_gen_id, cached_data['cache_params'], cached_data['problems_data']))
                    return {"status": "success"}
                except Exception as e:
                    logger.error(f"Failed to rebuild cache from stored data: {e}")
                    return {"status": "error", "message": str(e)}
        
        try:
            results = await self.engine.collective_rpc("activate_prebuilt_cache", args=(current_gen_id,))
            if all(results):
                return {"status": "success"}
        except Exception as e:
            logger.warning(f"Failed to activate prebuilt cache: {e}. Falling back to sync build.")
        
        cache_params = self.suffix_cache_manager._get_cache_params()
        problems_data = self.suffix_cache_manager._prepare_cache_data(problem_ids)
        
        try:
            await self.engine.collective_rpc("rebuild_cache_sync", args=(current_gen_id, cache_params, problems_data))
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Failed to rebuild cache via collective_rpc: {e}")
            return {"status": "error", "message": str(e)}
    
    async def set_hard_problems(self, hard_problems):
        try:
            await self.engine.collective_rpc("set_hard_problems", args=(hard_problems,))
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Failed to set hard problems via collective_rpc: {e}")
            return {"status": "error", "message": str(e)}

    async def get_acceptance_length_metric_for_problems(self, problem_ids):
        """
        Returns: {
                'avg_acceptance_length': float,
                'problem_metrics': {problem_id: {acceptance_length: [], avg_acceptance_length: float}}
        }
        """
        try:
            metrics = await self.engine.collective_rpc("get_acceptance_length_metric_for_problems", args=(problem_ids,))
            return {"status": "success", "metrics": metrics}
        except Exception as e:
            logger.error(f"Failed to get acceptance length metrics for problems via collective_rpc: {e}")
            return {"status": "error", "message": str(e)}

    async def clear_acceptance_metrics_for_problems(self, problem_ids):
        try:
            await self.engine.collective_rpc("clear_acceptance_metrics_for_problems", args=(problem_ids,))
            return {"status": "success"}
        except Exception as e:
            logger.error(f"Failed to clear acceptance metrics for problems via collective_rpc: {e}")
            return {"status": "error", "message": str(e)}

    async def queue_suffix_prebuild_async(self, batch, context: str, generation_id: int=None):
        """Queue suffix prebuilding work for background processing."""
        logger.info(f"queue_suffix_prebuild_async called: context={context}, generation_id={generation_id}")
        problem_ids = batch.non_tensor_batch.get("problem_id", None) if hasattr(batch, 'non_tensor_batch') else None
        if problem_ids is not None:
            cache_params = self.suffix_cache_manager._get_cache_params()
            problems_data = self.suffix_cache_manager._prepare_cache_data(problem_ids)
            
            self.suffix_cache_manager.generation_cache_store[generation_id] = {
                'cache_params': cache_params,
                'problems_data': problems_data
            }
            
            await self.engine.collective_rpc("prebuild_cache_async", args=(generation_id, cache_params, problems_data))
            logger.info(f"Background prebuild queued for generation_id={generation_id}, problem_ids={len(problem_ids)}")