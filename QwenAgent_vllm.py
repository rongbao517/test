from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from Constant import *
import torch
from Agent import Agent
import json_repair
from openai import OpenAI
import traceback

from typing import List
from vllm import LLM, SamplingParams
import os
import json






class QwenVllmAgent(Agent):
    """
    An agent that uses Qwen models for inference, accelerated by the vLLM engine
    for parallel processing of multiple prompts. It can also fall back to a standard
    OpenAI-compatible API if needed.
    """
    def __init__(self,
                 model_name: str,
                 system_prompt: str,
                 task: str = "",
                 device_parallel=1,
                 max_new_tokens: int = 2048,
                 temperature: float = 0.7,
                 top_p: float = 0.95,
                 frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0,
                 seed: int = 10086,
                 is_api: bool = False):
        """
        Initializes the QwenVllmAgent.

        Args:
            model_name (str): The name of the Qwen model to use.
            system_prompt (str): The system prompt or task description.
            task (str): A description of the task.
            max_new_tokens (int): The maximum number of new tokens to generate.
            temperature (float): The sampling temperature.
            top_p (float): The nucleus sampling probability.
            frequency_penalty (float): Penalty for frequent tokens.
            presence_penalty (float): Penalty for existing tokens.
            seed (int): The random seed for reproducibility.
            is_api (bool): If True, use an OpenAI-compatible API. If False, use local vLLM.
        """
        super().__init__(model_name, temperature, seed)
        assert model_name in [QWEN_AGENT_7B_NAME, QWEN_AGENT_72B_NAME,QWEN_2_5_AGENT_7B_NAME,QWEN_2_5_AGENT_72B_NAME], "QwenAgent do not accept " + model_name
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.task = task
        self.is_api = is_api
        self.device_parallel = device_parallel

        # --- vLLM Sampling Parameters ---
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            max_tokens=max_new_tokens
        )

        if not self.is_api:
            print(f"Initializing vLLM engine for model: {self.model_name}")
            # self.model = AutoModelForCausalLM.from_pretrained(MODEL_CACHE_FILE + model_name, torch_dtype="auto", device_map="auto",
                                                            #   local_files_only=True).to(self.device)
            model_path = os.path.join(MODEL_CACHE_FILE, model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE_FILE + model_name)
            # Load the model using the vLLM engine
            self.model = LLM(model=model_path, seed=self.seed, trust_remote_code=True,tensor_parallel_size=self.device_parallel,gpu_memory_utilization=0.8,)
            print("vLLM engine loaded successfully.")
        else:
            # This part remains for API-based inference (processes prompts sequentially)
            print(f"Initializing API interface for model: {self.model_name}")
            from openai import OpenAI
            self.model = OpenAI(api_key=MY_API_KEY, base_url=LOCAL_MODEL_SERVICE_FOR_406, timeout=30000)

    def query(self, prompts: List[str], plain: bool = False, print_prompt: bool = True) -> List:
        """
        Processes a batch of prompts in parallel using the vLLM engine.

        Args:
            prompts (List[str]): A list of user prompts to process.
            plain (bool): If True, returns the raw text response. If False, attempts to parse as JSON.
            print_prompt (bool): If True, prints the formatted prompt for the first message.

        Returns:
            List: A list of responses, either as parsed JSON objects or plain text strings.
                  Returns None for any prompt that fails to generate a valid response.
        """
        if not isinstance(prompts, list):
            raise TypeError("The 'prompts' argument must be a list of strings.")

        # Handle the API case by iterating and calling one by one
        if self.is_api:
            results = []
            for p in prompts:
                results.append(self._query_single_api(p, plain, print_prompt))
            return results

        # --- vLLM Parallel Inference ---
        formatted_prompts = []
        for prompt in prompts:
            # Truncate long prompts to avoid errors
            prompt = prompt[:32768]
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ]
            # Apply the model's specific chat template
            formatted_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted_text)

        if print_prompt and formatted_prompts:
            print("--- Example Formatted Prompt ---")
            print(formatted_prompts[0])
            print("---------------------------------")

        # Generate responses for all prompts in a single batch
        outputs = self.model.generate(formatted_prompts, self.sampling_params)

        results = []
        for output in outputs:
            response_text = output.outputs[0].text
            # Clean up potential markdown formatting for JSON
            response_text = response_text.strip().lstrip("```json").rstrip("```")

            try:
                if plain:
                    res = response_text
                else:
                    # Use json_repair to handle potentially malformed JSON
                    res = json_repair.loads(response_text)
                results.append(res)
            except Exception:
                print("--- FAILED TO PARSE RESPONSE ---")
                traceback.print_exc()
                print(response_text)
                print("--------------------------------")
                results.append(None) # Append None for failed responses

        return results
    
    def _query_single_api(self, prompt: str, plain: bool, print_prompt: bool):
        """Helper method to query a single prompt via the OpenAI API."""
        prompt = prompt[:32768]
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        if print_prompt:
            print(messages)

        try:
            resp = self.model.chat.completions.create(
                model=self.model_name.replace("Qwen/", ""),
                messages=messages,
                temperature=self.sampling_params.temperature,
                seed=self.seed,
                max_tokens=self.sampling_params.max_tokens,
                top_p=self.sampling_params.top_p,
                frequency_penalty=self.sampling_params.frequency_penalty,
                presence_penalty=self.sampling_params.presence_penalty,
            )
            response = resp.choices[0].message.content
            response = response.lstrip("```json").rstrip("```")
            
            if plain:
                return response
            else:
                return json_repair.loads(response)
        except Exception as e:
            traceback.print_exc()
            print(f"API call failed for prompt: {prompt}")
            return None


    def set_system_prompt(self, new_prompt: str):
        """Updates the system prompt."""
        self.system_prompt = new_prompt

    # def read(self, content: str):
    #     """
    #     Adds content to the model's input context without generating a response.
    #     This content will be used in subsequent calls to speak().
        
    #     Args:
    #         content (str): The text content to be added to the conversation history.
    #     """
    #     self.history.append({"role": "user", "content": content})
    #     print(f"Agent has read the following content into context:\n---\n{content}\n---")
    # # --- NEW FUNCTION END ---

    # # --- NEW FUNCTION START ---
    # def speak(self, prompt: str, plain: bool = False, print_prompt: bool = True):
    #     """
    #     Generates a response based on the entire conversation history (including system prompt
    #     and content from read()), and updates the history with the new exchange.

    #     Args:
    #         prompt (str): The user's latest prompt or question.
    #         plain (bool): If True, returns the raw string response. Otherwise, attempts to parse as JSON.
    #         print_prompt (bool): If True, prints the messages sent to the model.

    #     Returns:
    #         The model's response, either as a parsed JSON object or a raw string.
    #     """
    #     self.history.append({"role": "user", "content": prompt})

    #     if print_prompt:
    #         print(">>> Sending the following conversation history to the model:")
    #         print(self.history)
            
    #     response_content = ""
    #     if not self.is_api:
    #         # Local model inference
    #         with torch.no_grad():
    #             text = self.tokenizer.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
    #             # A simple truncation if the full history is too long for the model
    #             if len(text) > 32768:
    #                 text = text[-32768:]
                
    #             model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
    #             generated_ids = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens,
    #                                                 temperature=self.temperature)
    #             generated_ids = [
    #                 output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    #             ]
    #             response_content = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    #     else:
    #         # API-based model inference
    #         resp = self.model.chat.completions.create(
    #             model=self.model_name.replace("Qwen/", ""),
    #             messages=self.history,
    #             temperature=self.temperature,
    #             seed=self.seed,
    #             top_p=0.95,
    #             frequency_penalty=0,
    #             presence_penalty=0,
    #         )
    #         response_content = resp.choices[0].message.content
        
    #     # Add the assistant's response to the history to maintain context
    #     self.history.append({"role": "assistant", "content": response_content})

    #     # Process and return the response
    #     response_content = response_content.lstrip("```json").rstrip("```")
    #     res = None
    #     try:
    #         if plain:
    #             res = response_content
    #         else:
    #             res = json_repair.loads(response_content)
    #     except Exception as e:
    #         traceback.print_exc()
    #         print(f"Failed to parse model response:\n{response_content}")
    #         exit(-1)
    #     return res
    # # --- NEW FUNCTION END ---

    # # --- NEW FUNCTION START ---
    # def forget(self):
    #     """
    #     Resets the conversation history, retaining only the initial system prompt.
    #     """
    #     print("Agent is forgetting the conversation history...")
    #     self.history = [{"role": "system", "content": self.system_prompt}]
    #     print("History reset. Agent only remembers the system prompt.")