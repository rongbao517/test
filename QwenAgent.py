from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from Constant import *
import torch
from Agent import Agent
import json_repair
from openai import OpenAI
import traceback


class QwenAgent(Agent):
    def __init__(self, model_name, system_prompt, task="", device="cuda", max_new_tokens=512
                 , temperature=0.7, seed=10086, is_api=True):
        Agent.__init__(self, model_name, temperature, seed)
        assert model_name in [QWEN_AGENT_7B_NAME, QWEN_AGENT_72B_NAME,QWEN_2_5_AGENT_7B_NAME], "QwenAgent do not accept " + model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.model_name = model_name
        self.temperature = temperature
        self.is_api = is_api
        if not self.is_api:
            print("try to load model at " + MODEL_CACHE_FILE + model_name)
            self.model = AutoModelForCausalLM.from_pretrained(MODEL_CACHE_FILE + model_name, torch_dtype="auto", device_map="auto",
                                                              local_files_only=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE_FILE + model_name)
            # for the output to contain attention score

            # self.model.config.attn_implementation = "eager"
            self.model.set_attn_implementation('eager')
            
            # end of the setting for attention score
        else:
            print("init model interface for " + model_name)
            self.model = OpenAI(api_key=MY_API_KEY, base_url=LOCAL_MODEL_SERVICE_FOR_406, timeout=30000)
        self.system_prompt = system_prompt # or task description
        self.seed = seed
        self.task = task
        set_seed(self.seed)

    def query(self, prompt, plain=False, print_prompt=True):
        prompt = prompt[:32768] # in case, prompt is too long
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]
        if print_prompt:
            print(messages)
        if not self.is_api:
            with torch.no_grad():
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
                generated_ids = self.model.generate(**model_inputs, max_new_tokens=self.max_new_tokens,
                                                    temperature=self.temperature)
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        else:
            resp = self.model.chat.completions.create(
                model=self.model_name.replace("Qwen/", ""),
                messages=messages,
                temperature=self.temperature,
                seed=self.seed,
                # max_tokens=4096,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                # logprobs=True,
                # top_logprobs=5
            )
            response = resp.choices[0].message.content
        response = response.lstrip("```json").rstrip("```")

        res = None
        try:
            if plain:
                res = response
            else:
                res = json_repair.loads(response)
        except Exception as e:
            traceback.print_exc()
            print(response)
            exit(-1)
        return res

    def get_response(self, prompt, plain=False):
        return self.query(prompt, plain)

    def set_system_prompt(self, str):
        self.system_prompt = str

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