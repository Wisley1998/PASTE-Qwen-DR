# Copyright Sierra

import json
import copy
import time
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, Future

import numpy as np
from litellm import completion, token_counter
# from sentence_transformers import SentenceTransformer

from tau_bench.agents.base import Agent
from tau_bench.envs.base import Env
from tau_bench.types import SolveResult, Action, RESPOND_ACTION_NAME, RESPOND_ACTION_FIELD_NAME

max_num_steps = 5
background_agent_type = "guesser" # oracle, guesser, preparer

class ToolCallingStaticAgent(Agent):
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        guesser_config: Optional[Dict[str, Any]] = None,
        guesser_check: bool = False,
        baseline_config: list[Dict[str, Any]] = None,
    ):
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future: Optional[Future] = None
        self.guesser_config = guesser_config
        self.guesser_check = guesser_check
        self.baseline_config = baseline_config
        # self.embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-8B") if self.guesser_check else None
        self.embedding_model = None

    def _guesser(self, index: int, user_query_prompt: str, guesser_model: Dict[str, Any]):
        guesser = {
            "id": index,
            "input": "",
            "output": "",
            "input_token_length": 0,
            "output_token_length": 0,
            "price_cost": 0,
            "model": guesser_model,
            "time": 0
        }
        llm_start_time = time.time()
        res = completion(
            messages=[{"role": "system", "content": user_query_prompt}],
            model=guesser_model["model"],
            custom_llm_provider=guesser_model["provider"],
            temperature=guesser_model["temperature"],
            reasoning_effort=guesser_model["reasoning"],
        )
        llm_end_time = time.time()
        llm_duration = llm_end_time - llm_start_time
        guesser["input"] = [{"role": "system", "content": user_query_prompt}]
        guesser["output"] = res.choices[0].message.content
        guesser["input_token_length"] = res.usage.prompt_tokens
        guesser["reasoning_token_length"] = res.usage.completion_tokens_details.reasoning_tokens
        guesser["output_token_length"] = res.usage.completion_tokens - res.usage.completion_tokens_details.reasoning_tokens
        guesser["price_cost"] = res._hidden_params["response_cost"]
        guesser["time"] = llm_duration
        return guesser
    
    def _decider(self, index: int, messages_copy: List[Dict[str, Any]], env: Env, next_message: Dict[str, Any], guesser: Dict[str, Any], guesser_model: Dict[str, Any]):
        messages_copy.extend([next_message, {"role": "user", "content": guesser["output"]}])
        decider = {
            "id": index,
            "trajectory": []
        }
        return_messages = []
        for _ in range(max_num_steps):
            num_retries = 0
            while num_retries < 3:
                try:
                    llm_start_time = time.time()
                    res = completion(
                        messages=messages_copy,
                        model=guesser_model["model"],
                        custom_llm_provider=guesser_model["provider"],
                        tools=self.tools_info,
                        temperature=guesser_model["temperature"],
                        reasoning_effort=guesser_model["reasoning"],
                    )
                    llm_end_time = time.time()
                    llm_duration = llm_end_time - llm_start_time
                except Exception as e:
                    import traceback
                    print(traceback.format_exc())
                    print(f"Decider failed with error: {e}")
                if not res or not res.choices or not res.choices[0].message or not res.usage.completion_tokens_details:
                    num_retries += 1
                    continue
                else:
                    break
                
            next_message = res.choices[0].message.model_dump()
            action = message_to_action(next_message)
            tool_start_time = time.time()
            env_response = env.step(action)
            tool_end_time = time.time()
            tool_duration = tool_end_time - tool_start_time
            if not res.usage.completion_tokens_details:
                reasoning_token_length = 0
            else:
                reasoning_token_length = res.usage.completion_tokens_details.reasoning_tokens
            if action.name != RESPOND_ACTION_NAME:
                decider_log = {
                    "input": messages_copy,
                    "output": next_message,
                    "input_token_length": res.usage.prompt_tokens,
                    "reasoning_token_length": reasoning_token_length,
                    "output_token_length": res.usage.completion_tokens - reasoning_token_length,
                    "price_cost": res._hidden_params["response_cost"],
                    "time": llm_duration,
                    "execution_time": tool_duration,
                    "role": "tool",
                    "tool_call_id": next_message["tool_calls"][0]["id"],
                    "name": next_message["tool_calls"][0]["function"]["name"],
                    "content": env_response.observation
                }
                decider["trajectory"].append(copy.deepcopy(decider_log))
                next_message["tool_calls"] = next_message["tool_calls"][:1]
                tool_message = [
                    next_message,
                    {
                        "role": "tool",
                        "tool_call_id": next_message["tool_calls"][0]["id"],
                        "name": next_message["tool_calls"][0]["function"]["name"] + " (background task)",
                        "content": env_response.observation
                    },
                ]
                messages_copy.extend(
                    tool_message
                )
                return_messages.append(tool_message)
            else:
                decider_log = {
                    "input": messages_copy,
                    "output": next_message,
                    "input_token_length": res.usage.prompt_tokens,
                    "reasoning_token_length": reasoning_token_length,
                    "output_token_length": res.usage.completion_tokens - reasoning_token_length,
                    "price_cost": res._hidden_params["response_cost"],
                    "time": llm_duration,
                    "execution_time": tool_duration,
                    "role": "assistant",
                    "content": env_response.observation
                }
                decider["trajectory"].append(copy.deepcopy(decider_log))
                break
                
        return decider, return_messages

    def _background_task(self, messages_copy: List[Dict[str, Any]], env: Env, next_message: Dict[str, Any], turn_id: int) -> Dict[str, Any]:
        # Oracle Agent
        if background_agent_type == "guesser":
            guessers = []
            deciders = []
            final_return_messages = []
            guesser_log = {
                "role": "assistant",
                "content": next_message["content"],
                "turn_id": turn_id,
                "guessers": None,
                "deciders": None
            }
            conversation_history = ""
            for message in messages_copy:
                if message["role"] == "user":
                    conversation_history += f"{message['role']}: {message['content']}\n"
                elif message["role"] == "assistant" and message["content"] is not None:
                    conversation_history += f"{message['role']}: {message['content']}\n"
            conversation_history += "assistant: " + next_message["content"] + "\n"
            user_query_prompt = "Pretend you are a customer who is talking to a customer service Assistant. Given the conversation history, please continue the conversation. \nRules: - If the question is about authentication, then don't make up information, but just repeat your previous query.\n- If the assistant asking for confirmation, then you could confirm that.\n- If the assistant are providing some options, then you could decide which option you want to choose. \n\nConversation history:\n" + conversation_history + "\nUser:"
            
            if self.guesser_config is not None:
                type = self.guesser_config["type"]
                if type == "single":
                    guesser_models = [self.guesser_config["model"]]
                elif type == "multiple":
                    raise ValueError(f"Not supported for multiple guessers yet")
                else:
                    raise ValueError(f"Invalid guesser config type: {type}")

            for index, guesser_model in enumerate(guesser_models):
                guesser = self._guesser(index, user_query_prompt, guesser_model)
                guessers.append(guesser)

            
            if self.guesser_config is not None:
                if self.guesser_config["type"] == "single":
                    guesser = guessers[0]
                    decider, return_messages = self._decider(index, messages_copy, env, next_message, guesser, guesser_model)
                    deciders.append(decider)
                    final_return_messages = copy.deepcopy(return_messages)
                elif self.guesser_config["type"] == "multiple":
                    raise ValueError(f"Not supported for multiple guessers yet")
                else:
                    raise ValueError(f"Invalid guesser config type: {self.guesser_config['type']}")

            guesser_log["guessers"] = guessers
            guesser_log["deciders"] = deciders
            return guesser_log, final_return_messages
        elif background_agent_type == "preparer":
            return []
        else:
            raise ValueError(f"Invalid background agent type: {background_agent_type}")
        
    def _cosine_similarity(self, query1: str, query2: str) -> float:
        embeddings = self.embedding_model.encode([query1, query2])
        embedding1 = embeddings[0]
        embedding2 = embeddings[1]
        cosine_similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(cosine_similarity)

    def solve(
        self, env: Env, task_index: Optional[int] = None, max_num_steps: int = 30
    ) -> SolveResult:
        total_cost = 0.0
        env_reset_res = env.reset(task_index=task_index)
        obs = env_reset_res.observation
        info = env_reset_res.info.model_dump()
        reward = 0.0
        messages: List[Dict[str, Any]] = []
        self._future = None
        self.start_user = True
        durations = []
        count = 0
        counts = []
        traj_logs = [
            {"role": "system", "content": self.wiki, "turn_id": 0},
            {"role": "user", "content": obs, "turn_id": 1}
        ]
        guesser_logs = []
        baseline_trajectory = self.baseline_config[task_index]["traj"]
        traj_idx = 0
        while traj_idx < len(baseline_trajectory):
            curr = baseline_trajectory[traj_idx]
            messages.append(curr)
            if curr["role"] == "assistant" and "tool_calls" not in curr:
                messages_copy = copy.deepcopy(messages[:-1]) # remove the last assistant message, use from the curr["content"]
                # self._future = self._executor.submit(self._background_task, messages_copy, env, curr, len(messages))
                guesser_log, return_messages = self._background_task(messages_copy, env, curr, len(messages_copy))
                user_query_guess = guesser_log["guessers"][0]["output"]
                print("enter into background api calls")
                guesser_logs.append(guesser_log)
            if curr["role"] == "tool":
                count += 1
            if curr["role"] == "user":
                # return_messages = None
                # if self._future:
                #     try:
                #         guesser_log, return_messages = self._future.result()
                #         user_query_guess = guesser_log["guessers"][0]["output"]
                #         print("enter into background api calls")
                #         guesser_logs.append(guesser_log)
                #         # if return_messages is not None:
                #         #     for message in return_messages:
                #         #         messages.extend(message)
                #     except Exception as e:
                #         import traceback
                #         print(traceback.format_exc())
                #         print(f"Background task failed with error: {e}")
                #     finally:
                #         self._future = None

                durations.append(len(curr["content"].split(" ")) / 40 * 60)
                counts.append(count)
                count = 0
            traj_idx += 1
        for duration, count in zip(durations, counts):
            print(f"Duration: {duration}, Count: {count}")
        reward = self.baseline_config[task_index]["reward"]
        info = self.baseline_config[task_index]["info"]
        return SolveResult(
            reward=reward,
            info=info,
            messages=messages,
            total_cost=total_cost,
            user_typing_time=durations,
            num_tool_calls=counts,
            traj_logs=traj_logs,
            guesser_logs=guesser_logs
        )


def message_to_action(
    message: Dict[str, Any],
) -> Action:
    if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0 and message["tool_calls"][0]["function"] is not None:
        tool_call = message["tool_calls"][0]
        return Action(
            name=tool_call["function"]["name"],
            kwargs=json.loads(tool_call["function"]["arguments"]),
        )
    else:
        return Action(name=RESPOND_ACTION_NAME, kwargs={"content": message["content"]})


PREDICT_INSTRUCTION = f"""
# Instruction
Given the above conversation between you and user, assume user

At each step, your generation should have exactly the following format:
Thought:
<A single line of reasoning to process the context and inform the decision making. Do not include extra lines.>
Action:
{{"name": <The name of the action>, "arguments": <The arguments to the action in json format>}}

The Action will be parsed, so it must be valid JSON.

You should not use made-up or placeholder arguments.

For example, if the user says "I want to know the current weather of San Francisco", and there is such a tool available
{{
    "type": "function",
    "function": {{
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {{
            "type": "object",
            "properties": {{
                "location": {{
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                }},
                "format": {{
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                }},
            }},
            "required": ["location", "format"],
        }},
    }}
}}

Your response can be like this:
Thought:
Since the user asks for the weather of San Francisco in USA, the unit should be in fahrenheit. I can query get_current_weather to get the weather.
Action:
{{"name": "get_current_weather", "arguments": {{"location": "San Francisco, CA", "format": "fahrenheit"}}}}

And if the tool returns "70F", your response can be:
Thought:
I can answer the user now.
Action:
{{"name": {RESPOND_ACTION_NAME}, "arguments": {{"{RESPOND_ACTION_FIELD_NAME}": "The current weather of San Francisco is 70F."}}}}

Try to be helpful and always follow the policy.
"""