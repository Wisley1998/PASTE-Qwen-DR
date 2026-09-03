python run.py --agent-strategy tool-calling-static --env retail \
 --model gpt-4o --model-provider openai --user-model gpt-4o \
 --user-model-provider openai --user-strategy llm --max-concurrency 10 \
 --start-index 0 --end-index 115 \
 --guesser-config guess_configs/gpt5_low.json \
 --baseline-config ./historical_trajectories/gpt-4o-retail.json

python run.py --agent-strategy tool-calling-static --env retail \
 --model gpt-4o --model-provider openai --user-model gpt-4o \
 --user-model-provider openai --user-strategy llm --max-concurrency 10 \
 --start-index 0 --end-index 115 \
 --guesser-config guess_configs/gpt5_medium.json \
 --baseline-config ./historical_trajectories/gpt-4o-retail.json

python run.py --agent-strategy tool-calling-static --env retail \
 --model gpt-4o --model-provider openai --user-model gpt-4o \
 --user-model-provider openai --user-strategy llm --max-concurrency 10 \
 --start-index 0 --end-index 115 \
 --guesser-config guess_configs/gpt5_high.json \
 --baseline-config ./historical_trajectories/gpt-4o-retail.json


python run.py --agent-strategy tool-calling-static --env retail \
 --model gpt-4o --model-provider openai --user-model gpt-4o \
 --user-model-provider openai --user-strategy llm --max-concurrency 10 \
 --start-index 0 --end-index 115 \
 --guesser-config guess_configs/gemini2.5_flash_high.json \
 --baseline-config ./historical_trajectories/gpt-4o-retail.json

python run.py --agent-strategy tool-calling-static --env retail \
 --model gpt-4o --model-provider openai --user-model gpt-4o \
 --user-model-provider openai --user-strategy llm --max-concurrency 1 \
 --start-index 0 --end-index 115 \
 --guesser-config guess_configs/gemini2.5_flash_low.json \
 --baseline-config ./historical_trajectories/gpt-4o-retail.json

 python run.py --agent-strategy tool-calling-static --env retail \
 --model gpt-4o --model-provider openai --user-model gpt-4o \
 --user-model-provider openai --user-strategy llm --max-concurrency 5 \
 --start-index 0 --end-index 115 \
 --guesser-config guess_configs/gemini2.5_flash_medium.json \
 --baseline-config ./historical_trajectories/gpt-4o-retail.json

