This case is built based on the [tau-bench](https://github.com/sierra-research/tau-bench) repository.


## How to setup the environment
1. Go to the `speculative-action/e-commerce/tau-bench` directory
2. Create a virtual environment and install the dependencies
```bash
pip install -e .
```
3. Set up your OpenAI / Anthropic / Google / Mistral / AnyScale API keys as environment variables.
```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
MISTRAL_API_KEY=...
```


## How to generate the speculative action results
1. Go to the `speculative-action/e-commerce/tau-bench` directory
2. Run the `exp_static.sh` script to generate the speculative action results
```bash
./exp_static.sh
```
3. The results will be saved in the `results` directory.

   The sample results from gpt-5 family and gemini 2.5 flash family used in the paper are saved in the `results` directory.


## How to analyze the speculative action results
1. Go to the `speculative-action/e-commerce/tau-bench` directory
2. Run the `analysis_static_combine.ipynb` notebook to generate the figures in the paper.
3. The sample figures are saved in the `figures` directory.


