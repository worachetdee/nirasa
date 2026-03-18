.PHONY: install lint format test smoke-test train-tokenizer data-pipeline pretrain eval serve clean

install:
	pip install -e ".[dev]"

lint:
	ruff check nirasa/ tests/ scripts/

format:
	ruff format nirasa/ tests/ scripts/

test:
	pytest tests/ -v

smoke-test:
	python scripts/smoke_test.py --device cpu

train-tokenizer:
	python -m nirasa.tokenizer.train_tokenizer \
		--input_dir data/raw \
		--model_prefix data/tokenizer/nirasa_th_sp \
		--vocab_size 32000

data-pipeline:
	python scripts/run_data_pipeline.py

pretrain:
	python -m nirasa.training.train \
		--data_bin data/processed/th_wiki_qwen.bin \
		--data_idx data/processed/th_wiki_qwen.idx \
		--output_dir checkpoints/nirasa-7b-th \
		--max_steps 5000

eval:
	python -m nirasa.eval.run_eval \
		--model_path checkpoints/nirasa-7b-th \
		--benchmarks "thaiqa,xnli,wisesight,perplexity"

serve:
	python -m nirasa.serving.api_server \
		--model_path checkpoints/nirasa-7b-th \
		--host 0.0.0.0 \
		--port 8000

clean:
	rm -rf build/ dist/ *.egg-info .eggs/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
