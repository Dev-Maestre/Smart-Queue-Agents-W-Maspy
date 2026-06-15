#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
LIGHT_MODE=0
LIGHT_PLUS_MODE=0

if [[ "${1:-}" == "--light" ]]; then
  LIGHT_MODE=1
  shift
elif [[ "${1:-}" == "--light-plus" ]]; then
  LIGHT_PLUS_MODE=1
  shift
fi

cd "$ROOT_DIR"

# Opcional: use MASPY local em vez do pacote pip (export MASPY_ROOT=/caminho/para/MASPY)
if [[ -n "${MASPY_ROOT:-}" ]]; then
  export PYTHONPATH="$MASPY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON_BIN:-}" ]] || ! "$PYTHON_BIN" -c "import maspy" 2>/dev/null; then
  echo "Python com maspy nao encontrado."
  echo "Crie o venv: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  echo "Ou defina PYTHON_BIN apontando para seu interpretador."
  exit 1
fi

RUN_DIR="$ROOT_DIR/runs_output/combined_$(date +%Y%m%d_%H%M%S)"
export RUN_DIR

if [[ "$LIGHT_MODE" -eq 1 ]]; then
  echo "Modo LIGHT ativado: reduzindo carga de treino para gerar dados/graficos mais rapido."
  export RUN_LIGHT=1
  export LIGHT_SEEDS="123"
  export LIGHT_TRAIN_EPISODES_LONG=120
  export LIGHT_REPLAY_EPISODES=1
  export LIGHT_ENABLE_TENSORBOARD=0
  export LIGHT_CONV_WINDOW=10
  export LIGHT_CONV_PATIENCE=3
  export LIGHT_ADAPT_EVAL_STEPS=20
  export LIGHT_ADAPT_RECOVERY_WINDOW=4
  export LIGHT_ABLATION_EPISODES=80
  export LIGHT_ABLATION_GAMMAS="0.9,0.96,0.98"
elif [[ "$LIGHT_PLUS_MODE" -eq 1 ]]; then
  echo "Modo LIGHT+ ativado: mais dados com custo moderado para TCC."
  export RUN_LIGHT=1
  export LIGHT_SEEDS="123,124,125,126,127"
  export LIGHT_TRAIN_EPISODES_LONG=300
  export LIGHT_HORIZON=25
  export LIGHT_REPLAY_EPISODES=2
  export LIGHT_ENABLE_TENSORBOARD=0
  export LIGHT_RENDER_PYGAME=0
  export LIGHT_CONV_WINDOW=14
  export LIGHT_CONV_PATIENCE=4
  export LIGHT_ADAPT_EVAL_STEPS=24
  export LIGHT_ADAPT_RECOVERY_WINDOW=4
  export LIGHT_ABLATION_EPISODES=120
  export LIGHT_ABLATION_GAMMAS="0.9,0.96,0.98"
fi

echo "1) Rodando experimentos (MASPY + baseline)..."
"$PYTHON_BIN" run_experiments.py

echo "2) Gerando graficos (inclui novos graficos de RL)..."
"$PYTHON_BIN" plot_results.py

echo "Pronto."
echo ""
echo "Ultimo run:"
ls -1dt "$ROOT_DIR"/runs_output/* | head -n 1
LATEST_RUN="$(ls -1dt "$ROOT_DIR"/runs_output/* | head -n 1)"
if [[ -n "${LATEST_RUN:-}" ]]; then
  echo ""
  echo "Replay sugerido:"
  echo "$PYTHON_BIN $ROOT_DIR/ex-smart_intersection_pygame.py --replay-dir \"$LATEST_RUN\" --replay-step 0.5"
fi
