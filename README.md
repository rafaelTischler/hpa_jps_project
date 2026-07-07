# HPA-JPS Project

Este projeto implementa e compara A*, JPS, HPA* e HPA-JPS em grids 2D.

## Instalação

```bash
pip install -r requirements.txt
```

## Mapas do movingai

Baixe arquivos `.map` e `.scen` em `data/maps/` a partir do benchmark movingai.com.

Exemplo:

```bash
mkdir -p data/maps
# coloque os arquivos .map e .scen aqui
```

## Executar testes

```bash
pytest tests/
```

## Experimentos

Para rodar um experimento simples:

```bash
python main.py experiment --sizes 256 --densities low medium high --map-count 1
```

Os resultados são gravados em `results/raw/experiment_results.csv`.

## Gerar gráficos

```bash
python main.py plot --input results/raw/experiment_results.csv
```

Os gráficos serão salvos em `results/plots/`.

## Renderizar mapa

```bash
python main.py render --map data/maps/arena.map --algorithm hpajps --start 0,0 --goal 50,50
```
