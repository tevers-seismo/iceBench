# iceBench

**iceBench** is a Python library for detecting, picking, associating and locating icequakes with pretrained deep-learning phase pickers from [SeisBench](https://github.com/seisbench/seisbench).

It was developed for the MSc thesis *Icequake Detection and Location with Deep-Learning Phase Pickers at Grenzgletscher (Switzerland)* (T. Evers, LMU Munich & TU Munich, 2026).

## What it does

Icequakes contain most of their energy at frequencies above 50 Hz. The pretrained pickers in SeisBench operate at 100 Hz, so their internal resampling removes exactly this content. iceBench therefore offers the possibility for a **rescaling** of the input data: The rescaling idea follows Shi et al. (2024).

The library covers:

- **Rescaling** of waveforms for any pretrained SeisBench picker, with correct back-transformation of the pick times (per trace, including blinding).
- **Benchmarking** of model configurations (model × weight set × overlap × stacking × rescaling) and thresholds against a reference catalogue.
- **Continuous picking** of long data periods in chunks, with output as pick Parquet.
- **Association** of the picks into events with [PyOcto](https://github.com/yetinam/pyocto).
- **NonLinLoc** control files used for the probabilistic hypocentre inversion. NonLinLoc is not a Python package and has to be installed separately:  [NonLinLoc](https://github.com/ut-beg-texnet/NonLinLoc)
- **Coordinate conversion** of NonLinLoc locations from the local Cartesian frame to latitude/longitude.


## Usage
 
Runs are configured with a YAML file (see `examples/run_config_example.yaml`). Example scripts for each step of the workflow are provided in `examples/`:
 
| Step | Example script |
|---|---|
| Benchmark against a reference catalogue | `examples/run_benchmark_example.py` |
| Evaluation of the benchmark results | `examples/run_benchmark_evaluation_example1.py`, `examples/run_benchmark_evaluation_example2.py` |
| Continuous picking of a data period | `examples/run_continuos_example.py` |
| Association of analyst picks with PyOcto | `examples/associate_analystPicks.py` |
| Association of automatic picks for the reference events (benchmark) | `examples/associate_autoPicks.py` |
| Association of automatic picks from continuous run | `examples/associate_autoPicks_continous.py` |
| PyOcto parameters used in the thesis | `examples/associator_config.py` |
| Hypocentre inversion with NonLinLoc | `examples/NLLoc/` |
| Obspyck control files for manual picking | `examples/obspyck/` |
 

## Repository structure

```
icebench/            library code
  config.py          reading and checking the YAML configuration
  run.py             benchmark and continuous runs
  sources.py         waveform download per event or per time chunk
  preprocess.py      preprocessing (detrending)
  models.py          loading SeisBench models and weight sets
  transform.py       rescaling and back-transformation of picks
  predict.py         model application and pick extraction
  store.py           Parquet / QuakeML output
  associate.py       PyOcto association
  evaluate.py        pick matching, residual and detection statistics
  plot.py            plotting helpers
examples/
  NLLoc/             NonLinLoc control files and file generation, coordinate transformation
  obspyck/           Obspyck control files 
  ...
```

## Settings used in the thesis

The three final picking setups of the thesis (all with rescaling):

| Model | Weight set | P threshold | S threshold | Overlap | Stacking |
|---|---|---|---|---|---|
| PhaseNet | VCSEIS (`volpick`) | 0.276 | 0.224 | 0.9 | avg | transformed |
| PhaseNet | original | 0.412 | 0.182 | 0.9 | avg| transformed |
| EQTransformer | VCSEIS (`volpick`) | 0.312 | 0.158 | 0.5 | max | transformed |

PhaseNet with the VCSEIS weights is the recommended setup. 

