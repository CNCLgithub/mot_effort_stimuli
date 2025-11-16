# MOT Realtime and Retrospective Effort

This codebase contains all of the stimuli generating procedures using in: "A peak-end rule characterizes subjective attentional effort during moment-by-moment visual processing" 


## Installation

- Clone this repository
- Run `julia --projec=.` from inside the repo directory
- Run `using Pkg; Pkg.instantiate()`


## Organization

Each experiment has it's own script or folder, starting with `expX_*`. Output will be generated under `output` in the root directory of the repo. 
Each experiment script can be run by simply invoking `julia <SCRIPT>`.
For experiments 3 and 4, you can run `julia <FOLDER>/trials.jl`.

The other files either implement common functionality used in several experiments or provide debugging tools to help design new experiments. For example `render_dataset.jl` can be modified to render a dataset to images.

Julia manages package dependencies with `Project.toml` and `Manifest.toml` 

## Using generated output

In the `/output` folder, each run of an experiment script will generate the following set of files:

- `script.jl`: a backup of the script invoked to create the dataset
- `dataset.json`: The animation data for the set of trials (format described below)
- `examples.json`: Animation data for the example trials.
- `trial_list.json` : (Optional), The indices mapping animation data to trial IDs.
- `metrics.json` : The time course for each motion statistic used to implement time-varying difficulty. 

## Animation formats

The `dataset.json` file consists of a list of trials.
Each trial contains an entry of `positions`, which contains list of time points. 
Each time point is a list of 2D points (one for each object). 

The `examples.json` file contains the same format, but with fewer animations (usually only 3-4).
