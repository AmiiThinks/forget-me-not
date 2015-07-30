Setting up an experiment has three parts.

First, you need the environment/data parser/agent/learner/model/whatever files.
Ideally these have both a batch update and a one-step update mechanism.


Then you need the Structure for each element you might want to include. 
This is what allows them to be constructedon the fly by the Experiment class.
This also determines what will be saved to the logs.

Then you need the specific Experiment instance for your settings. It takes care 
of logging and getting the parts to work together (mediated by the Structure
wrappers)

The jobs.py file allows you to automate the submission of Experiments with 
different parameter sets. 