'''
BLINC Adaptive Prosthetics Toolkit
- Bionic Limbs for Improved Natural Control, blinclab.ca
anna.koop@gmail.com

A toolkit for running machine learning experiments on prosthetic limb data

This module file handles submitting experiments to Wesgrid

Usage: jobs.py --log_name LOG_DIR [OPTIONS]

Edit this file to set the experiment params. See class definition for job-submission parameters.
'''
from experiment import *
from local import base_dir, test_dir
import subprocess

# I think it is easier to edit this file than to use command-line args for the jobs
exp_params = {'base_dir': [base_dir],
              'platform': ['calgary'],
              'protocol': ['bib'],#, 'book1', 'book2', 'geo', 'news', 'obj1', 'obj2', 
                           #'paper1', 'paper2', 'paper3', 'paper4', 'paper5', 'pic', 'progp', 'progl', 'progc'],
              'model': ['FastCTW', 'PTW_FastCTW', 'FMN_FastCTW'],
              'depth': [16, 32, 48, 64]
              }

def pull_stats_from_file(filename):
    with open('logs/{}'.format(filename), 'r') as f:
        stats = f.readlines()[-7:]
        if 'Starting' in stats[-1]:
            return {'flaked': True}
        elif 'killed' in stats[-1]:
            return {'killed': stats[-1].split(':')[-1].strip()}
        data = {}
        for s in stats:
            try:
                parts = s.split(':')
                data[parts[0].strip()] = parts[1].split()[0]
            except IndexError:
                pass
        return data

def pull_params_from_name(filename):
    data = {'filename': filename}
    (data['protocol'], data['model'], depth, *parts) = filename.split('-')
    data['depth'] = depth.split('.')[0]
    return data
    
def get_all_logs():
    data = {}
    for f in os.listdir('logs'):
        if '.log' in f:
            name = f.split('.')[0]
            params = pull_params_from_name(f)
            params.update(pull_stats_from_file(f))
            data[name] = Series(params)
    return DataFrame(data)
    
def graph_results(data, protocol):
    import matplotlib.pyplot as plt
    subset = data[data.protocol == protocol][['model', 'depth', 'Size']].convert_objects(convert_numeric=True)
    models = sorted(subset.model.unique())
    depths = sorted(subset.depth.unique())
    for m in models:
        subset[subset.model==m].plot(x='depth', y='Size', label=m, xticks=depths)
    plt.legend()
    plt.title('Compressed size of {} as a function of depth for each model type'.format(protocol))
    

class JobSet(Structure):
    """
    JobSet takes the dictionary of parameter sets defined above 
    and parses them into a sequence of experiment.py calls.
    
    Usage: jobs.py --log_name DIRNAME [OPTIONS]
    
    ---log_name      The directory to store logs of the submission and output
    ---debug         Do not call qsub, just print out command
    ---run_now       Run the experiment directly (still submitting or not according to other parameters)
    ---safe_mode     Do not overwrite files
    ---num_minutes   Number of minutes to request
    ---num_hours
    """
    _fields = [Dir('log_name', required=True, keyword=False, default='logs'),
               Boolean('debug', default=True, transient=True),
               Boolean('run_now', default=False, transient=True),
               Boolean('safe_mode', default=True, transient=True),
               Integer('num_minutes', default=20, transient=True),
               Integer('num_hours', default=0, transient=True),
               ]
    
    def run(self, **kwargs):
        # figure out which keys have multiple options
        interesting_keys = [k for k in kwargs if len(kwargs[k]) > 1]
        param_sets = unique_dict_sets(kwargs)
        
        self.num_checked = 0
        self.num_submitted = 0
        for ps in param_sets:
            self.num_checked += 1
            infile = os.path.join(ps['base_dir'], ps['platform'], ps['protocol'])
            
            # check if we need to cat the file
            if not os.path.exists(infile):
                paths = [os.path.join(ps['base_dir'], ps['platform'], f) for f in ps['protocol'].split('_')]
                extra = "\n".join(["echo \"Checking the existence of the file {}\"".format(ps['protocol']),
                         "if [ ! -e {} ]".format(infile),
                         "  then `cat {} > {}`".format(' '.join(paths), infile),
                         "  echo \"...created\"",
                         "fi"
                         ])
            else:
                extra = ""
            
            
            #outfile = os.path.join(ps['base_dir'], ps['platform'], ps['protocol']+"_", ps['model'])
            #os.makedirs(os.path.dirname(outfile), exist_ok=True)
            more = True
            i = 0
            name = '-'.join([ps['protocol'], ps['model'], str(ps['depth'])])
            #if self.safe_mode:
            #   print("Checking if log file for {} exists...".format(name))
            #if os.path.exists(outfile):
            #    print("already there, turn off safe mode to overwrite.")
            #    continue
                
            argstring = "compress -m {model} -d {depth} {infile} {outfile}".format(model=ps['model'], 
                                                                                          infile=infile,
                                                                                depth=ps['depth'],
                                                                                outfile='/dev/null') 
                
            self.submit_job(name, argstring, extra)
            self.num_submitted += 1
        print("Submitted {} jobs out of {}".format(self.num_submitted,
                                                   self.num_checked))
    
    def get_jobfilename(self, **kwargs):
        filename = ''
        bits = {}
        for k in list(kwargs):
            if '_string' in k:
                filename = "_".join([filename, kwargs.pop(k)])
        return "_".join([filename, clean_string(kwargs)])
        
               
    def submit_job(self, filename, argstring, extra=None):
        """
        Submit specific experiment to the pbs experiment queue
        Save the submission file with the jobid
        
        If debug is on, print job command rather than submitting it.
        If run_now is on, run the experiment directly.
        """
        sh = self.pbs_template(filename, argstring, extra)
        tmpfile = os.path.join(self.log_dir, filename)
        print("Scheduling {} ... ".format(filename), end=""); sys.stdout.flush()
    
        with open(tmpfile, 'w') as shfile:
            shfile.write(sh)
        cmd = "qsub {}".format(tmpfile)
        if self.debug:
            print("\n"+cmd)
            pname = 'DEBUG'
        else:
            try:
                P = subprocess.check_output(cmd, shell=True)
                pname = P.decode('ascii').strip()
                print("Submitted {}".format(pname))
            except Exception as e:
                print("Problem calling {}\n{}".format(cmd, e))
                pname = "FAILED"
        if self.run_now:
            print("Running experiment")
            try:
                import z
                z.__main__(argstring.split())
            except Exception as e:
                print("Problem with experiment: {}".format(e))
        jobscript = "{}.{}.sh".format(filename, pname)
        print("Saving {} file".format(jobscript))
        script_path = os.path.join(self.log_dir, jobscript)
        os.rename(tmpfile, script_path)
        return script_path
        

    def pbs_template(self, filename, argstring, extra=""):
        lines = ["#!/bin/sh",
                 "",
                 "#PBS -S /bin/sh",
                 "#PBS -j oe",
                 "#PBS -r n",
                 "#PBS -o {0}/{1}.$PBS_JOBID.log".format(self.log_dir, 
                                                         filename),
                 "#PBS -l nodes=1:ppn=1," #no comma here on purpose
                  "walltime={}:{}:00,mem=4000mb".format(self.num_hours, self.num_minutes),
                 "",
                 extra,
                 "cd $PBS_O_WORKDIR",
                 "echo \"Current working directory is `pwd`\"",
                 "echo \"Starting run at: `date`\"",
                 "alias pypy=/home/akoop/pypy3-2.4-linux_x86_64-portable/bin/pypy3",
                 "pypy z.py {}".format(argstring), 
                 "echo \"Completed run with exit code $? at: `date`\""]
        
        return "\n".join(lines)

    
            
if __name__ == "__main__":
    if len(sys.argv) > 1:
        print("Creating...")
        setup = JobSet.from_args(sys.argv[1:])
        print("Running...")
        setup.run(**exp_params)
        print("Done")
    else:
        print("Supply at least one command-line argument to run for real")
        setup = JobSet.from_args(['--run_now', '--debug', '--safe_mode', '--log_name', 'logs'])
        #setup.get_parser().print_help()
        print("Running in debug mode.")
        setup.run(**exp_params)
