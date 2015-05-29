'''
BLINC Adaptive Prosthetics Toolkit
- Bionic Limbs for Improved Natural Control, blinclab.ca
anna.koop@gmail.com

A toolkit for running machine learning experiments on prosthetic limb data

This module file handles submitting experiments to Wesgrid
'''
from experiment import *
from local import base_dir, test_dir

# I think it is easier to edit this file than to use command-line args for the jobs
exp_params = {'base_dir': [base_dir],
              'platform': ['calgary'],
              'protocol': ['book1'], #, 'pic', 'obj1', 'progc'],
              'model': ['FastCTW', 'PTW:FastCTW', 'FMN:FastCTW', 'CTW_KT', 'KT', 'CTW:PTW'],
              #'depth': [32, 48, 64]
              }


class JobSet(Structure):
    #TODO: maybe this doesn't have to be a full-blown parsable structure
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
            outfile = os.path.join(ps['base_dir'], ps['platform'], ps['protocol']+"_", ps['model'])
            name = "{}-{}".format(ps['model'], ps['protocol'])
            if self.safe_mode:
                print("Checking if log file for {} exists...".format(name))
                if os.path.exists(outfile):
                    print("already there, turn off safe mode to overwrite.")
                    continue
                
            # here is where we should loop over depth if CTW in model            
            argstring = "compress -m {model} {infile} {outfile}".format(model=ps['model'], 
                                                                        infile=infile, 
                                                                        outfile=outfile)              
        
            self.submit_job(name, argstring)
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
        
               
    def submit_job(self, filename, argstring):
        """
        Submit specific experiment to the pbs experiment queue
        Save the submission file with the jobid
        
        If debug is on, print job command rather than submitting it.
        If run_now is on, run the experiment directly.
        """
        sh = self.pbs_template(filename, argstring)
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
        

    def pbs_template(self, filename, argstring):
        lines = ["#!/bin/sh",
                 "",
                 "#PBS -S /bin/sh",
                 "#PBS -j oe",
                 "#PBS -r n",
                 "#PBS -o {0}/{1}.$PBS_JOBID.log".format(self.log_dir, 
                                                         filename),
                 "#PBS -l nodes=1:ppn=1," #no comma here on purpose
                  "walltime={}:{}:00,mem=1gb".format(self.num_hours, self.num_minutes),
                 "",
                 "cd $PBS_O_WORKDIR",
                 "echo \"Current working directory is `pwd`\"",
                 "echo \"Starting run at: `date`\"",
                 "python z.py {}".format(argstring), 
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
