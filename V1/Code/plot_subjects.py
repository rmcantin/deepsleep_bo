import numpy as np
import os
import argparse
import math
import json
import matplotlib.pyplot as plt

from collections import Counter
from scipy.signal import butter, hilbert, lfilter
from auditory_stimulation_trigger import AuditoryStimulationEvaluation
from frequencies_detector import FrequenciesDetector
from slow_waves_decoder import SlowWavesDecoder

import bayesopt
from bayesoptmodule import BayesOptContinuous

class BayesOptStimuli(BayesOptContinuous):

    def __init__(self, file_name, 
                 opt_mask = np.array([0,1,0,1,1,1]), 
                 alphas = np.array([1,0,1,1]),
                 eu = False,
                 dbd = False):

        self.fig = None
        self.axs = None

    def plot_metrics(self,metrics,color='b.',a=1.0,label=None):

        cmaes = metrics[:,0]
        csds = metrics[:,1]
        pups = metrics[:,2]
        pnoups = metrics[:,3]

        if self.fig is None:
            self.fig, self.axs = plt.subplots(1,3,tight_layout=True)
            self.axs[0].set_xlabel('CMAE')
            self.axs[0].set_ylabel('-PUP')
            self.axs[1].set_xlabel('CMAE')
            self.axs[1].set_ylabel('PnotUP')
            self.axs[2].set_xlabel('-PUP')
            self.axs[2].set_ylabel('PnotUP')
        self.axs[0].plot(cmaes,-pups,color,alpha=a)
        self.axs[1].plot(cmaes,pnoups,color,alpha=a)
        self.axs[2].plot(-pups,pnoups,color,alpha=a,label=label)

    def save_figure(self,fig_name):

        self.fig.legend(loc='upper right')
        self.fig.set_size_inches(12.0,5.0)
        self.fig.savefig(fig_name,dpi=100)


    def evaluateSample(self, Xin):

        return 0

def main(args):

    # ---------------------------------- Obtain arguments ------------------------------------

    data_dir = os.path.abspath(args.directory)
    results_name = "results_all"

    subject_ids = (args.subjects)
    if len(subject_ids)>7:
        print("Please select 7 subjects or less")
        return 0

    alphas = np.zeros(4)
    results_path = os.path.join(os.getcwd(),results_name)
    print("RESULTS WILL BE SAVED AS:",results_path)

    # ---------------------------------- Problem definition ----------------------------------

    opt_mask = np.array([0,1,0,1,1,1])
    n = np.sum(opt_mask)

    results = {}

    score = -1
    query = np.zeros(6)

    colors = ['r+','b+','g+','m+','y+','c+','k+']

    bo = BayesOptStimuli("")
    i = 0
    for subject_id in subject_ids:
        if i >= 7:
            print("More than 7 files match the provided Subject IDs. Stopping.")
            break
        s_str = "_"+subject_id+"_"
        for file in os.listdir(data_dir):
            if (".json" in file) and (("_"+subject_id+"_") in file):
                filename = os.path.join(data_dir,file)
                with open(filename,'r') as fp:
                    data = json.load(fp)
                    try:
                        metrics_dp = np.array(data["Metric Datapoints"])
                    except:
                        continue
                    
                    c = colors[i]
                        
                    metrics_dp = metrics_dp[~np.all(metrics_dp==0,axis=1)]
                    bo.plot_metrics(metrics_dp[:17],c,a=0.8) # 16 initial points + default metrics
                    c = c.replace("+",".")
                    bo.plot_metrics(metrics_dp[17:],c,label=subject_id)
                i+=1
                continue

    fig_name = results_path + '.png'
    bo.save_figure(fig_name)
    print("FIGURE SAVED AT:",fig_name)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("-d","--directory",default="../Models",help="Directorio con el fichero a optimizar.")
    ap.add_argument("-s","--subjects",nargs='+',default=["ST3","ST5","ST6","ST10"],help="Sujetos a mostrar.")

    args = ap.parse_args()
        
    main(args)