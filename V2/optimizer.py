import numpy as np
import os
import argparse
import math
import json
import matplotlib.pyplot as plt
import time

import evaluator

import bayesopt
from bayesoptmodule import BayesOptContinuous

class BayesOptStimuli(BayesOptContinuous):

    def __init__(self, directory, file_name, 
                 opt_mask = np.array([1,1,1]), 
                 alphas = np.array([0.1,0,1,1]),
                 lb = np.array([-80-  0.1, 0.8]), 
                 dv = np.array([-60,  0.1, 1]),
                 ub = np.array([-20,  1.0, 10]),
                 dbd = False,
                 sample_bounds = np.array([0,0])):

        # Problem dimension equals the number of parameters to optimize
        # This is deduced from the mask
        self.n = int(np.sum(opt_mask))

        # If mask is all ceros, optimization is not possible
        assert self.n > 0, "Mask is all zeros, no parameters will be optimized"

        super().__init__(self.n)

        # Prepare bounds for each parameter
        
        # Lower bounds for all parameters
        # Default values for all parameters
        # Upper bounds for all parameters

        print("Optimizing",self.n,"parameters")
        print("Mask:",opt_mask)
        print("Lower bounds:",lb)
        print("Default values:",dv)
        print("Upper bounds:",ub)

        self.lower_bound = lb[opt_mask>0]
        self.default_values = dv
        self.upper_bound = ub[opt_mask>0]
        self.opt_mask = opt_mask
        self.file_name = file_name
        self.directory = directory
        self.alphas = alphas

        # Prepare arrays for metric recolection
        self.metrics = np.zeros(6)
        self.default_metrics = np.zeros(6)
        self.error_cases = []

        # Declare optimization mode:
        #
        # "dbd" -> "Divided By Default"- Each metric is divided by the value obtained 
        #                                from the default parameters.
        self.dbd = dbd
        
        # Declare if there is a need to take a subset of the data from the file
        self.sample_bounds = sample_bounds

        # Data points for saving and plotting
        self.metric_dp = np.zeros((1,4))
        self.query_dp = np.zeros((1,3))

    def compute_default_metrics(self):

        self.evaluateSample(self.default_values[self.opt_mask>0])
        self.default_metrics = self.metrics

        return self.default_metrics

    def compute_alphas_dbd(self):
    
        self.alphas = np.array([self.alphas[0]/self.default_metrics[2],
                                self.alphas[1]/self.default_metrics[1],
                                self.alphas[2]/self.default_metrics[3],
                                self.alphas[3]/self.default_metrics[4]])

        return self.alphas

    def set_alphas(self, alphas):

        self.alphas = alphas

    def return_full_query(self,Xin):

        Q = self.default_values
        Q[self.opt_mask>0] = Xin

        return Q        

    def plot_metrics(self,fig_name):

        self.metric_dp = self.metric_dp[1:]

        init_it = self.parameters['n_init_samples']

        cmaes_init = self.metric_dp[:init_it,0]
        pups_init = self.metric_dp[:init_it,2]
        pnoups_init = self.metric_dp[:init_it,3]

        cmaes = self.metric_dp[init_it:,0]
        pups = self.metric_dp[init_it:,2]
        pnoups = self.metric_dp[init_it:,3]

        best_cmae = self.metric_dp[-1,0]
        best_pup = self.metric_dp[-1,2]
        best_pnotup = self.metric_dp[-1,3]

        fig, axs = plt.subplots(1,3,tight_layout=True)
        axs[0].plot(cmaes,-pups,'b.')
        axs[0].plot(cmaes_init,-pups_init,'b+')
        axs[0].plot(best_cmae,-best_pup,'r.')
        axs[0].set_xlabel('CMAE')
        axs[0].set_ylabel('-PUP')
        axs[1].plot(cmaes,pnoups,'b.')
        axs[1].plot(cmaes_init,pnoups_init,'b+')
        axs[1].plot(best_cmae,best_pnotup,'r.')
        axs[1].set_xlabel('CMAE')
        axs[1].set_ylabel('PnotUP')
        axs[2].plot(-pups,pnoups,'b.')
        axs[2].plot(-pups_init,pnoups_init,'b+')
        axs[2].plot(-best_pup,best_pnotup,'r.')
        axs[2].set_xlabel('-PUP')
        axs[2].set_ylabel('PnotUP')

        #plt.show()
        fig.set_size_inches(12.0,4.0)
        fig.savefig(fig_name,dpi=100)


    def evaluateSample(self, Xin):
        Q = self.return_full_query(Xin)
    
        if self.sample_bounds[0] != 0 or self.sample_bounds[1] != 0:
            print("start", self.sample_bounds[0], "end", self.sample_bounds[1])
            self.metrics = evaluator.process(dir=self.directory, file=self.file_name, threshold_max=Xin[0], k_pll=Xin[1], f_nco=Xin[2],
                                            start_sample=self.sample_bounds[0], end_sample=self.sample_bounds[1])
        else:
            self.metrics = evaluator.process(dir=self.directory, file=self.file_name, threshold_max=Xin[0], k_pll=Xin[1], f_nco=Xin[2])
        
        cmae = self.metrics[2]
        csd = self.metrics[1]
        pas_up_phase = self.metrics[3]
        pas_not_up_phase = self.metrics[4]
        
        self.metric_dp = np.vstack((self.metric_dp,[cmae,csd,pas_up_phase,pas_not_up_phase]))
        self.query_dp = np.vstack((self.query_dp,Q))
        
        try:
            if np.isnan(self.metrics).any():
                raise TypeError("Encountered NaNs during evaluation")

            r = 2.0
            r = self.alphas[0]*cmae + self.alphas[1]*csd - self.alphas[2]*pas_up_phase + self.alphas[3]*pas_not_up_phase
            return r

        except:

            print("AN ERROR OCCURRED. RETURNING CONSTANT LIAR.")
            self.error_cases = self.error_cases + [Q.tolist()]
            self.metrics = np.zeros(6)
            
            r = 2.0
            return r
            
def main(args):

    # ---------------------------------- Obtain I/O arguments ------------------------------------
    model_dir = args.directory
    file = args.file
        
    model_name = "bopt_"+file
    results_name = "results_"+file

    opt_mode = "a"
    alphas = np.zeros(4)

    dbd = args.divbydef
    p = args.plot
    
    sample_bounds = np.array([args.start_sample, args.end_sample])
    
    if sample_bounds[0] > sample_bounds[1]:
        print("ERROR. Please input valid beginning and end time stamps (more than one sample and start < end) or simply remove them to process the whole file")
        return
        
    if sample_bounds[0] != sample_bounds[1]:
        model_name = model_name + "_" + str(sample_bounds[0]) + "-" + str(sample_bounds[1])
        results_name = results_name + "_" + str(sample_bounds[0]) + "-" + str(sample_bounds[1])
    else:
        model_name = model_name + "_noBounds"
        results_name = results_name + "_noBounds"
        
    # Introduce a timestamp to not override different results
    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    model_name = f"{model_name}_{timestamp}"
    results_name = f"{results_name}_{timestamp}"

    alpha_str = "_a"
    if len(args.alphas)!=4:
        print("ERROR. Please input 4 values for alphas.")
        return

    for i,a in enumerate(args.alphas):
        alphas[i] = float(a)
        alpha_str = alpha_str + "_" + a.replace(".","")
    model_name = model_name + alpha_str
    results_name = results_name + alpha_str

    if dbd:
        model_name = model_name + "_dbd"
        results_name = results_name + "_dbd"
    
    model_path = os.path.join(model_dir,model_name)
    print("MODEL WILL BE SAVED AS:",model_path)
    results_path = os.path.join(model_dir,results_name)
    print("RESULTS WILL BE SAVED AS:",results_path)

    # ---------------------------------- Problem definition ----------------------------------

    lb = np.zeros(3)
    dv = np.zeros(3)
    ub = np.zeros(3)
    opt_mask = np.zeros(3)

    if (len(args.lowbounds) != 3) or (len(args.defval) != 3) or (len(args.upbounds) != 3) or (len(args.optmask) !=3 ):
        print("Error. Please enter 3 values for bounds, mask and default values.")
        return

    for i in range(3):
        lb[i] = float(args.lowbounds[i])
        dv[i] = float(args.defval[i])
        ub[i] = float(args.upbounds[i])
        opt_mask[i] = int(args.optmask[i])

    n = np.sum(opt_mask)

    results = {}

    score = -1
    query = np.zeros(3)

    params = {}
    params['n_init_samples'] = int(args.init)
    params['n_iterations'] = int(args.iter)
    params['n_iter_relearn'] = 1
    params['l_type'] = 'mcmc'
    params['init_method'] = 2 # Sobol
    params['load_save_flag'] = 2 # Save, don't load
    params['save_filename'] = model_path + '.txt'
    params['verbose_level'] = 5 # Log
    params['log_filename'] = model_path + "_log.txt"
    params['force-jump'] = int(args.force_jump) # 2 or 3 for default
    params['noise'] = args.noise # 1e-6 by default in bayesopt, 1e-4 in optimizer
    
    # To fixate the seed and better evaluate the results
    params['random_seed'] = 123456
    np.random.seed(123456)

    bo = BayesOptStimuli(model_dir,file,opt_mask,alphas,lb,dv,ub,dbd,sample_bounds)
    bo.parameters = params
    default_metrics = bo.compute_default_metrics()
    if dbd:
        alphas = bo.compute_alphas_dbd()

    score, x_out, error = bo.optimize()
    query = bo.return_full_query(x_out)
    
    metrics = np.zeros(6)
    error_cases = bo.error_cases

    if not error:

        _ = bo.evaluateSample(x_out)
        best_metrics = bo.metrics
        metrics = bo.metric_dp[1:]
        queries = bo.query_dp[1:]

    np.savez(results_path+".npz",score=score,query=query,best_metrics=best_metrics,default_metrics=default_metrics,metrics_dp=metrics,queries_dp=queries)
    
    results= {"Query":query.tolist(),"Best Metrics":best_metrics,"Default Metrics":default_metrics,"Metric Datapoints":metrics.tolist(),"Query Datapoints":queries.tolist(),"Errors":error_cases}

    with open(results_path+".json",'w') as fp:
        json.dump(results,fp)

    if(p):
        fig_name = results_path + '.png'
        print("SAVING FIGURE AT:",fig_name)
        bo.plot_metrics(fig_name)

if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("-f","--file",required=True,help="Identificador del fichero a optimizar. (e.g. '_AR1_0')")
    ap.add_argument("-d","--directory",default="../Data",help="Directorio con el fichero a optimizar.")
    ap.add_argument("-a","--alphas",default=['0.1','0.0','1.0','1.0'],nargs='+',help="Coeficientes para la suma pesada. Se requieren 4: CMAE, CSD, PUP y PnotUP")
    ap.add_argument("-lb","--lowbounds",default=['-80','0.1','0.8'],nargs='+',help="Límites inferiores del espacio de exploración. Se requieren 3: th_max, k_pll y f_nco (ya no se modifican th_min, down_up_th ni target phase).")
    ap.add_argument("-dv","--defval",default=['-40','0.1','1.5'],nargs='+',help="Valores por defecto de los parámetros. Se requieren 3, igual que lb.")
    ap.add_argument("-ub","--upbounds",default=['-20','1.0','10'],nargs='+',help="Límites superiores del espacio de exploración. Se requieren 3, igual que lb.")
    ap.add_argument("-op","--optmask",default=['1','1','1'],nargs='+',help="Máscara de optimización. Se requieren 3, igual que lb. Los parámetros con 0 usarán siempre dv, los parámetros con más de 0 serán optimizados.")
    ap.add_argument("-start","--start_sample",default=0,type=int,help="Instante de la grabación a partir del cual extraer datos. Tiene que ser un instante válido y menor que -end (e.g. 955167).")
    ap.add_argument("-end","--end_sample",default=0,type=int,help="Instante de la grabación a partir del cual dejar de extraer datos. Tiene que ser un instante válido y mayor que -start (e.g. 1460208).")
    ap.add_argument("-in","--init",default=16,help="Iteraciones para construcción del modelo inicial (exploración, no optimización)")
    ap.add_argument("-it","--iter",default=32,help="Iteraciones del proceso de optimización.")
    ap.add_argument("-fj","--force_jump",default=2,help="Número de iteraciones sin mejora antes de dar un salto aleatorio.")
    ap.add_argument("-ns","--noise",default=0.0001,help="Ruido aleatorio aplicado a las muestras.")
    ap.add_argument("-dbd","--divbydef",action="store_true",help="Dividir métricas por valor por defecto. (Se recomienda usarlo)")
    ap.add_argument("-p","--plot",action="store_true",help="Dibujar métricas en los puntos explorados al final.")

    args = ap.parse_args()
        
    main(args)