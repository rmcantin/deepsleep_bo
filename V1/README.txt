This document explains the main parts of this directory and how to
execute the optimization code and interpret its output.

--- STRUCTURE ---

This folder is organized in three subfolders:

- Code: 	Contains the developed optimization code. Detailed description below.

- Models:	Output folder for logs and final models.

- Data:	Store here the DeepSleep dataset for the program to find it.
		As provided, this folder only contains a "dummy.txt" file to prevent compression programs
		to delete it for being empty.

--- EXECUTION ---

The main part of the code relies on two files:

- optimizer.py:	Main optimization script. Use -h option for parameter and option info.
			The in-script help and this text should suffice for standard use.

			In order to replicate results, use:

			python optimizer.py -f <subject> -dbd -p

			For <subject>, use the subject identifier (e.g. 'ST4').
			Following the dataset name standard, introducing 'ST4' will indicate the
			script to look for a file whose name contains '_ST4_'.

			The script allows for parameter modification:

			Options -lb and -ub each select the upper and lower bounds for exploration.
			Both these two options, -dv (default values) and -op (optimization mask) expect
			six values (one per parameter, explained in the in-script help).
			If left unspecified, they use predefined values. Be wary of the 'f_nco' parameter
			default value: 10. This is intended for exploration but may not be desirable.

			Lastly, the -a options allows to weight each metric (CMAE, CSD, PUP and PnotUP).
			If specified, it expects four values. Else, default values are [0.1,0.0,1.0,1.0].
			Since execution is single-thread, it is possible to execute the program several times
			with different configurations in parallel.

			At the end of the execution, the code returns 4 files at the 'Models' folder.
			Two start with 'results', which contain info about best results obtained, and
			two start with 'bopt' which contain logs of the optmization process.
			Using the -p option provides an extra 'results' file, which is a png format plot.

			Files results_X.json and results_X.npz contain:

			- Best obtained config (labeled "Query" in json and "query" in npz)
			- Best obtained metrics ("Best Metrics" and "best_metrics")
			- Metrics for default values ("Default Metrics" and "default_metrics")
			- Metrics for all samples ("Metric Datapoints" and "metric_dp")
			- Configuration for all samples ("Query Datapoints" and "query_dp")

			Configurations are size-6 arrays: 

			th_min, th_max, down_up_th, k_pll, f_nco and target_phase (in the same order as in the options)

			Metrics are size-7 arrays:

			cm, csd, cmae, pas_up, pas_not_up, detecciones and estimulaciones

			The .png files contains three plots for 2D relations of CMAE, PASUP and PASnotUP.
			

- plot_subjects.py:	Given a list of subject ids, read their json results and compile a single image with 
				combined plots for all subjects. Each color represents a different subject, 
				and can combine up to seven subjects at once.

