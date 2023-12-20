------------------------- ANT COLONY OPTIMIZATION PROGRAM README -----------------------

Welcome to this implementation of the Ant Colony Optimization algorithm!
Please refer to this README for instructions on how to run the code and further details on files included in this package


----------------------------------- FILE STRUCTURE -------------------------------------

The source code can be found in the 'src' directory, with the python script used to run the algorithmm being named 'ACO_implementation.py'. 
This folder also contains the input data for the algorithm in the files 'brazil58.xml' and 'burma14'
CSV files containing data gathered as part of various experiments can be found in the 'res' directory. This data was used to generate figures and results contained in the report.


------------------------------------- RUNNING THE CODE ---------------------------------

The implemented ACO program was successfully tested using Python 3.10.2, although there is no guarantee that it will run properly on different versions.

It requires two Python libraries in order to run:
    - xml.etree.ElementTree
    - numpy
xml.etree.ElementTree is installed by default with Python, whereas numpy can be installed using pip ('pip install numpy') via the command line

Before running the program, please also ensure that the xml files containing the data ('brazil58.xml' and 'burma14.xml') are in the same folder as the program 'ACO_implementation.py'

Once dependencies are installed, the program 'ACO_implementation.py' can then be run from the command line (from its directory) using:
    $ python ACO_implementation.py

From there, the program prompts the user to choose the type of ACO to run, as well as the input data file to use and multiple parameters associated with the ACO variant.

The different variants of ACO available to the user are the following:
    - Standard ACO (s)
    - Max-Min Ant System ACO (m)
    - Elitist Ant System ACO (e)
    - Rank-Based Ant System (r)

Standard ACO will prompt the user for the following parameters:
    - The name of the input file (either 'brazil58.xml' or 'burma14') 
    - The number of ants and iterations to use
    - The values for the alpha, beta, rho and Q parameters

Standard ACO can also be run with or without a local search heuristic, which include the following:
    - Tabu Search (t)
    - Hill Climbing (h)
    - 2-opt (2)
    - No local search heuristic (none)

NOTE: be advised that running the Standard ACO with Tabu Search can be very time-consuming (especially for the larger 'brazil58' data). 
It will complete, but most likely very slowly. It is therefore recommended to use a low number of tabu iterations when picking this option.

Max-Min Ant System, Elitist, and Rank-Based ACO approaches will prompt the user for the same parameters as Standard ACO, but do not support adding a local search heuristic.

Max-Min Ant System also requires:
    - The maximum pheromone value limit
    - The minimum pheromone value limit
