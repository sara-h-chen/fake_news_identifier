#######################################################
#        NATURAL LANGUAGE PROCESSING SUMMATIVE        #
#######################################################
# Runs on Python3. Easy setup requires that you have  #
# conda installed:                                    #
# https://conda.io/docs/user-guide/install/index.html #
#######################################################

import argparse
import numpy as np

import lstm_deep
import rnn_deep
import shallow


#######################################
#            MAIN METHOD              #
#######################################

if __name__ == '__main__':
    # TODO: Write the command line arguments to make switches between modes
    np.random.seed(1337)
    metrics = lstm_deep.Metrics()

    dir_to_data = 'data/news_ds.csv'

