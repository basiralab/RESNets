from .ACE import settings as st
from .ACE.encoder import Encoder
from .netNormExtended import netNormExtended
from .utilities import *

"""
 The implementation of RESNets framework.
 Details can be found in the original paper: will be filled when accepted
   ---------------------------------------------------------------------
     This file contains the implementation of three key steps of our RESNets framework:
     (1) Embedding of baseline population and test subject,
     (2) Building of CBT of the population and embedding of the CBT
     (3) Selection of top K similar subject at baseline and prediction of follow up data:
     
                        test_trajectory = RESNets(testing_subject, baseline_population, follow_up_data, K, n_r)
                        
                Inputs:
                          n_r: number of regions of interest(ROIs)
                          n_s: number of subjects
                          n_t: number of follow-up timepoints
                          n_f: size of feature vector = (n_r * (n_r - 1) / 2)
                          
                          test_subject = row vector with dimension (1 x n_f)
                          baseline_population: matrix of feature vectors (n_s x n_f) 
                          follow_up_data: Tensor of populations (n_t x n_s x n_f) 
                          K: number of similar subjects to be selected when predicting follow-up trajectory
                          n_r: number of regions of interest(ROIs)
                 Outputs:
                         test_trajectory: prediction of test subject at each timepoint (n_t x n_f)
                         
     To evaluate our framework we used Leave-One-Out cross validation strategy.
To test RESNets on random data, we defined the function 'simulateData' where the size of the dataset is chosen by the user.
 ---------------------------------------------------------------------
     Copyright 2020 Ahmet Serkan Göktaş, Istanbul Technical University.
     Please cite the above paper if you use this python code.
     All rights reserved.
"""


def RESNets(testing_subject, baseline_population, follow_up_data, K, n_r):

    model = 'arga_ae'
    settings = st.get_settings_new(model)
    enc = Encoder(settings)

    n_feature = testing_subject.shape[1]
    n_s = baseline_population.shape[0]

    embeddings = build_embeddings(enc, baseline_population, n_s, n_r)
    test_embedding = build_embeddings(enc, testing_subject, 1, n_r)
    baseline = np.zeros((1, n_s, n_r, n_r)) #reshaping baseline_population for feeding it into netNormExtended

    for sub in range(n_s):
        adjacency = to_2d(baseline_population[sub], n_r)
        baseline[0][sub] = adjacency

    cbt = netNormExtended(baseline, n_s, n_r, 1) #there is only one view for each subject
    # embed cbt
    cbt_embedded = build_embeddings(enc, cbt, n_subject=1, n_reg=n_r, isFlat= False)
    # subtract embedded cbt from each embedding)
    residuals = [preprocess(np.abs(cbt_embedded - np.reshape(embedding, (1, n_r)))) for embedding in embeddings]
    test_residual = preprocess(np.abs(cbt_embedded - np.reshape(test_embedding, (1, n_r))))
    similarities = [test_residual.dot(residual.T)[0][0] / (LA.norm(test_residual) * LA.norm(residual)) for residual in residuals]

    result_followup = np.zeros((follow_up_data.shape[0], 1, n_feature)) #A feature vector for each timepoint

    for timepoint in range(result_followup.shape[0]):
        top_k_similar_indexes = get_top_k(np.array(similarities), K)
        prediction = np.zeros((1, n_feature))
        for index in top_k_similar_indexes:
            prediction = prediction + follow_up_data[timepoint][index]

        prediction = prediction / K #Averaging
        result_followup[timepoint] = prediction

    return result_followup




