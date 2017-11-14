import numpy as np

#Extrapolates from a given seed sequence
def generate_from_seed(model, seed, sequence_length, include_seed_in_output=False, uncenter_data=False, data_variance=[], data_mean=[]):
    assert seed.shape[0] == 1
    #print "seed.shape[1] = %d" % seed.shape[1]
    #print "seed.shape[2] = %d" % seed.shape[2]
    seedSeq = seed.copy()
    output = []
    if include_seed_in_output:
        for i in xrange(seedSeq.shape[1]):
            output.append(seedSeq[0][i])
        for i in xrange(10):
            output.append(np.zeros(seedSeq.shape[2]))

    #The generation algorithm is simple:
    #Step 1 - Given A = [X_0, X_1, ... X_n], generate X_n + 1
    #Step 2 - Concatenate X_n + 1 onto A
    #Step 3 - Repeat MAX_SEQ_LEN times
    for it in xrange(sequence_length):
        seedSeqNew = model.predict(seedSeq) #Step 1. Generate X_n + 1
        #Step 2. Append it to the sequence
        if it == 0:
            # Zero pad beginning so that our final generated sequence is of length:
            # seed.shape[1] + sequence_length
            output.append(np.zeros(seedSeqNew.shape[-1]))
            for i in xrange(seedSeqNew.shape[1]):
                output.append(seedSeqNew[0][i].copy())
        else:
            output.append(seedSeqNew[0][seedSeqNew.shape[1]-1].copy()) 
        newSeq = seedSeqNew[0][seedSeqNew.shape[1]-1]
        newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
        # newSeq += np.random.randn(*newSeq.shape)*0.01
        seedSeq = np.concatenate((seedSeq, newSeq), axis=1)
        #print(seedSeq.shape)
        #seedSeq = seedSeq[:,1:,:]
        #print(seedSeq.shape)

    if uncenter_data:
        assert len(data_variance) > 0
        assert len(data_mean) > 0
        # Finally, post-process the generated sequence so that we have valid frequencies
        # We're essentially just undo-ing the data centering process
        for i in xrange(len(output)):
            output[i] *= data_variance
            output[i] += data_mean
            
    return output
    

    #print(seedSeq.shape)
    #for it in xrange(sequence_length):
        #seedSeqNew = model.predict(seedSeq) #Step 1. Generate X_n + 1
        ##Step 2. Append it to the sequence
        #for i in xrange(seedSeqNew.shape[1]):
            #output.append(seedSeqNew[-1][i].copy())
        ##newSeq = seedSeqNew[0][seedSeqNew.shape[1]-1]
        ##newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
        #newSeq = seedSeqNew[-1][np.newaxis,:]
        ##seedSeq = np.concatenate((seedSeq, newSeq), axis=0)
        #seedSeq = newSeq
        #print(seedSeq.shape)

    ##Finally, post-process the generated sequence so that we have valid frequencies
    ##We're essentially just undo-ing the data centering process
    #for i in xrange(len(output)):
        #output[i] *= data_variance
        #output[i] += data_mean
    #return output
