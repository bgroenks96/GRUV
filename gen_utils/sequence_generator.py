import numpy as np

# Extrapolates from a given seed sequence drawn from a training data set.
def generate_from_example_seed(model, seed, max_seq_len, include_raw_seed=False, include_model_seed=False, uncenter_data=False, data_variance=[], data_mean=[]):
    assert seed.shape[0] == 1
    #print "seed.shape[1] = %d" % seed.shape[1]
    #print "seed.shape[2] = %d" % seed.shape[2]
    seedSeq = seed.copy()
    output = []
    seq_len = 0
    if include_raw_seed:
        for i in xrange(seedSeq.shape[1]):
            output.append(seedSeq[0][i])
            seq_len += 1
        for i in xrange(10):
            output.append(np.zeros(seedSeq.shape[2]))
            seq_len += 1

    #The generation algorithm is simple:
    #Step 1 - Given A = [X_0, X_1, ... X_n], generate X_n + 1
    #Step 2 - Concatenate X_n + 1 onto A
    #Step 3 - Repeat MAX_SEQ_LEN times
    gen_itr = 0
    while seq_len + gen_itr < max_seq_len:
        seedSeqNew = model.predict(seedSeq) #Step 1. Generate X_n + 1
        #Step 2. Append it to the sequence
        if gen_itr == 0 and include_model_seed:
            for i in xrange(seedSeqNew.shape[1]):
                output.append(seedSeqNew[0][i].copy())
                gen_itr += 1
        else:
            output.append(seedSeqNew[0][seedSeqNew.shape[1]-1].copy()) 
            gen_itr += 1
        newSeq = seedSeqNew[0][seedSeqNew.shape[1]-1]
        newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
        # newSeq += np.random.randn(*newSeq.shape)*0.01
        seedSeq = np.concatenate((seedSeq, newSeq), axis=1)
        seedSeq = seedSeq[:,1:,:]

    if uncenter_data:
        assert len(data_variance) > 0
        assert len(data_mean) > 0
        # Finally, post-process the generated sequence so that we have valid frequencies
        # We're essentially just undo-ing the data centering process
        for i in xrange(len(output)):
            output[i] *= data_variance
            output[i] += data_mean
            
    return output

# Generation algorithm for generative model that feeds 'seed' to the model repeatedly until the output sequence is
# of length 'max_seq_len'.
def generate_from_random_seed(model, seeds, max_seq_len, batch_size=None, uncenter_data=False, target_variance=[], target_mean=[]):
    print(seeds.shape)
    out_dims = 0
    outputs = np.zeros((seeds.shape[0], 0, out_dims))
    seq_len = 0
    # Start from our random seed and generate new sequences until the output length is equal to max_seq_len.
    # This algorithm assumes that the given model is stateful; we feed the same seed in each time.
    while seq_len < max_seq_len:
        next_seq = model.predict(seeds, batch_size=batch_size)
        if out_dims == 0:
            # Auto detect model output dimensions
            out_dims = next_seq.shape[2]
            outputs = outputs.reshape((outputs.shape[0],0,out_dims))
        outputs = np.concatenate((outputs, next_seq), axis=1)
        seq_len += next_seq.shape[1]
    
    if seq_len > max_seq_len:
        # Clamp output length to max_seq_len
        outputs = outputs[:,0:max_seq_len,:]
    if uncenter_data:
        assert len(target_variance) > 0
        assert len(target_mean) > 0
        for i in xrange(len(outputs)):
            outputs[i] *= target_variance
            outputs[i] += target_mean
            
    return outputs
        
        
        

