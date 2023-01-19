from glob import glob

import tensorboard_reducer as tbr

# LOG_dir = "TB_logs/test_batchtranslator/version_20"

# LOG_dir = "runs/Jan18_12-18-58_BMEDYER-GPU2_ERM"
# LOG_dir = "runs/Jan18_12-19-43_BMEDYER-GPU2_MixStyle" 
# LOG_dir = "runs/Jan18_12-36-38_BMEDYER-GPU2_CLIP_ERM"
# LOG_dir = "runs/Jan18_12-36-57_BMEDYER-GPU2_CLIP_MixStyle"

# LOG_dir = "runs/Jan18_20-19-22_BMEDYER-GPU2_MixStyle" # shuffle crossdomain 
LOG_dir = "runs/Jan18_20-31-25_BMEDYER-GPU2_CLIP_MixStyle" # shuffle crossdomain

TEST_DOMAIN = 0

input_event_dirs = sorted(glob(LOG_dir))
# where to write reduced TB events, each reduce operation will be in a separate subdirectory

overwrite = False
reduce_ops = ("mean", "min", "max", "median", "std", "var")

events_dict = tbr.load_tb_events(input_event_dirs)

# number of recorded tags. e.g. would be 3 if you recorded loss, MAE and R^2
n_scalars = len(events_dict)
n_steps, n_events = list(events_dict.values())[0].shape

# print(n_scalars, n_steps, n_events) # 8 18 1
# print(events_dict['Acc/env0_in']['value'].tolist())

def find_max_on_train_domain(test_domain=0):
    domains = [0, 1, 2, 3]
    domains.remove(test_domain)

    final_results = None
    for domain_id in domains:
        results = events_dict['Acc/env{}_out'.format(domain_id)]['value'].tolist()

        if final_results is None:
            final_results = results
        else:
            final_results = [i + results[p_i] for p_i, i in enumerate(final_results)]

    final_results = [i/len(domains) for i in final_results]
    # print([i/len(domains) for i in final_results])

    max_value = max(final_results)
    max_index = final_results.index(max_value)
    print(max_index)

    test_results = events_dict['Acc/env{}_in'.format(test_domain)]['value'].tolist()

    print(final_results, '\n *** \n', test_results, max_index, len(final_results))

    print(test_results[max_index])


find_max_on_train_domain(test_domain=TEST_DOMAIN)
