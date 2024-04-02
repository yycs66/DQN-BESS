<<<<<<< HEAD
'''
Utility functions to support the make_offer.py code
'''

import numpy as np
import os
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floatingN):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_json(filename, filedir='./'):
    '''Find the status of your resources and of the next market configuration'''
    with open(os.path.join(filedir,f'{filename}.json'), 'r') as f:
        json_dict = json.load(f)
    return json_dict

def split_mktid(mktid):
    """Splits the market_id string into the market type and the time string (YYYYmmddHHMM)"""

    split_idx = [i for i, char in enumerate(mktid) if char == '2'][0]
    mkt_type = mktid[:split_idx]
    start_time = mktid[split_idx:]
    return mkt_type, start_time

def compute_offers(resources, times, demand, renewables):
    """Takes the status and forecast and makes an offer dictionary"""
    # We will loop through keys to fill our offer out, making updates as needed
    status = resources['status']
    klist = list(status.keys())
    my_soc = status[klist[0]]['soc']
    offer_keys = ['cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp', 'block_ch_mc', 'block_dc_mc',
                  'block_soc_mc', 'block_ch_mq', 'block_dc_mq', 'block_soc_mq', 'soc_end',
                  'bid_soc', 'init_en', 'init_status', 'ramp_up', 'ramp_dn', 'socmax', 'socmin',
                  'soc_begin', 'eff_ch', 'eff_dc', 'chmax', 'dcmax']
    offer_vals = [3, 3, 0, 0, 0, 0, [-20, -10, 0, 10], 125, 125, [250, 50, 208, 100], 128,
                  False, 0, 0, 9999, 9999, 608, 128, my_soc, 0.9, 1, 125, 125]
    use_time = [True, True, True, True, True, True, True, True, True, True, False, False, False,
                False, False, False, False, False, False, False, False, True, True]
    offer_out = {}
    for rid in status.keys():
        resource_offer = {}
        for i, key in enumerate(offer_keys):
            if use_time[i]:
                time_dict = {}
                for t in times:
                    time_dict[t] = offer_vals[i]
            else:
                time_dict = offer_vals[i]
            resource_offer[key] = time_dict
        offer_out[rid] = resource_offer
    return offer_out

def save_offer(offer, time_step):
    """Saves the offer in json format to the correct resource directory"""
    # Write the data dictionary to a JSON file
    if time_step != 4:
        json_file = f'offer_{time_step}.json'
        with open(json_file, "w") as f:
            json.dump(offer, f, cls=NpEncoder, indent=4)
=======
'''
Utility functions to support the make_offer.py code
'''

import numpy as np
import os
import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floatingN):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def load_json(filename, filedir='./'):
    '''Find the status of your resources and of the next market configuration'''
    with open(os.path.join(filedir,f'{filename}.json'), 'r') as f:
        json_dict = json.load(f)
    return json_dict

def split_mktid(mktid):
    """Splits the market_id string into the market type and the time string (YYYYmmddHHMM)"""

    split_idx = [i for i, char in enumerate(mktid) if char == '2'][0]
    mkt_type = mktid[:split_idx]
    start_time = mktid[split_idx:]
    return mkt_type, start_time

def compute_offers(resources, times, demand, renewables):
    """Takes the status and forecast and makes an offer dictionary"""
    # We will loop through keys to fill our offer out, making updates as needed
    status = resources['status']
    klist = list(status.keys())
    my_soc = status[klist[0]]['soc']
    offer_keys = ['cost_rgu', 'cost_rgd', 'cost_spr', 'cost_nsp', 'block_ch_mc', 'block_dc_mc',
                  'block_soc_mc', 'block_ch_mq', 'block_dc_mq', 'block_soc_mq', 'soc_end',
                  'bid_soc', 'init_en', 'init_status', 'ramp_up', 'ramp_dn', 'socmax', 'socmin',
                  'soc_begin', 'eff_ch', 'eff_dc', 'chmax', 'dcmax']
    offer_vals = [3, 3, 0, 0, 0, 0, [-20, -10, 0, 10], 125, 125, [250, 50, 208, 100], 128,
                  False, 0, 0, 9999, 9999, 608, 128, my_soc, 0.9, 1, 125, 125]
    use_time = [True, True, True, True, True, True, True, True, True, True, False, False, False,
                False, False, False, False, False, False, False, False, True, True]
    offer_out = {}
    for rid in status.keys():
        resource_offer = {}
        for i, key in enumerate(offer_keys):
            if use_time[i]:
                time_dict = {}
                for t in times:
                    time_dict[t] = offer_vals[i]
            else:
                time_dict = offer_vals[i]
            resource_offer[key] = time_dict
        offer_out[rid] = resource_offer
    return offer_out

def save_offer(offer, time_step):
    """Saves the offer in json format to the correct resource directory"""
    # Write the data dictionary to a JSON file
    if time_step != 4:
        json_file = f'offer_{time_step}.json'
        with open(json_file, "w") as f:
            json.dump(offer, f, cls=NpEncoder, indent=4)
>>>>>>> a572ce81a4fba6669fedb6dde2510e2f8269e39c
