import data

feature_set = {'merchant_name': {'enabled': True, 'output_size': 10},
                    'user_reference': {'enabled': True, 'output_size': 10},
                    'eighth_of_day': {'enabled': True, 'output_size': 8},
                    'day_of_week': {'enabled': True, 'output_size': 7},
                    'amount': {'enabled': True, 'output_size': 1},
                    'sys_category': {'enabled': True, 'output_size': 10}}

corpus = data.Features('merchant_seqs_by_tx', 5, feature_set, False, True)
