import gzip
import pickle

def prepare_data(file_path):

    with gzip.open(file_path, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

def get_labels(order='label_index'):
    
    # check label ordering
    if order == 'label_index':
        # reutnr kvasir labels in label_index format
        return {
            'dyed-lifted-polyps': 0, 'dyed-resection-margins': 1, 'esophagitis': 2,
            'normal-cecum': 3, 'normal-pylorus': 4, 'normal-z-line': 5, 'polyps': 6,
            'ulcerative-colitis': 7,
        }

    # return kvasir labels in index_label format
    return {
        0: 'dyed-lifted-polyps', 1: 'dyed-resection-margins', 2: 'esophagitis',
        3: 'normal-cecum', 4: 'normal-pylorus', 5: 'normal-z-line', 6: 'polyps',
        7: 'ulcerative-colitis',
    }