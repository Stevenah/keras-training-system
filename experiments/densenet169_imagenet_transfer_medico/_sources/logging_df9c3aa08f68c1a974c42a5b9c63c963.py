
from utils.util import pad_string

import numpy as np

import os

BASIC_METRICS = {
    'TP': 'true positive',
    'TN': 'true negative',
    'FP': 'false positive',
    'FN': 'false negative'
}

ADVANCED_METRICS = {
    'f1': 'f1',
    'rec': 'recall',
    'acc': 'accuracy',
    'prec': 'metthews',
    'spec': 'precision',
    'mcc': 'specificity'
}

TABLE_ROW_DIVIDER = '-'

TEMP_DIRECTORY = '../tmp'

def write_table( file_path, table_header, table_content ):
    
    file_action = 'a' if os.path.exists(file_path) else 'w'
    table_size = 100

    if len(table_header) > table_size:
        table_size = len(table_header) + 10

    with open( file_path, file_action ) as f:

        f.write(pad_string('', table_size, '-', '-'))
        f.write(pad_string(f' {table_header}', table_size, ' ', '|'))
        f.write(pad_string('', table_size, '-', '|'))

        for table_row in table_content:
            if table_row == TABLE_ROW_DIVIDER: f.write( pad_string( '', table_size, '-', '|' ) )
            else: f.write( pad_string( table_row, table_size, ' ', '|' ) )

        f.write(pad_string('', table_size, '-', '-'))
        f.write('\n')        


def log_cross_validation_results( file_name, results, experiment_name, folds ):

    file_path = os.path.join( TEMP_DIRECTORY, file_name )
    
    table_header = f'{ folds } K-fold summary for { experiment_name }'
    table_content = [ ]

    division_point  = max( map( len, ADVANCED_METRICS ) )

    for key, metric in ADVANCED_METRICS.items():
        table_content.append( f'{ metric.ljust( division_point ) } | { np.mean( results[key] ) }' )

    write_table( file_path, table_header, table_content )

def log_class_legend( file_name, class_names ):

    file_path = os.path.join( TEMP_DIRECTORY, file_name )

    table_header = 'Class legend'
    table_content = [ ]

    division_point = max( map( len, class_names ) )

    for index, class_name in enumerate( class_names ):
        table_content.append( f'{ class_name.ljust( division_point ) } | { index }' )

    write_table( file_path, table_header, table_content )

def log_confusion_table( file_name, confusion_matrix ):

    file_path = os.path.join( TEMP_DIRECTORY, file_name )

    table_header = 'Confusion table'

    table_content = [ f' { row }' for row in np.array_str(confusion_matrix, max_line_width=1000000).split('\n') ]

    write_table( file_path, table_header, table_content )

def log_to_results_comparison( results, experiment_name, folds ):    

    file_path = 'all_results.txt'

    table_header = f' { folds } K-fold summary for { experiment_name }'
    table_content = [ ]

    division_point  = max( map( len, ADVANCED_METRICS.values() ) )
    
    for key, metric in ADVANCED_METRICS.items():
        table_content.append( f'{ metric.ljust( division_point ) } | { np.mean( results[key] ) }' )

    write_table(file_path, table_header, table_content)

def log_class_results( results, class_name, class_index ):

    file_path = os.path.join(TEMP_DIRECTORY, 'class_results.txt')

    table_header = f'{class_name} summary'
    table_content = [ ]

    division_point  = max(max(map(len, ADVANCED_METRICS.values())), max(map(len, BASIC_METRICS.values())))
    print(division_point)

    for key, metric in BASIC_METRICS.items():
        table_content.append( f'{ metric.ljust(division_point) } | { results[key] }' )

    table_content.append(TABLE_ROW_DIVIDER)

    for key, metric in ADVANCED_METRICS.items():
        table_content.append( f'{ metric.ljust(division_point) } | { results[key] }' )
        
    write_table(file_path, table_header, table_content)

def log_misclassifications( misclassifications, class_name):

    file_path = os.path.join(TEMP_DIRECTORY, 'class_misclassifications.txt')

    table_header = f'Misclassified {class_name}'
    table_content = [ f' { miss_file_name }' for miss_file_name in misclassifications ]

    write_table(file_path, table_header, table_content)

def log_model_results( file_name, results, model_name ):

    file_path = os.path.join(TEMP_DIRECTORY, file_name)
    
    table_header = f'Model summary for split {model_name}'
    table_content = []

    division_point  = max(map(len, ADVANCED_METRICS.values()))

    for key, metric in ADVANCED_METRICS.items():
        table_content.append( f'{ metric.ljust( division_point ) } | { np.mean( np.mean(results[key]) ) }' )

    write_table(file_path, table_header, table_content)
