
from utils.util import pad_string
from itertools import imap

import numpy as np

PRETTY_METRICS = {
    'f1': 'f1'
    'rec': 'recall'
    'acc': 'accuracy'
    'prec': 'metthews'
    'spec': 'precision'
    'mcc': 'specificity'
}

def write_table(f, table_header, table_content, table_size = 100 ):

    f.write(pad_string("", table_size, "-", "-"))
    f.write(pad_string(table_header, table_size, " ", "|"))
    f.write(pad_string("", table_size, "-", "|"))

    for table_row in table_content:
        f.write( pad_string( table_row, table_size, " ", "|" ) )

    f.write(pad_string("", table_size, "-", "-"))
    f.write('\n')


def write_results_metrics(f, results, table_size):
        f.write(pad_string(f" f1             | {np.mean(results['f1'])}", table_size, " ", "|"))
        f.write(pad_string(f" recall         | {np.mean(results['rec'])}", table_size, " ", "|"))
        f.write(pad_string(f" accuracy       | {np.mean(results['acc'])}", table_size, " ", "|"))
        f.write(pad_string(f" metthews       | {np.mean(results['prec'])}", table_size, " ", "|"))
        f.write(pad_string(f" precision      | {np.mean(results['spec'])}", table_size, " ", "|"))
        f.write(pad_string(f" specificity    | {np.mean(results['mcc'])}", table_size, " ", "|"))

def write_cross_validation_resutls(f, title, results, table_size):
        f.write(pad_string("", table_size, "-", "-"))
        f.write(pad_string(title, table_size, " ", "|"))
        f.write(pad_string("", table_size, "-", "|"))

        write_results_metrics(f, results, table_size)

        f.write(pad_string("", table_size, "-", "-"))

def log_cross_validation_results(results, experiment_name, folds):

    table_title = f" {folds} K-fold summary for {experiment_name}"
    file_path = '../tmp/kfold_summary.txt'
    table_size = 100

    if len(table_title) > table_size: 
        table_size = len(summary_title) + 10

    division_point  = max(imap(len, PRETTY_METRICS))
    table_content = [ f'{ metric.ljust(division_point) } | { np.mean(results[key]) }' for key, metric in PRETTY_METRICS]

    with open( file_path, 'w' ) as f:
        write_table(f, table_header, table_content, table_size)

    return file_path

def log_class_legend(classes):

    table_title = "Class legend"
    file_path = '../tmp/class_legend'
    table_size = 100

    division_point = max(imap(len, classes))
    table_content = [f'{class_name.ljust(division_point)} | {index }'for class_name, index in classes.iteritems()]

    with open(file_path, 'a') as f:
        write_table(f, table_header, table_content, table_size)

def log_confusion_table(confusion_matrix):

    file_path = '../tmp/confusion_matrix.txts'
    table_size = 100

    with open(file_path, 'a') as f:

        f.write(pad_string("", table_size, "-", "-"))
        f.write(pad_string(" confusion table", table_size, " ", "|"))
        f.write(pad_string("", table_size, "-", "|"))

        for row in confusion_matrix:
            f.write(pad_string(f" {np.array_str(row)}", table_size, " ", "|"))
        
        f.write(pad_string("", table_size, "-", "-"))
        f.write('\n')


def log_to_results_comparison(results, experiment_name, folds):    

    file_path = 'all_results.txt'
    summary_title = f" {folds} K-fold summary for {experiment_name}"
    table_size = 100

    if len(summary_title) > table_size: 
        table_size = len(summary_title) + 10

    with open( file_path, 'a' ) as f:
        write_cross_validation_resutls(f, summary_title, results, table_size)

def write_class_missclassification_files(file_name, file_names, class_name, table_size):
    with open(file_name, 'a') as file:
        file.write(pad_string("", table_size, "-", "-"))
        file.write(pad_string(f" missclassified {class_name}", table_size, " ", "|"))
        file.write(pad_string("", table_size, "-", "|"))

        for miss_file_name in file_names:
            file.write(pad_string(f" {miss_file_name}", table_size, " ", "|"))
        
        file.write(pad_string("", table_size, "-", "-"))
        file.write('\n')

def write_kvasir_legend(file_name, table_size):

    with open(file_name, 'a') as file:

        file.write(pad_string("", table_size, "-", "-"))
        file.write(pad_string(" kvasir legend", table_size, " ", "|"))
        file.write(pad_string("", table_size, "-", "|"))
        
        file.write(pad_string(f"dyed-lifted-polyps     | 0", table_size, " ", "|"))
        file.write(pad_string(f"dyed-resection-margins | 1", table_size, " ", "|"))
        file.write(pad_string(f"esophagitis            | 2", table_size, " ", "|"))
        file.write(pad_string(f"normal-cecum           | 3", table_size, " ", "|"))
        file.write(pad_string(f"normal-pylorus         | 4", table_size, " ", "|"))
        file.write(pad_string(f"normal-z-line          | 5", table_size, " ", "|"))
        file.write(pad_string(f"polyps                 | 6", table_size, " ", "|"))
        file.write(pad_string(f"ulcerative-colitis     | 7", table_size, " ", "|"))

        file.write(pad_string("", table_size, "-", "-"))
        file.write('\n')

def write_confusion_table(file_name, confusion, table_size):

    with open(file_name, 'a') as file:

        file.write(pad_string("", table_size, "-", "-"))
        file.write(pad_string(" confusion table", table_size, " ", "|"))
        file.write(pad_string("", table_size, "-", "|"))

        for row in confusion:
            file.write(pad_string(f" {np.array_str(row)}", table_size, " ", "|"))
        
        file.write(pad_string("", table_size, "-", "-"))
        file.write('\n')

def write_model_summary(file_name, metrics, table_size):

    with open(file_name, 'a') as file:

        file.write(pad_string("", table_size, "-", "-"))
        file.write(pad_string(" model summary", table_size, " ", "|"))
        file.write(pad_string("", table_size, "-", "|"))
        
        file.write(pad_string(f" f1             | {np.mean(metrics['f1'])}", table_size, " ", "|"))
        file.write(pad_string(f" recall         | {np.mean(metrics['rec'])}", table_size, " ", "|"))
        file.write(pad_string(f" accuracy       | {np.mean(metrics['acc'])}", table_size, " ", "|"))
        file.write(pad_string(f" metthews       | {np.mean(metrics['prec'])}", table_size, " ", "|"))
        file.write(pad_string(f" precision      | {np.mean(metrics['spec'])}", table_size, " ", "|"))
        file.write(pad_string(f" specificity    | {np.mean(metrics['mcc'])}", table_size, " ", "|"))

        file.write(pad_string("", table_size, "-", "-"))
        file.write('\n')

def write_class_summary(file_name, metrics, class_name, class_index, table_size):

    with open(file_name, 'a') as file:

        file.write(pad_string("", table_size, "-", "-"))
        file.write(pad_string(f" {class_name} summary", table_size, " ", "|"))
        file.write(pad_string("", table_size, "-", "|"))
        
        file.write(pad_string(f" true positive  | {metrics['TP'][class_index]}", table_size, " ", "|"))
        file.write(pad_string(f" true negative  | {metrics['TN'][class_index]}", table_size, " ", "|"))
        file.write(pad_string(f" false positive | {metrics['FP'][class_index]}", table_size, " ", "|"))
        file.write(pad_string(f" false negative | {metrics['FN'][class_index]}", table_size, " ", "|"))

        file.write(pad_string("", table_size, "-", "|"))

        file.write(pad_string(f" f1             | {metrics['f1'][class_index]}", table_size, " ", "|"))
        file.write(pad_string(f" recall         | {metrics['rec'][class_index]}", table_size, " ", "|"))
        file.write(pad_string(f" accuracy       | {metrics['acc'][class_index]}", table_size, " ", "|"))
        file.write(pad_string(f" metthews       | {metrics['mcc'][class_index]}", table_size, " ", "|"))
        file.write(pad_string(f" precision      | {metrics['prec'][class_index]}", table_size, " ", "|"))
        file.write(pad_string(f" specificity    | {metrics['spec'][class_index]}", table_size, " ", "|"))

        file.write(pad_string("", table_size, "-", "-"))
        file.write('\n')