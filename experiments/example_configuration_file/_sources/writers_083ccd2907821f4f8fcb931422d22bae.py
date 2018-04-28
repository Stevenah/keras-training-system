
from utils import pad_string

import numpy as np

def write_kfold_summary(file_name, metrics, experiment_name, folds, table_size, method='w'):
    
    with open(file_name, method) as file:

        file.write(pad_string("", table_size, "-", "-"))
        file.write(pad_string(f" {folds} K fold summary for {experiment_name}", table_size, " ", "|"))
        file.write(pad_string("", table_size, "-", "|"))

        file.write(pad_string(f" f1             | {np.mean(metrics['f1'])}", table_size, " ", "|"))
        file.write(pad_string(f" recall         | {np.mean(metrics['rec'])}", table_size, " ", "|"))
        file.write(pad_string(f" accuracy       | {np.mean(metrics['acc'])}", table_size, " ", "|"))
        file.write(pad_string(f" metthews       | {np.mean(metrics['prec'])}", table_size, " ", "|"))
        file.write(pad_string(f" precision      | {np.mean(metrics['spec'])}", table_size, " ", "|"))
        file.write(pad_string(f" specificity    | {np.mean(metrics['mcc'])}", table_size, " ", "|"))

        file.write(pad_string("", table_size, "-", "-"))

def write_class_missclassification_files(file_name, file_names, class_name, table_size):
    with open(file_name, 'w') as file:
        file.write(pad_string("", table_size, "-", "-"))
        file.write(pad_string(f" missclassified {class_name}", table_size, " ", "|"))
        file.write(pad_string("", table_size, "-", "|"))

        for miss_file_name in file_names:
            file.write(pad_string(f" {miss_file_name}", table_size, " ", "|"))
        
        file.write(pad_string("", table_size, "-", "-"))
        file.write('\n')

def write_kvasir_legend(file_name, table_size):

    with open(file_name, 'w') as file:

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

    with open(file_name, 'w') as file:

        file.write(pad_string("", table_size, "-", "-"))
        file.write(pad_string(" confusion table", table_size, " ", "|"))
        file.write(pad_string("", table_size, "-", "|"))

        for row in confusion:
            file.write(pad_string(f" {np.array_str(row)}", table_size, " ", "|"))
        
        file.write(pad_string("", table_size, "-", "-"))
        file.write('\n')

def write_model_summary(file_name, metrics, table_size):

    with open(file_name, 'w') as file:

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

    with open(file_name, 'w') as file:

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