import sys
sys.path.append('..')

from argparse import ArgumentParser
from lge_cnn.nn.models import BaselineNet
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import os, shutil, logging, warnings, copy, pickle, torch

"""
    Training and evaluation
"""


def train(hparams):
    results = []
    test_results = []
    for num in range(hparams.num_models):

        entry = {}

        if hparams.num_models > 1:
            print("Training model {} of {}.".format(num+1, hparams.num_models))

        model = BaselineNet(hparams)
        print("Training baseline CNN model.")


        print("Trainable parameters: {}".format(model.count_trainable_parameters()))
        model.cuda()

        # loggers
        tb_logger = TensorBoardLogger(save_dir=hparams.logdir, name=hparams.name)
        
        # checkpointing
        checkpoint = pl.callbacks.model_checkpoint.ModelCheckpoint()

        # early stopping and trainer
        if not hparams.no_early:
            early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=0.00,
                                                patience=hparams.max_epochs // 4, verbose=False, mode='min')
            trainer = pl.Trainer.from_argparse_args(hparams, callbacks=[early_stop_callback], weights_summary='full', checkpoint_callback=checkpoint)
        else:
            trainer = pl.Trainer.from_argparse_args(hparams, early_stop_callback=False, weight_summary='full', checkpoint_callback=checkpoint)

        # training
        trainer.logger = tb_logger
        trainer.fit(model)
        
        # loading best checkpoint
        best_model = torch.load(checkpoint.best_model_path)
        model.load_state_dict(best_model['state_dict'])

        # evaluation
        model.cuda()
        test_mse = model.mse(global_average=True, cuda=True)
        test_results.append(test_mse)
        print("Test MSE: {:.4e}".format(test_mse))

        # save entry
        entry['test_mse'] = test_mse
        entry['state_dict'] = model.state_dict()
        entry['hparams'] = copy.copy(hparams)

        # close all datasets
        model.close()

        # add to results list
        results.append(entry)

        print("**********************")

    if hparams.num_models > 1:
        print("Ensemble training complete.")
        for i, mse in enumerate(test_results):
            print("Model {} of {}: test MSE {:.3e}".format(i+1, hparams.num_models, mse))

    return results


"""
    Command line parser
"""

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = BaselineNet.add_model_specific_args(parser)

    parser.add_argument('--name', type=str)
    parser.add_argument('--logdir', type=str)
    parser.add_argument('--clear_logs', action='store_true', default=False)
    parser.add_argument('--no_early', action='store_true', default=False)
    parser.add_argument('--num_models', type=int, default=1)

    

    hparams = parser.parse_args()

    if hparams.clear_logs:
        log_path = os.path.join(hparams.logdir, hparams.name)
        if os.path.exists(log_path):
            shutil.rmtree(log_path)

    results = train(hparams)

    filename = hparams.name + "_results.pickle"
    
    with open(filename, 'wb') as file:
        pickle.dump(results, file)