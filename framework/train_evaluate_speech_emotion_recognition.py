import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from data_providers import IEMOCAP
from arg_extractor import get_args
from experiment_builder import ExperimentBuilder
from model_architectures import *
from torch.nn.utils.rnn import pack_sequence
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

def pad_collate(batch):
  (xx, yy) = zip(*batch)

  xx_trans = [x.transpose(0,1) for x in xx]
  #print(xx_trans.shape)
  xx_pad = pad_sequence(xx_trans, batch_first=True)

  y_list = torch.Tensor([y for y in yy])

  return xx_pad, y_list

if __name__ == '__main__':
    args = get_args()  # get arguments from command line
    rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
    torch.manual_seed(seed=args.seed)  # sets pytorch's seed

    train_data = IEMOCAP(experiment_name=args.experiment_name, mode='train')
    val_data = IEMOCAP(experiment_name=args.experiment_name, mode='val')
    test_data = IEMOCAP(experiment_name=args.experiment_name, mode='test')

    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=pad_collate, shuffle=True,  num_workers=0)
    val_data_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=pad_collate, shuffle=True,  num_workers=0)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=pad_collate, shuffle=True,  num_workers=0)
    #
    custom_blstm = LSTMBlock(
        input_dim=args.input_dim,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        output_dim=args.num_classes,
        num_layers=args.num_layers)


    conv_experiment = ExperimentBuilder(network_model=custom_blstm,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        weight_decay_coefficient=args.weight_decay_coefficient,
                                        use_gpu=args.use_gpu,
                                        continue_from_epoch=args.continue_from_epoch,
                                        train_data=train_data_loader, val_data=val_data_loader,
                                        test_data=test_data_loader)  # build an experiment object

    experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
