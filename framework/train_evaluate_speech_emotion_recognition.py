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
  (xx, yy, zz) = zip(*batch)
  #print(xx[0].shape)
  xx_trans = [torch.Tensor(x.transpose(1,0)) for x in xx]

  xx_pad = pad_sequence(xx_trans, batch_first=True)

  #print(xx_pad.shape)

  y_list = torch.Tensor([y for y in yy])

  z_list = torch.Tensor([z for z in zz])

  return xx_pad, y_list, z_list

if __name__ == '__main__':
    args = get_args()  # get arguments from command line
    rng = np.random.RandomState(seed=args.seed)  # set the seeds for the experiment
    torch.manual_seed(seed=args.seed)  # sets pytorch's seed


    if False and args.mpc:
        experiment_name="mspc"
    elif True:
        experiment_name = args.experiment_name

    train_data = IEMOCAP(experiment_name=experiment_name, layer_no=args.layer_no, mode='train')
    #print(args.experiment_name)
    val_data = IEMOCAP(experiment_name=experiment_name, layer_no=args.layer_no, mode='val')
    test_data = IEMOCAP(experiment_name=experiment_name, layer_no=args.layer_no, mode='test')

    #print(train_data[0])
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, collate_fn=pad_collate, shuffle=True,  num_workers=0)
    val_data_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=pad_collate, shuffle=True,  num_workers=0)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=pad_collate, shuffle=True,  num_workers=0)
    #
    custom_blstm = LSTMBlock(
        input_dim=args.input_dim,
        dropout=args.drop_out,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        output_dim=args.num_classes,
        num_layers=args.num_layers)


    conv_experiment = ExperimentBuilder(network_model=custom_blstm,
                                        layer_no=args.layer_no,
                                        SER=args.SER,
                                        experiment_name=args.experiment_name,
                                        num_epochs=args.num_epochs,
                                        gender_MTL=args.genderMTL,
                                        experiment_no=args.experiment_no,
                                        weight_decay_coefficient=args.weight_decay_coefficient,
                                        use_gpu=args.use_gpu,
                                        continue_from_epoch=args.continue_from_epoch,
                                        train_data=train_data_loader, val_data=val_data_loader,
                                        test_data=test_data_loader,
                                        lr=args.learning_rate,
                                        batch_size=args.batch_size)  # build an experiment object

    experiment_metrics, test_metrics = conv_experiment.run_experiment()  # run experiment and return experiment metrics
