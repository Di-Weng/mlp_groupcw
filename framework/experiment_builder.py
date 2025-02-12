import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mockingjay.nn_mockingjay import MOCKINGJAY
from runner_mockingjay import get_mockingjay_model
import tqdm
import os
import numpy as np
import time
from downstream.solver import get_mockingjay_optimizer
import sys
from apex import amp

from storage_utils import save_statistics
emotion_classes = {"ang": 0, "hap": 1,  "neu": 2, "sad": 3}

class ExperimentBuilder(nn.Module):
    def __init__(self, beta, SER, layer_no, network_model, batch_size, experiment_no, experiment_name, num_epochs, gender_MTL, train_data, val_data,
                 test_data, weight_decay_coefficient, use_gpu, lr, continue_from_epoch=-1):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param experiment_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param num_epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param use_gpu: A boolean indicating whether to use a GPU or not.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()


        self.layer_no=layer_no
        self.beta=beta
        self.gender_MTL=gender_MTL
        self.SER=SER
        self.lr = lr
        self.batch_size = batch_size
        self.experiment_name = experiment_name
        self.experiment_no = experiment_no
        self.model = network_model
        self.current_epoch_emo_count = {0: 0, 1: 0, 2: 0, 3: 0}
        self.current_epoch_correct_count = {0: 0, 1: 0, 2: 0, 3: 0}

        self.eval_current_epoch_emo_count = {0: 0, 1: 0, 2: 0, 3: 0}
        self.eval_current_epoch_correct_count = {0: 0, 1: 0, 2: 0, 3: 0}

        self.num_epochs = num_epochs

        options_feature_based = {
            'ckpt_file': 'MPC/mockingjay-500000.ckpt',
            'load_pretrain': 'True',
            'no_grad': 'True',
            'dropout': 'default'
        }

        if self.experiment_name == "mpc":
            # self.mockingjay=MOCKINGJAY(options=options_feature_based, inp_dim=160)
            self.mockingjay = get_mockingjay_model(from_path='MPC/mockingjay-500000.ckpt')
        elif self.experiment_name.startswith('mpc_finetune'):
            options = {
                # change path; needs small model
                'ckpt_file': 'MPC/mpcbase/mockingjay-500000.ckpt',
                'load_pretrain': 'True',
                'no_grad': 'False',
                'dropout': 'default'
            }
            self.mockingjay_model = MOCKINGJAY(options=options, inp_dim=160)

        if torch.cuda.device_count() > 1 and use_gpu:
            self.device = torch.cuda.current_device()
            self.model.to(self.device)
            self.model = nn.DataParallel(module=self.model)
            print('Use Multi GPU', self.device)
        elif torch.cuda.device_count() == 1 and use_gpu:
            self.device =  torch.cuda.current_device()
            self.model.to(self.device)  # sends the model from the cpu to the gpu

            print('Use GPU', self.device)
        else:
            print("use CPU")
            self.device = torch.device('cpu')  # sets the device to be CPU
            print(self.device)







        self.model.reset_parameters()  # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        print('System learnable parameters')
        num_conv_layers = 0
        num_linear_layers = 0
        total_num_parameters = 0
        for name, value in self.named_parameters():
            print(name, value.shape)
            if all(item in name for item in ['conv', 'weight']):
                num_conv_layers += 1
            if all(item in name for item in ['linear', 'weight']):
                num_linear_layers += 1
            total_num_parameters += np.prod(value.shape)

        print('Total number of parameters', total_num_parameters)
        print('Total number of conv layers', num_conv_layers)
        print('Total number of linear layers', num_linear_layers)

        # static learning rate
        if self.experiment_name.startswith('mpc_finetune'):
            # self.params: only for finetune
            self.params = list(self.mockingjay_model.named_parameters()) + list(self.model.named_parameters())
            self.optimizer = get_mockingjay_optimizer(params=self.params,
                                                 lr=self.lr,
                                                 warmup_proportion=0.7,
                                                 training_steps= int(3872 * self.num_epochs / self.batch_size) + 1)

            # apex 混合精度
            [self.model, self.mockingjay_model], optimizer = amp.initialize([self.model, self.mockingjay_model],
                                                                            self.optimizer, opt_level="O1")


        else:
            self.optimizer = optim.Adam(self.parameters(), amsgrad=False,
                                        weight_decay=weight_decay_coefficient, lr=self.lr)

        # self.learning_rate_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
        #                                                                     T_max=num_epochs,
        #                                                                     eta_min=0.00002)

        # Generate the directory names
        self.experiment_folder = os.path.abspath(experiment_name+"_"+experiment_no)
        self.experiment_logs = os.path.abspath(os.path.join(self.experiment_folder, "result_outputs"))
        self.experiment_saved_models = os.path.abspath(os.path.join(self.experiment_folder, "saved_models"))

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_acc = 0.

        if not os.path.exists(self.experiment_folder):  # If experiment directory does not exist
            os.mkdir(self.experiment_folder)  # create the experiment directory
        if not os.path.exists(self.experiment_logs):
            os.mkdir(self.experiment_logs)  # create the experiment log directory
        if not os.path.exists(self.experiment_saved_models):
            os.mkdir(self.experiment_saved_models)  # create the experiment saved models directory

        self.criterion = nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU

        if continue_from_epoch == -2:  # if continue from epoch is -2 then continue from latest saved model
            self.state, self.best_val_model_idx, self.best_val_model_acc = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx='latest')  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = int(self.state['model_epoch'])

        elif continue_from_epoch > -1:  # if continue from epoch is greater than -1 then
            self.state, self.best_val_model_idx, self.best_val_model_acc = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = continue_from_epoch
        else:
            self.state = dict()
            self.starting_epoch = 0

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def run_train_iter(self, x, y, z):
        """
        Receives the inputs and targets for the model and runs a training iteration. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)

        x, y, z = x.float().to(device=self.device), y.long().to(
            device=self.device), z.long().to(
            device=self.device)  # send data to device as torch tensors

        if False and self.experiment_name=="mpc":
            #print(x.shape)
            x=x.cpu()
            x=self.mockingjay.forward(spec=x, all_layers=True, tile=True)
            print(x.shape)
            x=x[:,self.layer_no,:,:]
        elif self.experiment_name.startswith('mpc_finetune'):
            # mockingjay_model forward
            x = self.mockingjay_model(x.transpose(0, 1)).transpose(0, 1)

        out1, out2 = self.model.forward(x)  # forward the data in the model

        #print(out1.shape)
        #print(y.shape)
        loss=0
        if self.SER:
            loss +=F.cross_entropy(input=out1, target=y)  # compute loss

        if self.gender_MTL:
            #print("training gender..")
            loss+=self.beta*F.cross_entropy(input=out2, target=z)

        if(self.experiment_name.startswith('mpc_finetune')):
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
            loss.backward()  # backpropagate to compute gradients for current iter loss

            self.optimizer.step()  # update network parameters
            # self.learning_rate_scheduler.step(epoch=self.current_epoch)


        _, predicted1 = torch.max(out1.data, 1)  # get argmax of predictions
        _, predicted2 = torch.max(out2.data, 1)  # get argmax of predictions
        accuracy1 = np.mean(list(predicted1.eq(y.data).cpu()))  # compute accuracy
        accuracy2 = np.mean(list(predicted2.eq(z.data).cpu()))
        for label_idx, label in enumerate(y.data):
            label = int(label)
            self.current_epoch_emo_count[label]+=1
            if predicted1[label_idx]== label:
                self.current_epoch_correct_count[label]+=1

        # un_accuracy1 =
        return loss.cpu().detach().numpy(), accuracy1, accuracy2

    def run_evaluation_iter(self, x, y, z):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode

        x, y, z = x.float().to(device=self.device), y.long().to(
            device=self.device), z.long().to(
            device=self.device)  # convert data to pytorch tensors and send to the computation device

        if False and self.experiment_name=="mpc":
            x=x.cpu()
            #print(x.shape)
            x=self.mockingjay.forward(spec=x, all_layers=True, tile=True)
            print(x.shape)
            x=x[:,self.layer_no,:,:]
        elif self.experiment_name.startswith('mpc_finetune'):
            # mockingjay_model forward
            x = self.mockingjay_model(x.transpose(0, 1)).transpose(0, 1)


        #print(x.shape)
        out1, out2 = self.model.forward(x)  # forward the data in the model

        loss = F.cross_entropy(input=out1, target=y)  # compute loss

        if self.gender_MTL:
            print("training gender...")
            loss+= self.beta*F.cross_entropy(input=out2, target=z)
        _, predicted1 = torch.max(out1.data, 1)  # get argmax of predictions
        _, predicted2 = torch.max(out2.data, 1)  # get argmax of predictions
        accuracy1 = np.mean(list(predicted1.eq(y.data).cpu()))  # compute accuracy
        accuracy2 = np.mean(list(predicted2.eq(z.data).cpu()))  # compute accuracy

        for label_idx, label in enumerate(y.data):
            label = int(label)
            self.eval_current_epoch_emo_count[label]+=1
            if predicted1[label_idx]== label:
                self.eval_current_epoch_correct_count[label]+=1

        return loss.cpu().detach().numpy(), accuracy1, accuracy2

    def save_model(self, model_save_dir, model_save_name, model_idx, best_validation_model_idx,
                   best_validation_model_acc):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_acc: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.

        """
        self.state['network'] = self.state_dict()  # save network parameter and other variables.
        self.state['best_val_model_idx'] = best_validation_model_idx  # save current best val idx
        self.state['best_val_model_acc'] = best_validation_model_acc  # save current best val acc

        torch.save(self.state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath


    def delete_model(self, model_save_dir, model_save_name, model_idx):
        """
        Delete the model files other than the best model
        """
        best_model_path = os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx)))
        file_list = os.listdir(model_save_dir)
        for f in file_list:
            f_path = os.path.join(model_save_dir, f)

            # remove all
            os.remove(f_path)
            print('remove all models')
            #
            # if f_path != best_model_path:
            #     os.remove(f_path)
            #     print('file removed: {}'.format(f_path))

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state, state['best_val_model_idx'], state['best_val_model_acc']

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"val_Gender_acc":[], "val_SER_acc":[], "train_acc_GENDER": [], "train_acc_SER": [], "train_loss": [],
                        "val_loss": [], "ua_train": [], "ua_eval":[]}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.num_epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_acc_SER": [], "train_acc_GENDER": [], "train_loss": [],
                                    "val_Gender_acc": [], "val_SER_acc": [], "val_loss": []}
            self.current_epoch = epoch_idx
            # init un_accuracy
            self.current_epoch_emo_count = {0: 0, 1: 0, 2: 0, 3: 0}
            self.current_epoch_correct_count = {0: 0, 1: 0, 2: 0, 3: 0}

            with tqdm.tqdm(total=len(self.train_data), ascii=True) as pbar_train:  # create a progress bar for training
                try:
                    for idx, (x, y, z) in enumerate(self.train_data):  # get data batches
                        loss, accuracy1, accuracy2 = self.run_train_iter(x=x, y=y, z=z)  # take a training iter step
                        current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                        current_epoch_losses["train_acc_SER"].append(accuracy1)  # add current iter acc to the train acc list
                        current_epoch_losses["train_acc_GENDER"].append(accuracy2)
                        pbar_train.update(1)
                        pbar_train.set_description("loss: {:.4f}, accuracy_SER: {:.4f}, accuracy_Gender: {:.4f}".format(loss, accuracy1, accuracy2))
                        torch.cuda.empty_cache()
                except KeyboardInterrupt:
                    pbar_train.close()
                    raise
                pbar_train.close()
            # get un_accuracy result

            self.eval_current_epoch_emo_count = {0: 0, 1: 0, 2: 0, 3: 0}
            self.eval_current_epoch_correct_count = {0: 0, 1: 0, 2: 0, 3: 0}

            with tqdm.tqdm(total=len(self.val_data), ascii=True) as pbar_val:  # create a progress bar for validation
                try:
                    for x, y, z in self.val_data:  # get data batches
                        loss, accuracy1, accuracy2 = self.run_evaluation_iter(x=x, y=y, z=z)  # run a validation iter
                        current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                        current_epoch_losses["val_SER_acc"].append(accuracy1)  # add current iter acc to val acc lst.
                        current_epoch_losses["val_Gender_acc"].append(accuracy2)  # add current iter acc to val acc lst.
                        pbar_val.update(1)  # add 1 step to the progress bar
                        pbar_val.set_description("loss: {:.4f}, accuracy_SER: {:.4f}, accuracy_Gender: {:.4f}".format(loss, accuracy1, accuracy2))
                        torch.cuda.empty_cache()
                except KeyboardInterrupt:
                    pbar_val.close()
                    raise
                pbar_val.close()
            val_mean_accuracy = np.mean(current_epoch_losses['val_SER_acc'])
            if val_mean_accuracy > self.best_val_model_acc:  # if current epoch's mean val acc is greater than the saved best val acc then
                self.best_val_model_acc = val_mean_accuracy  # set the best val model acc to be current epoch's val accuracy
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx

            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(
                    value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            ua_train=0
            ua_val=0
            for key, value in self.current_epoch_correct_count.items():
                ua_train+=value/self.current_epoch_emo_count[key]/4
            total_losses["ua_train"].append(ua_train)
            for key, value in self.eval_current_epoch_correct_count.items():
                ua_val+=value/self.eval_current_epoch_emo_count[key]/4
            total_losses["ua_eval"].append(ua_val)



            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False)  # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = ", ".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}:".format(epoch_idx), out_string, "epoch time", epoch_elapsed_time, "seconds")
            self.state['model_epoch'] = epoch_idx
            if(self.best_val_model_idx == epoch_idx):
                # replace best model

                #remove all model files, THEN save new best one.
                for model_filename in os.listdir(self.experiment_saved_models):
                    current_model = os.path.join(self.experiment_saved_models, model_filename)
                    os.remove(current_model)
                self.save_model(model_save_dir=self.experiment_saved_models,
                                # save model and best val idx and best val acc, using the model dir, model name and model idx
                                model_save_name="train_model", model_idx=epoch_idx,
                                best_validation_model_idx=self.best_val_model_idx,
                                best_validation_model_acc=self.best_val_model_acc)

            # save(update) latest model every epoch
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest',
                            best_validation_model_idx=self.best_val_model_idx,
                            best_validation_model_acc=self.best_val_model_acc)

            # early stopping
            # criteria = int(self.num_epochs * 0.1)
            # window_valloss = total_losses['val_loss'][-(criteria+1):]
            # sorted_window_valloss = sorted(window_valloss,reverse = False)
            #if (len(window_valloss) >= (criteria + 1)):
            #    if (sorted_window_valloss[0] == window_valloss[0]):
            #        print("Early Stop at Epoch {}:".format(epoch_idx), "Best val model index", self.best_val_model_idx, "Best val model accuracy", self.best_val_model_acc)
            #        break

        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        #                         # load best validation model
                        model_save_name="train_model")
        current_epoch_losses = {"test_Gender_acc": [],"test_SER_acc": [], "test_loss": []}  # initialize a statistics dict

        self.eval_current_epoch_emo_count = {0: 0, 1: 0, 2: 0, 3: 0}
        self.eval_current_epoch_correct_count = {0: 0, 1: 0, 2: 0, 3: 0}

        with tqdm.tqdm(total=len(self.test_data), ascii=True) as pbar_test:  # ini a progress bar
            try:
                for x, y, z in self.test_data:  # sample batch
                    loss, accuracy1, accuracy2 = self.run_evaluation_iter(x=x,
                                                              y=y, z=z)  # compute loss and accuracy by running an evaluation step
                    current_epoch_losses["test_loss"].append(loss)  # save test loss
                    current_epoch_losses["test_SER_acc"].append(accuracy1)  # save test accuracy
                    current_epoch_losses["test_Gender_acc"].append(accuracy2)  # save test accuracy
                    pbar_test.update(1)  # update progress bar status
                    pbar_test.set_description(
                        "loss: {:.4f}, SER_ accuracy: {:.4f},  Test_ accuracy: {:.4f},".format(loss, accuracy1, accuracy2))  # update progress bar string output
                    torch.cuda.empty_cache()
            except KeyboardInterrupt:
                pbar_test.close()
                raise
            pbar_test.close()
        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format
        ua_test = 0
        for key, value in self.eval_current_epoch_correct_count.items():
            ua_test += value / self.eval_current_epoch_emo_count[key] / 4
        test_losses["ua_test"] = [ua_test]
        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)

        # print('Sorting model files')
        # self.delete_model(model_save_dir=self.experiment_saved_models, model_save_name="train_model",
        #                   model_idx=self.best_val_model_idx)

        return total_losses, test_losses
