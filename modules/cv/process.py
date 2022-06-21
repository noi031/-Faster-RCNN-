from tqdm import tqdm
import torch, math
from torch import nn

class warmup_computer():
    def __init__(self, warm_up_steps, total_len):
        self.warm_up_steps, self.total_len = warm_up_steps, total_len
    def get_lr(self, step):
        step += 1
        if step < self.warm_up_steps:
            return step / self.warm_up_steps
        else:
            step_past = step - self.warm_up_steps
            step_total = self.total_len - self.warm_up_steps
            return (1 + math.cos(step_past / step_total * math.pi)) / 2

class Trainprocess():
    def __init__(self, model_wrapper, train_loader, dev_loader = None, seed = 20):
        """
        This class wraps the training process.
        Parameters:
            model_wrapper: a class consisting of model and loss. It has a 'get_loss' method
                to compute loss(all inputs consistant with the dataloader construction),
                which returns a dictionary containing everything you want to show on the screen
                and save in the training process and a loss tensor needed to be backwarded.
                In order to control the training and testing process, it should also contain a 'train'
                method to set the model to the training state and a 'eval' method to set the model to
                the evaluation state.It also has a 'parameters' method to return all parameters in the model.
                It also has a 'save' method which can be used as save(epoch, validation_metric)
                where epoch is the current epoch id and validation_loss is the loss dictionary returned by
                the 'get_loss' function. If dev_loader is None, such a 'save' function will receive a None
                type as its validation loss. If dev_loader is not None, model_wrapper should also contain
                a metric_eval method to compute evaluation metrics for each epoch, which will be called
                as metric_eval(all inputs consistant with the dataloader construction) and returns a dictionary
                like the get_loss function. A metric_agg method is also needed to aggregate all results after
                validation which will be called as metric_agg(metric_list) and what it returns(a dictionary)
                will be the thing that shows on the screen and is saved finally.
            train_loader: dataloader for the train set.
            dev_loader: dataloader for the dev set, default None.
            seed: random seed, default 20.
        """
        self.model_wrapper, self.train_loader, self.dev_loader = model_wrapper, train_loader, dev_loader
        torch.random.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    def validate(self):
        print('validating:')
        self.model_wrapper.eval()
        metric_list = []
        with tqdm(self.dev_loader, ncols = 100) as pbar:
            for data in pbar:
                with torch.no_grad():
                    metric_list.append(self.model_wrapper.metric_eval(data))
        final_metric = self.model_wrapper.metric_agg(metric_list)
        print('validation results:', final_metric)
        return final_metric

    def train(self, max_epoch = 5, lr = 1e-3, warmup = True, warmup_steps = 2000):
        """
        This function starts training.
        Parameters:
            max_epoch: number of epochs to iterate, default 5.
            lr: learning rate, default 1e-3.
            wramup: Whether to warm up.
        """
        optimizer = torch.optim.SGD(self.model_wrapper.parameters(), lr = lr, momentum = 0.9,
                                    weight_decay = 2e-4)
        if warmup:
            warmuper = warmup_computer(warmup_steps, len(self.train_loader) * max_epoch)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = warmuper.get_lr)
        else:
            scheduler = None
        for epoch in range(max_epoch):
            print('epoch%d\nTraining:' %epoch)
            self.model_wrapper.train()
            with tqdm(self.train_loader, ncols = 120) as pbar:
                for data in pbar:
                    optimizer.zero_grad()
                    loss_dict = self.model_wrapper.get_loss(data)
                    nn.utils.clip_grad_norm_(self.model_wrapper.parameters(),
                                                max_norm = 20, norm_type = 2)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    pbar.set_postfix(loss_dict)
            self.model_wrapper.save(epoch, None)
            if self.dev_loader is not None:
                self.model_wrapper.save(epoch, self.validate())
        if not max_epoch:
            self.model_wrapper.save(0, self.validate())

class Testprocess():
    def __init__(self, model_wrapper, test_loader):
        """
        model_wrapper: a class consisting of your model. It should have a 'predict' method which
        will be used as predict(inputs) and returns a list of tensors to be saved.
        It should also have a 'result_to_text' method which converts a sample result tensor into
        one string line.
        """
        self.model_wrapper, self.test_loader = model_wrapper, test_loader
    def test(self, filepath):
        """
        filepath: filepath for saving the output.
        """
        print('Testing:')
        f = open(filepath, 'w', encoding = 'utf-8')
        self.model_wrapper.eval()
        with tqdm(self.test_loader, ncols = 100) as pbar:
            for data in pbar:
                with torch.no_grad():
                    output = self.model_wrapper.predict(data)
                    for x in output:
                        f.write(self.model_wrapper.result_to_text(x) + '\n')
        f.close()
        print('prediction done.')
