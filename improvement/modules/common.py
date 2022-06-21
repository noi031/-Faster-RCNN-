import os, shutil, torch

class multitask_balancer():
    def __init__(self, loss_num, rou = 0.9):
        """
        Balance the multitask losses.
        Parameters:
            loss_num: number of tasks(losses).
            rou: moving average coefficient, float between 0 and 1.
        """
        self.avg_backd, self.avg_ford = torch.zeros([loss_num]), torch.zeros([loss_num])
        self.rou, self.rou_p = rou, 1
    def balance(self, loss_tensor):
        """
        Returns a new loss_tensor on multitask losses.
        Parameters:
            loss_tensor: multitask losses concatenated as a 1d-tensor.
        """
        assert len(loss_tensor) == len(self.avg_backd)
        avg_backd = torch.as_tensor(self.avg_backd, dtype = torch.float32, device = loss_tensor.device)
        avg_ford = torch.as_tensor(self.avg_ford, dtype = torch.float32, device = loss_tensor.device)
        avg_backd += (1 - self.rou) * self.rou_p * loss_tensor.detach()
        avg_ford = self.rou * avg_ford + (1 - self.rou) * loss_tensor.detach()
        self.rou_p *= self.rou
        self.avg_backd, self.avg_ford = avg_backd.cpu(), avg_ford.cpu()
        vol = avg_ford / (1e-6 + avg_backd)
        vol /= (1e-6 + vol.sum())
        return loss_tensor / (1e-6 + avg_backd) * vol

class data_reformation():
    def __init__(self, datapath, filereader, writefile):
        """
        This class perpares your data for faster training.
        Parameters:
            datapath: directory where you store all the data. Every file/subdirectory under this path is
            a collection of samples. After class initialization the whole directory will be reformed and
            therefore, you'd better backup your data in case of errors.
            filereader: an iterable object that returns all samples in a file/directory. This object will
                be initialized by filereader(filepath).
            writefile: a function that writes a sample into a directory. It will be called as
                writefile(sample_object, dirpath)
        """
        name_list, self.datasize, self.datapath = os.listdir(datapath), 0, self.datapath
        print('computing data size...')
        for name in name_list:
            for _ in filereader(os.path.join(datapath, name)):
                self.datasize += 1
        self.curid = 0
        def build_tree(l, r, path):
            if l == r:
                flag = 1
                while self.curid < len(name_list) and flag:
                    try:
                        sample = curfile.__next__()
                        flag = 0
                    except:
                        if self.curid:
                            filepath = os.path.join(datapath, name_list[self.curid - 1])
                            if os.path.isfile(filepath):
                                os.remove(filepath)
                            else:
                                shutil.rmtree(filepath)
                        name = name_list[self.curid]
                        curfile = filereader(os.path.join(datapath, name))
                        self.curid += 1
                if flag:
                    raise ValueError('Error occured in data preparation!')
                writefile(sample, path)
            else:
                mid = l + r >> 1
                lpath, rpath = os.path.join(path, '0'), os.path.join(path, '1')
                os.mkdir(lpath);os.mkdir(rpath)
                build_tree(l, mid, lpath);build_tree(mid + 1, r, rpath)
        build_tree(0, self.datasize - 1, datapath)
    def __getitem__(self, index):
        """
        This function returns the filepath with the provided index.
        Parameters:
            index: index of samples needed.
        """
        l, r, path = 0, self.datasize, self.datapath
        while l < r:
            mid = l + r >> 1
            path = os.path.join(path, str(int(index > mid)))
            if index <= mid:
                r = mid
            else:
                l = mid + 1
        return os.path.join(path, os.listdir(path))
    def __len__(self):
        """
        This function returns the total size of all data.
        """
        return self.datasize
