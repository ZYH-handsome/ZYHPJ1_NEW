import os
import numpy as np
from tqdm import tqdm
import albumentations as A

class Runner:
    """
    The class to train and evaluate the model
    """
    def __init__(self, model, loss_fn, metric, batch_size=32, optimizer=None, lr_scheduler=None, early_stop=False, patience=2, data_augmentation=False):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_fn = loss_fn
        self.metric = metric
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.patience = patience
        self.data_augmentation = data_augmentation

        self.best_score = 0

        self.train_loss = []
        self.train_score = []

        self.valid_loss = []
        self.valid_score = []

    def train(self, train_set, valid_set, epoch=5, save_dir=None, log_iter=100):

        assert self.optimizer, 'Optimizer has not been loaded.'

        if not save_dir:
            save_dir = 'best_model'

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # data augmentation
        def augment(img):
            transform = A.Compose([
                A.Rotate(limit=10, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1,
                    scale_limit=0.0,
                    rotate_limit=0,
                    p=0.5
                )
            ])
            return transform(image=img)['image']

        if self.data_augmentation:
            train_imgs, train_labs = train_set
            img_size = train_imgs.shape[-1]
            img_side_len = int(np.sqrt(img_size))
            aug_imgs = train_imgs.reshape(-1, img_side_len, img_side_len)

            for i in range(len(aug_imgs)):
                aug_imgs[i] = augment(aug_imgs[i])

            aug_imgs = aug_imgs.reshape(-1, img_size)

            train_imgs = np.concatenate((train_imgs, aug_imgs), axis=0)
            train_labs = np.concatenate((train_labs, train_labs))
            train_set = [train_imgs, train_labs]

        for k in range(epoch):
            # permute the data
            train_X, train_y = train_set
            idx = np.random.permutation(train_X.shape[0])
            train_X = train_X[idx]
            train_y = train_y[idx]

            # begin training
            for i in tqdm(range(train_X.shape[0] // self.batch_size + 1)):
                X = train_X[self.batch_size * i: self.batch_size * (i + 1)]
                y = train_y[self.batch_size * i: self.batch_size * (i + 1)]

                train_loss, train_score = self.eval(X, y)
                self.train_loss.append(train_loss)
                self.train_score.append(train_score)

                self.loss_fn.backward()

                self.optimizer.step()

                if i % 50 == 0:
                    valid_loss, valid_score = self.eval(valid_set[0], valid_set[1])
                    self.valid_loss.append(valid_loss)
                    self.valid_score.append(valid_score)

                if i % log_iter == 0:
                    print(f'epoch:{k+1}, iter:{i}\n'
                          f'train_loss:{train_loss}, train_score:{train_score}\n'
                          f'valid_loss:{valid_loss}, valid_score:{valid_score}\n')

            if self.lr_scheduler:
                self.lr_scheduler.step()

            model_dir = os.path.join(save_dir, 'models')
            if not os.path.exists(model_dir):
                os.mkdir(model_dir)

            save_path = os.path.join(model_dir, f'epoch_{k+1}.pickle')
            self.model.save_model(save_path)

            if self.valid_score[-1] > self.best_score:
                print(f'Best score has been updated:{self.best_score:.5f}->{self.valid_score[-1]:.5f}')
                best_model_save_path = os.path.join(model_dir, f'best_model.pickle')
                self.best_score = self.valid_score[-1]
                self.model.save_model(best_model_save_path)
            else:
                self.patience -= 1

            if self.patience <= 0 and self.early_stop:
                print(f'[Train]Early stop at epoch{k+1}')
                break


    def eval(self, data, labels):
        predicts = self.model(data)
        loss = self.loss_fn(labels, predicts)
        score = self.metric(labels, predicts)
        return loss, score