import os.path
import random
import logging
import numpy as np
import pandas as pd
from sklearn import preprocessing
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Sampler

from data.dataset import Textdataset, Tabdataset


def setupTransformers(num_party, num_features):
    from PIL import Image

    class Split(object):
        """
        Args:
            start: start column
            end: end column
        """

        def __init__(self, start, end):
            assert isinstance(start, int) or (isinstance(end, int))
            self.start = start
            self.end = end

        def __call__(self, image):
            """
            Args:
                image: PIL image
                Return: PIL image
            """
            array = np.array(image).copy()
            ndim = array.ndim
            if ndim == 2:
                # mnist shape (28,28)
                return Image.fromarray(array[:, self.start:self.end])
            elif ndim == 3:
                # cifar10 shape (32,32,3)
                return Image.fromarray(array[:, self.start:self.end, :])
            else:
                logging.info('Wrong input!')
                exit()

    transformers = []
    end_index = 0
    for i in range(num_party):
        start_index = end_index
        end_index = start_index + num_features[i]
        transformers.append(transforms.Compose([Split(start=start_index, end=end_index), transforms.ToTensor()]))
    return transformers


def loadDataset(dataset_name, num_party, num_features, max_length=80):
    database = '/home/qpy/datasets' if os.path.exists(
        '/home/qpy/datasets') else 'C:\\Users\\Qiupys\\PycharmProjects\\datasets'
    tran_datasets, test_datasets = [], []
    if dataset_name == 'mnist':
        transformers = setupTransformers(num_party=num_party, num_features=num_features)
        for i in range(num_party):
            from torchvision.datasets import MNIST
            tran_datasets.append(MNIST(root=database, train=True, transform=transformers[i], download=True))
            test_datasets.append(MNIST(root=database, train=False, transform=transformers[i]))
    elif dataset_name == 'cifar10':
        transformers = setupTransformers(num_party=num_party, num_features=num_features)
        for i in range(num_party):
            from torchvision.datasets import CIFAR10
            tran_datasets.append(CIFAR10(root=database, train=True, transform=transformers[i], download=True))
            test_datasets.append(CIFAR10(root=database, train=False, transform=transformers[i]))
    elif dataset_name == 'cifar100':
        transformers = setupTransformers(num_party=num_party, num_features=num_features)
        for i in range(num_party):
            from torchvision.datasets import CIFAR100
            tran_datasets.append(CIFAR100(root=database, train=True, transform=transformers[i], download=True))
            test_datasets.append(CIFAR100(root=database, train=False, transform=transformers[i]))
    elif dataset_name == 'emotion':
        transformers = setupTransformers(num_party=num_party, num_features=num_features)
        for i in range(num_party):
            tran_datasets.append(ImageFolder(os.path.join(database, 'emotion/train'), transformers[i]))
            test_datasets.append(ImageFolder(os.path.join(database, 'emotion/test'), transformers[i]))
    elif dataset_name == 'imdb':
        train_data, test_data = pd.read_csv(os.path.join(database, 'imdb/train.csv')), pd.read_csv(
            os.path.join(database, 'imdb/test.csv'))
        # train_data.info()

        train_data.dropna(subset=['text'], inplace=True)
        # # random sampling total data for experiment
        # train_data, test_data = train_data.sample(frac=0.001, random_state=0), test_data.sample(frac=0.001, random_state=0)
        logging.info(train_data['sentiment'].value_counts())

        train_texts, y_train = np.array(train_data['text']), np.array(train_data['sentiment'] == 'pos')
        test_texts, y_test = np.array(test_data['text']), np.array(test_data['sentiment'] == 'pos')

        train_datasets, test_datasets = [], []
        max_len = max_length
        start_index = 0
        for i in range(num_party):
            train_datasets.append(
                Textdataset(train_texts, label=y_train, start=start_index, end=start_index + num_features[i],
                            max_len=max_len))
            test_datasets.append(
                Textdataset(test_texts, label=y_test, start=start_index, end=start_index + num_features[i],
                            max_len=max_len))
            start_index += num_features[i]
        return train_datasets, test_datasets
    elif dataset_name == 'criteo':
        columns = ['label', *(f'I{i}' for i in range(1, 14)), *(f'C{i}' for i in range(1, 27))]
        df = pd.read_csv(os.path.join(database, 'criteo/dac_sample.txt'), sep='\t', names=columns).fillna(
            method='bfill', axis=0)
        df.dropna(how='any', axis=0, inplace=True)

        # Preprocessing Integer Features
        integer_cols = [c for c in columns if 'I' in c]
        df[integer_cols] = preprocessing.MinMaxScaler().fit_transform(df[integer_cols])

        # Preprocess Categorical Features
        remove_list = ['C3', 'C4', 'C7', 'C10', 'C11', 'C12', 'C13', 'C15', 'C16', 'C18', 'C19', 'C21', 'C24', 'C26']
        df = df.drop(remove_list, axis=1)
        cat_cols = [c for c in columns if 'C' in c and c not in remove_list]
        df[cat_cols] = df[cat_cols].apply(preprocessing.LabelEncoder().fit_transform)
        # for col in cat_cols:
        #     preprocessing.OneHotEncoder(sparse=False).fit_transform(df[[col]])

        X, y = df.drop('label', axis=1), df['label'].astype(int)
        logging.info(y.value_counts())
        # # Preprocessing Label Imbalance
        # from imblearn.under_sampling import RandomUnderSampler
        # rus = RandomUnderSampler()
        # X, y = rus.fit_resample(X, y)
        # logging.info(y.value_counts())
        from imblearn.over_sampling import SMOTE
        sm = SMOTE()
        X, y = sm.fit_resample(X, y)
        logging.info(y.value_counts())

        # Generate Dataset
        X, y = np.array(X), np.array(y)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        train_datasets, test_datasets = [], []
        start_index = 0
        for i in range(num_party):
            train_datasets.append(
                Tabdataset(X_train, label=y_train, start=start_index, end=start_index + num_features[i]))
            test_datasets.append(Tabdataset(X_test, label=y_test, start=start_index, end=start_index + num_features[i]))
            start_index += num_features[i]
        return train_datasets, test_datasets
    elif dataset_name == 'bank':
        data = pd.read_csv(os.path.join(database, 'bankruptcy.csv'))
        X, y = data.drop('Bankrupt?', axis=1), data['Bankrupt?'].astype(int)
        logging.info(y.value_counts())
        from imblearn.over_sampling import SMOTE
        sm = SMOTE()
        X, y = sm.fit_resample(X, y)
        logging.info(y.value_counts())
        X, y = np.array(X), np.array(y)
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
        train_datasets, test_datasets = [], []
        start_index = 0
        for i in range(num_party):
            train_datasets.append(
                Tabdataset(X_train, label=y_train, start=start_index, end=start_index + num_features[i]))
            test_datasets.append(Tabdataset(X_test, label=y_test, start=start_index, end=start_index + num_features[i]))
            start_index += num_features[i]
        return train_datasets, test_datasets
    else:
        logging.info('Dataset does not exist!!!')
        exit()
    return tran_datasets, test_datasets


def dataLoader(datasets, batch_size):
    loaders = []
    order = list(range(len(datasets[0])))
    random.shuffle(order)

    class MySampler(Sampler):
        r"""Samples elements according to the previously generated order.
        """

        def __init__(self, data_source, order):
            super().__init__(data_source)
            self.data_source = data_source
            self.order = order

        def __iter__(self):
            return iter(self.order)

        def __len__(self):
            return len(self.data_source)

    for dataset in datasets:
        loaders.append(
            DataLoader(dataset, batch_size, shuffle=False, sampler=MySampler(dataset, order), drop_last=True))
    return loaders
