# -*- coding: utf-8 -*-
import numpy as np
import unittest

from dataset import Dataset


class DatasetTest(unittest.TestCase):
    def test_init(self):
        dataset = Dataset()

        self.assertEqual(len(dataset), 10240)

        for i in range(10):
            data = dataset[i]
            self.assertEqual(data.shape, (256,))
            self.assertEqual(data.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()

