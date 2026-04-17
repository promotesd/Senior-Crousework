import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class RSICD(BaseDataset):
    """
    RSICD

    Reference:
    Exploring models and data for remote sensing image caption generation

    URL: https://ieeexplore.ieee.org/document/8240966

    Dataset statistics:
    # identities: 10921 images, 5 sentences/image, 224*224
    """
    dataset_dir = 'RSICD'

    def __init__(self, root='', verbose=True):
        super(RSICD, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        # self.img_dir = op.join(self.dataset_dir, 'imgs/')
        self.img_dir = self.dataset_dir

        ##在RSICD中找到训练，验证和测试的json文件，并且读取json文件中的所有内容，返回字典
        self.train_annos = read_json(op.join(self.dataset_dir, 'rsicd_train.json'))
        self.test_annos = read_json(op.join(self.dataset_dir, 'rsicd_test.json'))
        self.val_annos = read_json(op.join(self.dataset_dir, 'rsicd_val.json'))
        self._check_before_run()


        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> RSICD Images and Captions are loaded")
            self.show_dataset_info()

  
    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            image_id = 0
            pre_pid = annos[0]['image_id']
            for anno in annos:
                pid = int(anno['image_id'])
                pid_container.add(pid)
                img_path = op.join(self.img_dir, anno['image'])
                caption = anno['caption']  # caption list
                if pre_pid != pid:
                    image_id += 1
                    pre_pid = pid
                # dataset.append((pid, image_id, img_path, caption))
                dataset.append((pid, image_id, img_path, caption))
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            caption_img_paths = []

            img2pid = {}

            for anno in annos:
                img_path = op.join(self.img_dir, anno['image'])

                if img_path not in img2pid:
                    cur_pid = len(img2pid)
                    img2pid[img_path] = cur_pid
                    pid_container.add(cur_pid)

                    img_paths.append(img_path)
                    image_pids.append(cur_pid)
                else:
                    cur_pid = img2pid[img_path]


                caption_list = anno['caption']
                if isinstance(caption_list, str):
                    caption_list = [caption_list]

                for caption in caption_list:
                    captions.append(caption)
                    caption_pids.append(cur_pid)
                    caption_img_paths.append(img_path)

            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "caption_pids": caption_pids,
                "captions": captions,
                "caption_img_paths": caption_img_paths,
            }
            return dataset, pid_container

        ##以下代码有问题，img_paths.append(img_path)，我们在做验证的时候，只希望每一张图像只出现一次，而不是一张图像按照anno出现的次数而在
        ##数据中多次叠加，因此我们将代码修改。
        # else:
        #     dataset = {}
        #     img_paths = []
        #     captions = []
        #     image_pids = []
        #     caption_pids = []
        #     pid = 0
        #     caption_img_paths = []
        #     for anno in annos:
        #         pid_container.add(pid)
        #         img_path = op.join(self.img_dir, anno['image'])
        #         img_paths.append(img_path)
        #         image_pids.append(pid)
        #         caption_list = anno['caption']  # caption list
        #         for caption in caption_list:
        #             captions.append(caption)
        #             caption_pids.append(pid)
        #             caption_img_paths.append(img_path)
        #         pid += 1
        #     dataset = {
        #         "image_pids": image_pids,
        #         "img_paths": img_paths,
        #         "caption_pids": caption_pids,
        #         "captions": captions,
        #         "caption_img_paths": caption_img_paths
        #     }
        #     return dataset, pid_container


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))