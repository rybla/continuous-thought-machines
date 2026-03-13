import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
import random
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
from datasets import load_dataset


class SortDataset(Dataset):
    def __init__(self, N):
        self.N = N

    def __len__(self):
        return 10000000

    def __getitem__(self, idx):
        data = torch.zeros(self.N).normal_()
        ordering = torch.argsort(data)
        inputs = data
        return (inputs), (ordering)


class QAMNISTDataset(Dataset):
    """A QAMNIST dataset that includes plus and minus operations on MNIST digits."""

    def __init__(
        self,
        base_dataset,
        num_images,
        num_images_delta,
        num_repeats_per_input,
        num_operations,
        num_operations_delta,
    ):
        self.base_dataset = base_dataset

        self.num_images = num_images
        self.num_images_delta = num_images_delta
        self.num_images_range = self._calculate_num_images_range()

        self.operators = ["+", "-"]
        self.num_operations = num_operations
        self.num_operations_delta = num_operations_delta
        self.num_operations_range = self._calculate_num_operations_range()

        self.num_repeats_per_input = num_repeats_per_input

        self.current_num_digits = num_images
        self.current_num_operations = num_operations

        self.modulo_base = 10

        self.output_range = [0, 9]

    def _calculate_num_images_range(self):
        min_val = self.num_images - self.num_images_delta
        max_val = self.num_images + self.num_images_delta
        assert (
            min_val >= 1
        ), f"Minimum number of images must be at least 1, got {min_val}"
        return [min_val, max_val]

    def _calculate_num_operations_range(self):
        min_val = self.num_operations - self.num_operations_delta
        max_val = self.num_operations + self.num_operations_delta
        assert (
            min_val >= 1
        ), f"Minimum number of operations must be at least 1, got {min_val}"
        return [min_val, max_val]

    def set_num_digits(self, num_digits):
        self.current_num_digits = num_digits

    def set_num_operations(self, num_operations):
        self.current_num_operations = num_operations

    def _get_target_and_question(self, targets):
        question = []
        equations = []
        num_digits = self.current_num_digits
        num_operations = self.current_num_operations

        # Select the initial digit
        selection_idx = np.random.randint(num_digits)
        first_digit = targets[selection_idx]
        question.extend([selection_idx] * self.num_repeats_per_input)
        # Set current_value to the initial digit (mod is applied in each operation)
        current_value = first_digit % self.modulo_base

        # For each operation, build an equation line
        for _ in range(num_operations):
            # Choose the operator ('+' or '-')
            operator_idx = np.random.randint(len(self.operators))
            operator = self.operators[operator_idx]
            encoded_operator = -(operator_idx + 1)  # -1 for '+', -2 for '-'
            question.extend([encoded_operator] * self.num_repeats_per_input)

            # Choose the next digit
            selection_idx = np.random.randint(num_digits)
            digit = targets[selection_idx]
            question.extend([selection_idx] * self.num_repeats_per_input)

            # Compute the new value with immediate modulo reduction
            if operator == "+":
                new_value = (current_value + digit) % self.modulo_base
            else:  # operator is '-'
                new_value = (current_value - digit) % self.modulo_base

            # Build the equation string for this step
            equations.append(
                f"({current_value} {operator} {digit}) mod {self.modulo_base} = {new_value}"
            )
            # Update current value for the next operation
            current_value = new_value

        target = current_value
        question_readable = "\n".join(equations)
        return target, question, question_readable

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        images, targets = [], []
        for _ in range(self.current_num_digits):
            image, target = self.base_dataset[np.random.randint(self.__len__())]
            images.append(image)
            targets.append(target)

        observations = torch.repeat_interleave(
            torch.stack(images, 0), repeats=self.num_repeats_per_input, dim=0
        )
        target, question, question_readable = self._get_target_and_question(targets)
        return observations, question, question_readable, target


class ImageNet(Dataset):
    def __init__(self, which_split, transform):
        """
        Most simple form of the custom dataset structure.
        Args:
            base_dataset (Dataset): The base dataset to sample from.
            N (int): The number of images to construct into an observable sequence.
            R (int): number of repeats
            operators (list): list of operators from which to sample
            action to take on observations (str): can be 'global' to compute operator over full observations, or 'select_K', where K=integer.
        """
        dataset = load_dataset("imagenet-1k", split=which_split, trust_remote_code=True)

        self.transform = transform
        self.base_dataset = dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data_item = self.base_dataset[idx]
        image = self.transform(data_item["image"].convert("RGB"))
        target = data_item["label"]
        return image, target


class MazeImageFolder(ImageFolder):
    """
    A custom dataset class that extends the ImageFolder class.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)

    Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=Image.open,
        is_valid_file=None,
        which_set="train",
        augment_p=0.5,
        maze_route_length=10,
        trunc=False,
        expand_range=True,
    ):
        super(MazeImageFolder, self).__init__(
            root, transform, target_transform, loader, is_valid_file
        )
        self.which_set = which_set
        self.augment_p = augment_p
        self.maze_route_length = maze_route_length
        self.all_paths = {}
        self.trunc = trunc
        self.expand_range = expand_range

        self._preload()
        print("Solving all mazes...")
        for index in range(len(self.preloaded_samples)):
            path = self.get_solution(self.preloaded_samples[index])
            self.all_paths[index] = path

    def _preload(self):
        preloaded_samples = []
        with tqdm(
            total=self.__len__(), initial=0, leave=True, position=0, dynamic_ncols=True
        ) as pbar:

            for index in range(self.__len__()):
                pbar.set_description("Loading mazes")
                path, target = self.samples[index]
                sample = self.loader(path)
                sample = np.array(sample).astype(np.float32) / 255
                preloaded_samples.append(sample)
                pbar.update(1)
                if self.trunc and index == 999:
                    break
        self.preloaded_samples = preloaded_samples

    def __len__(self):
        if hasattr(self, "preloaded_samples") and self.preloaded_samples is not None:
            return len(self.preloaded_samples)
        else:
            return super().__len__()

    def get_solution(self, x):
        x = np.copy(x)
        # Find start (red) and end (green) pixel coordinates
        start_coords = np.argwhere((x == [1, 0, 0]).all(axis=2))
        end_coords = np.argwhere((x == [0, 1, 0]).all(axis=2))

        if len(start_coords) == 0 or len(end_coords) == 0:
            print("Start or end point not found.")
            return None

        start_y, start_x = start_coords[0]
        end_y, end_x = end_coords[0]

        current_y, current_x = start_y, start_x
        path = [4] * self.maze_route_length

        pi = 0
        while (current_y, current_x) != (end_y, end_x):
            next_y, next_x = -1, -1  # Initialize to invalid coordinates
            direction = -1  # Initialize to an invalid direction

            # Check Up
            if current_y > 0 and (
                (x[current_y - 1, current_x] == [0, 0, 1]).all()
                or (x[current_y - 1, current_x] == [0, 1, 0]).all()
            ):
                next_y, next_x = current_y - 1, current_x
                direction = 0

            # Check Down
            elif current_y < x.shape[0] - 1 and (
                (x[current_y + 1, current_x] == [0, 0, 1]).all()
                or (x[current_y + 1, current_x] == [0, 1, 0]).all()
            ):
                next_y, next_x = current_y + 1, current_x
                direction = 1

            # Check Left
            elif current_x > 0 and (
                (x[current_y, current_x - 1] == [0, 0, 1]).all()
                or (x[current_y, current_x - 1] == [0, 1, 0]).all()
            ):
                next_y, next_x = current_y, current_x - 1
                direction = 2

            # Check Right
            elif current_x < x.shape[1] - 1 and (
                (x[current_y, current_x + 1] == [0, 0, 1]).all()
                or (x[current_y, current_x + 1] == [0, 1, 0]).all()
            ):
                next_y, next_x = current_y, current_x + 1
                direction = 3

            path[pi] = direction
            pi += 1

            x[current_y, current_x] = [
                255,
                255,
                255,
            ]  # mark the current as white to avoid going in circles
            current_y, current_x = next_y, next_x
            if pi == len(path):
                break

        return np.array(path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        sample = np.copy(self.preloaded_samples[index])

        path = np.copy(self.all_paths[index])

        if self.which_set == "train":
            # Randomly rotate -90 or +90 degrees
            if random.random() < self.augment_p:
                which_rot = random.choice([-1, 1])
                sample = np.rot90(sample, k=which_rot, axes=(0, 1))
                for pi in range(len(path)):
                    if path[pi] == 0:
                        path[pi] = 3 if which_rot == -1 else 2
                    elif path[pi] == 1:
                        path[pi] = 2 if which_rot == -1 else 3
                    elif path[pi] == 2:
                        path[pi] = 0 if which_rot == -1 else 1
                    elif path[pi] == 3:
                        path[pi] = 1 if which_rot == -1 else 0

            # Random horizontal flip
            if random.random() < self.augment_p:
                sample = np.fliplr(sample)
                for pi in range(len(path)):
                    if path[pi] == 2:
                        path[pi] = 3
                    elif path[pi] == 3:
                        path[pi] = 2

            # Random vertical flip
            if random.random() < self.augment_p:
                sample = np.flipud(sample)
                for pi in range(len(path)):
                    if path[pi] == 0:
                        path[pi] = 1
                    elif path[pi] == 1:
                        path[pi] = 0

        sample = torch.from_numpy(np.copy(sample)).permute(2, 0, 1)

        blue_mask = (sample[0] == 0) & (sample[1] == 0) & (sample[2] == 1)

        sample[:, blue_mask] = 1
        target = path

        if not self.expand_range:
            return sample, target
        return (sample * 2) - 1, (target)


piet1_right_class = 0
piet1_up_class = 1
piet1_left_class = 2
piet1_down_class = 3

piet1_start_color = (0, 255, 0)
piet1_startDirection_color = (0, 255, 255)
piet1_left_color = (255, 255, 0)
piet1_right_color = (255, 0, 255)
piet1_end_color = (255, 0, 0)


class Piet1ImageFolder(ImageFolder):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        loader=Image.open,
        is_valid_file=None,
        which_set="train",
        augment_p=0.5,
        program_route_length=10,
        trunc=False,
        expand_range=True,
    ):
        super(Piet1ImageFolder, self).__init__(
            root, transform, target_transform, loader, is_valid_file
        )
        self.which_set = which_set
        self.augment_p = augment_p
        self.program_route_length = program_route_length
        self.all_paths = {}
        self.trunc = trunc
        self.expand_range = expand_range

        self._preload()
        print("Solving all programs...")
        for index in range(len(self.preloaded_samples)):
            path = self.get_solution(self.preloaded_samples[index])
            self.all_paths[index] = path

    def _preload(self):
        preloaded_samples = []
        preloaded_paths = []
        with tqdm(
            total=self.__len__(), initial=0, leave=True, position=0, dynamic_ncols=True
        ) as pbar:
            for index in range(self.__len__()):
                pbar.set_description("Loading programs")
                path, target = self.samples[index]
                sample = self.loader(path)
                sample = np.array(sample).astype(np.float32) / 255
                preloaded_samples.append(sample)

                # TODO: load path from path.replace(".png", ".pkl")
                preloaded_paths.append(TODO)

                pbar.update(1)
                if self.trunc and index == 999:
                    break
        self.preloaded_samples = preloaded_samples
        self.preloaded_paths = preloaded_paths

    def __len__(self):
        if hasattr(self, "preloaded_samples") and self.preloaded_samples is not None:
            return len(self.preloaded_samples)
        else:
            return super().__len__()

    def get_solution(self, x):
        """
        Solve a program
        """

        classes_count = 4
        """
        classification classes options count
        """

        """
        
        Notes:
        - "up" is negative y direction
        - "right" is position x direction
        - direction classes are in clockwise order, mod 4

        """

        x = np.copy(x)

        # Find important position

        start_pos: tuple[int, int] = tuple(
            np.argwhere((x == piet1_start_color).all(axis=2)).tolist()
        )
        assert len(start_pos) == 2, f"Invalid start: {start_pos}"

        startDirection_pos: tuple[int, int] = tuple(
            np.argwhere((x == piet1_startDirection_color).all(axis=2)).tolist()
        )
        assert (
            len(startDirection_pos) == 2
        ), f"Invalid startDirection: {startDirection_pos}"

        end_pos: tuple[int, int] = tuple(
            np.argwhere((x == piet1_end_color).all(axis=2)).tolist()
        )
        assert len(end_pos) == 2, f"Invalid end: {end_pos}"

        current_pos = start_pos

        def diff_pos(p1: tuple[int, int], p2: tuple[int, int]) -> tuple[int, int]:
            """
            p1 - p2
            """

            return (p1[0] - p2[0], p1[1] - p2[1])

        def at(p):
            return x[p[0], p[1]]

        # Determine starting direction

        startDirection_diff = diff_pos(startDirection_pos, start_pos)

        if startDirection_diff == (1, 0):
            current_dir = piet1_right_class
        elif startDirection_diff == (0, -1):
            current_dir = piet1_up_class
        elif startDirection_diff == (-1, 0):
            current_dir = piet1_left_class
        elif startDirection_diff == (0, 1):
            current_dir = piet1_right_class
        else:
            assert False, f"Invalid startDirectionDir: {startDirection_diff}"

        path = [classes_count] * self.program_route_length
        i = 0

        def turn_left():
            nonlocal current_dir
            current_dir = (current_dir + 1) % 4

        def turn_right():
            nonlocal current_dir
            current_dir = (current_dir - 1) % 4

        def update():
            """
            Updates state based on rules for current position's color.
            """

            current_color = at(current_pos)

            # check current position to update direction
            if current_color == piet1_left_color:
                turn_left()
            elif current_color == piet1_right_color:
                turn_right()
            else:
                # don't update direction
                pass

        def move():
            """
            Moves according to state, and records movement in path.
            """

            nonlocal current_pos

            path[i] = current_dir

            x, y = current_pos

            if current_dir == piet1_right_class:
                current_pos = (x + 1, y)
            elif current_dir == piet1_up_class:
                current_pos = (x, y - 1)
            elif current_dir == piet1_left_class:
                current_pos = (x - 1, y)
            elif current_dir == piet1_down_class:
                current_pos = (x, y + 1)
            else:
                assert False, f"Invalid current_dir: {current_dir}"

        while current_pos != end_pos:
            update()
            move()

            i += 1

            if i == len(path):
                break

        return np.array(path)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        sample = np.copy(self.preloaded_samples[index])

        path = np.copy(self.all_paths[index])

        if self.which_set == "train":
            # Randomly rotate -90 or +90 degrees
            if random.random() < self.augment_p:
                which_rot = random.choice([-1, 1])
                sample = np.rot90(sample, k=which_rot, axes=(0, 1))
                for pi in range(len(path)):
                    if path[pi] == 0:
                        path[pi] = 3 if which_rot == -1 else 2
                    elif path[pi] == 1:
                        path[pi] = 2 if which_rot == -1 else 3
                    elif path[pi] == 2:
                        path[pi] = 0 if which_rot == -1 else 1
                    elif path[pi] == 3:
                        path[pi] = 1 if which_rot == -1 else 0

            # Random horizontal flip
            if random.random() < self.augment_p:
                sample = np.fliplr(sample)
                for pi in range(len(path)):
                    if path[pi] == 2:
                        path[pi] = 3
                    elif path[pi] == 3:
                        path[pi] = 2

            # Random vertical flip
            if random.random() < self.augment_p:
                sample = np.flipud(sample)
                for pi in range(len(path)):
                    if path[pi] == 0:
                        path[pi] = 1
                    elif path[pi] == 1:
                        path[pi] = 0

        sample = torch.from_numpy(np.copy(sample)).permute(2, 0, 1)

        blue_mask = (sample[0] == 0) & (sample[1] == 0) & (sample[2] == 1)

        sample[:, blue_mask] = 1
        target = path

        if not self.expand_range:
            return sample, target

        return (sample * 2) - 1, (target)


class ParityDataset(Dataset):
    def __init__(self, sequence_length=64, length=100000):
        self.sequence_length = sequence_length
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        vector = 2 * torch.randint(0, 2, (self.sequence_length,)) - 1
        vector = vector.float()
        negatives = (vector == -1).to(torch.long)
        cumsum = torch.cumsum(negatives, dim=0)
        target = (cumsum % 2 != 0).to(torch.long)
        return vector, target
