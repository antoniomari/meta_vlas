"""Utils for evaluating policies in LIBERO simulation environments."""
import math
import os

import gym
import imageio
import numpy as np
import tensorflow as tf
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark.mu_creation import *  # noqa
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.bddl_base_domain import BDDLUtils
from libero.libero.envs.objects import get_object_dict
from libero.libero.utils import task_generation_utils
from libero.libero.utils.bddl_generation_utils import (
    get_object_dict as get_object_num_dict,
)
from libero.libero.utils.bddl_generation_utils import (
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import get_scene_class, get_scene_dict, register_mu
from libero.libero.utils.task_generation_utils import (
    generate_bddl_from_task_info,
    register_task_info,
)
from PIL import Image



def generate_mu_with_distractor_objects(mu_cls, min_distractors, max_distractors, distractor_seed):
    """Generate a version of the initial state distribution with distractor objects. The number and position of the
    distractor objects are determined by distractor_seed.
    Notes:
    1) In the context of an environment generated with mu_cls returned by this function, env.reset() will
    NOT randomize distractors. Instead, the distractors will be fixed for the duration of the environment's lifetime.
    If randomizing the distractors is desired, a new environment should be generated with a different distractor_seed.
    2) If one wishes to prevent correlations between the distractors chosen for different environments, different
    distractor_seed values should be used for the different environments.
    """

    assert max_distractors >= min_distractors

    large_objects = {
        "basket",
        "basin_faucet",
        "chefmate_8_frypan",
        "desk_caddy",
        "dining_set_group",
        "faucet",
        "flat_stove",
        "microwave",
        "rack",
        "short_cabinet",
        "short_fridge",
        "slide_cabinet",
        "white_cabinet",
        "white_storage_box",
        "window",
        "wine_rack",
        "wooden_cabinet",
        "wooden_shelf",
        "wooden_tray",
        "wooden_two_layer_shelf",
    }
    exclude_objects = large_objects.union({"cherries", "corn", "mayo", "salad_dressing", "target_zone"})

    rng = np.random.default_rng(distractor_seed)
    all_object_categories = set(get_object_dict().keys())
    mu = mu_cls()
    object_categories_already_used = []
    for category_name in mu.fixture_object_dict.keys():
        object_categories_already_used.append(category_name)
    for category_name in mu.movable_object_dict.keys():
        object_categories_already_used.append(category_name)
    object_categories_already_used = set(object_categories_already_used)

    remaining_objects = sorted(list(all_object_categories - object_categories_already_used - exclude_objects))

    # Choose a subset of objects to be distractors
    num_distractors = rng.integers(min_distractors, max_distractors + 1)
    selected_objects = list(rng.choice(remaining_objects, num_distractors, replace=False))

    print(
        f"Selected {num_distractors} distractor(s) from {len(remaining_objects)} remaining objects: {selected_objects}"
    )

    class MuWithDistractorObjects(mu_cls):
        def __init__(self, *args, **kwargs):

            # Temporarily set this to False so superclass can initialize properly
            self._include_distractor_in_init_states = False
            super().__init__(*args, **kwargs)
            self._include_distractor_in_init_states = True

            self.distractor_objects = selected_objects
            distractor_object_num_info = {category: 1 for category in self.distractor_objects}

            self.movable_object_dict.update(get_object_num_dict(distractor_object_num_info))
            self.define_regions()

        @property
        def init_states(self):
            if self._include_distractor_in_init_states:
                states = super().init_states
                for distractor_object_category in self.distractor_objects:
                    assert len(self.movable_object_dict[distractor_object_category]) == 1
                    object_name = self.movable_object_dict[distractor_object_category][0]
                    states.append(("On", object_name, f"{self.workspace_name}_{distractor_object_category}_init_region"))
                return states
            else:
                return super().init_states

        def define_regions(self):
            def is_point_in_ranges(x, y, ranges):
                for x1, y1, x2, y2 in ranges:
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        return True
                return False

            def sample_point_outside_ranges(ranges, xlim, ylim):
                i = 0
                while i < 10_000_000: ## TODO(YY): a bit dangerous- keep sampling until you find a valid point to place the distractor
                    x = rng.uniform(*xlim)
                    y = rng.uniform(*ylim)
                    if not is_point_in_ranges(x, y, ranges):
                        return x, y
                    i += 1
                    if i % 100_000 == 0:
                        print("still trying to sample point outside of ranges...")
                raise ValueError("Could not sample point outside of ranges")

            if self._include_distractor_in_init_states:
                super().define_regions()
                for distractor_object_category in self.distractor_objects:
                    assert len(self.movable_object_dict[distractor_object_category]) == 1

                    # Pick and x, y outside of the current regions
                    current_ranges = []
                    for region in self.xy_region_kwargs_list:
                        if "init_region" not in region["region_name"]:
                            continue
                        radius = 0.15
                        if any(object_category in region["region_name"] for object_category in large_objects):
                            radius = 0.25
                        x1, y1, x2, y2 = region["ranges"][0]
                        current_ranges.append((x1 - radius, y1 - radius, x2 + radius, y2 + radius))
                    x, y = sample_point_outside_ranges(current_ranges, xlim=(-0.3, 0.3), ylim=(-0.275, 0.275))
                    self.regions.update(
                        self.get_region_dict(
                            region_centroid_xy=[x, y],
                            region_name=f"{distractor_object_category}_init_region",
                            target_name=self.workspace_name,
                            region_half_len=0.00001,
                        )
                    )
                    self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

            else:
                super().define_regions()

    return MuWithDistractorObjects

# def generate_swapped_object_mu(mu_cls, swapping_objs_dict):
#     """Generate a version of the initial state distribution with swapped objects."""
#     class SwappedObjectMu(mu_cls):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)

#         @property
#         def init_states(self):
#             states =  super().init_states
#             new_states = []
#             for state in states:
#                 if state[1] in swapping_objs_dict.keys():
#                     new_obj = swapping_objs_dict[state[1]]
#                     without_suffix = "_".join(new_obj.split('_')[:-1])
#                     new_state = ("On", new_obj, f"{self.workspace_name}_{without_suffix}_init_region")
#                     new_states.append(new_state)
#                 else:
#                     new_states.append(state)
#             return new_states
#     return SwappedObjectMu

def generate_translated_mu(mu_cls, obj_of_interest, translation_scales_dict, translation_seed):
    """Generate a version of the initial state distribution with each object of interest translated by the corresponding amount in translation_list."""
    class TranslatedMu(mu_cls):
        def __init__(self, *args, **kwargs):
            self.obj_of_interest_to_region_map = {}
            super().__init__(*args, **kwargs)


        def define_regions(self):
            super().define_regions()
            # rng = np.random.default_rng(translation_seed)

            for obj, (translation_x, translation_y) in translation_scales_dict.items():
                for condition in self.init_states:
                    if condition[1] == obj and condition[0] == "On" and condition[2].endswith("init_region"):
                        # Get object's current initial state region
                        region_name = condition[2].replace(self.workspace_name + "_", "")
                        current_range = self.regions[region_name]["ranges"]
                        # Translate the region
                        assert len(current_range) == 1
                        x1, y1, x2, y2 = current_range[0]
                        width = x2 - x1
                        height = y2 - y1
                        # translation_x_mgtd = rng.uniform(min_translation_scale * width, max_translation_scale * width)
                        # translation_y_mgtd = rng.uniform(min_translation_scale * height, max_translation_scale * height)
                        # translation_x = translation_x_mgtd * rng.choice([-1, 1])
                        # translation_y = translation_y_mgtd * rng.choice([-1, 1])
                        new_range =  [
                            (
                            x1 + translation_x,
                            y1 + translation_y,
                            x2 + translation_x,
                            y2 + translation_y,
                            )
                        ]
                        # Overwrite the initial state region
                        self.regions[region_name]["ranges"] = new_range
                        self.obj_of_interest_to_region_map[obj] = new_range
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    return TranslatedMu



def generate_expanded_mu(mu_cls, expansion_obj_of_interest, expansion_half_len_factor, remove_train_distractors=False):
    """Generate a version of the initial state distribution with an expanded region for the objects of interest."""

    class ExpandedMu(mu_cls):
        def __init__(self, *args, **kwargs):
            self.obj_of_interest_to_region_map = {}
            super().__init__(*args, **kwargs)

        def define_regions(self):
            super().define_regions()

            # Iterate through objects of interest
            region_names = {}
            for obj in expansion_obj_of_interest:
                for condition in self.init_states:
                    if condition[1] == obj and condition[0] == "On" and condition[2].endswith("init_region"):
                        # Get object's current initial state region
                        region_name = condition[2].replace(self.workspace_name + "_", "")
                        region_names[obj] = region_name
                        current_range = self.regions[region_name]["ranges"]
                        # Expand the region
                        assert len(current_range) == 1
                        x1, y1, x2, y2 = current_range[0]
                        width = x2 - x1
                        height = y2 - y1
                        new_range = [
                            (
                                x1 - expansion_half_len_factor * width,
                                y1 - expansion_half_len_factor * height,
                                x2 + expansion_half_len_factor * width,
                                y2 + expansion_half_len_factor * height,
                            )
                        ]
                        # Overwrite the initial state region
                        self.regions[region_name]["ranges"] = new_range
                        self.obj_of_interest_to_region_map[obj] = new_range

            # if permute_objs_of_interest:

            #     obj_keys, region_values = list(self.obj_of_interest_to_region_map.keys()), list(self.obj_of_interest_to_region_map.values())
            #     while True:
            #         print('permututing where the objects of interest will land...')
            #         permuted_values = np.random.permutation(region_values)
            #         if not np.all(permuted_values == region_values):
            #             break
            #     self.obj_of_interest_to_region_map = dict(zip(obj_keys, permuted_values))

            #     for k in list(region_names.keys()):
            #         region_name = region_names[k]
            #         self.regions[region_name]["ranges"] = self.obj_of_interest_to_region_map[k]



            if remove_train_distractors:
                for region in list(self.regions.keys()):
                    obj_name = region.split('_init_region')[0]
                    obj_in_obj_of_interest = any(obj.startswith(obj_name) for obj in expansion_obj_of_interest)
                    if not obj_in_obj_of_interest:
                        # print(f"removing {obj_name}")
                        del self.regions[region]
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    return ExpandedMu


## TODO(YY): integrate this into debug OOD
def generate_expanded_mu_PERMUTE(mu_cls, expansion_obj_of_interest, expansion_half_len_factor, remove_train_distractors=False):
    """Generate a version of the initial state distribution with an expanded region for the objects of interest."""

    class ExpandedMu(mu_cls):
        def __init__(self, *args, **kwargs):
            self.obj_of_interest_to_region_map = {}
            super().__init__(*args, **kwargs)

        def define_regions(self):
            super().define_regions()

            # Iterate through objects of interest
            region_names = {}
            new_ranges_list = []
            for obj in expansion_obj_of_interest:
                for condition in self.init_states:
                    if condition[1] == obj and condition[0] == "On" and condition[2].endswith("init_region"):
                        # Get object's current initial state region
                        region_name = condition[2].replace(self.workspace_name + "_", "")
                        region_names[obj] = region_name
                        current_range = self.regions[region_name]["ranges"]
                        # Expand the region
                        assert len(current_range) == 1
                        x1, y1, x2, y2 = current_range[0]
                        width = x2 - x1
                        height = y2 - y1
                        new_range = [
                            (
                                x1 - expansion_half_len_factor * width,
                                y1 - expansion_half_len_factor * height,
                                x2 + expansion_half_len_factor * width,
                                y2 + expansion_half_len_factor * height,
                            )
                        ]
                        new_ranges_list.append(new_range)
                        # Overwrite the initial state region
                        # self.regions[region_name]["ranges"] = new_range
                        # self.obj_of_interest_to_region_map[obj] = new_range

            assert len(expansion_obj_of_interest) == 2 ## lil hack
            mapping = {0: 1, 1: 0} # 2-length permutation :)
            for i, obj in enumerate(expansion_obj_of_interest):
                for condition in self.init_states:
                    if condition[1] == obj and condition[0] == "On" and condition[2].endswith("init_region"):
                        region_name = condition[2].replace(self.workspace_name + "_", "")
                        self.regions[region_name]["ranges"] = new_ranges_list[mapping[i]]
                        self.obj_of_interest_to_region_map[obj] = new_ranges_list[mapping[i]]

            # if permute_objs_of_interest:

            #     obj_keys, region_values = list(self.obj_of_interest_to_region_map.keys()), list(self.obj_of_interest_to_region_map.values())
            #     while True:
            #         print('permututing where the objects of interest will land...')
            #         permuted_values = np.random.permutation(region_values)
            #         if not np.all(permuted_values == region_values):
            #             break
            #     self.obj_of_interest_to_region_map = dict(zip(obj_keys, permuted_values))

            #     for k in list(region_names.keys()):
            #         region_name = region_names[k]
            #         self.regions[region_name]["ranges"] = self.obj_of_interest_to_region_map[k]


            ## YY: treating "expansion_obj_of_interest" as all obj_of_interest
            if remove_train_distractors:
                for region in list(self.regions.keys()):
                    obj_name = region.split('_init_region')[0]
                    obj_in_obj_of_interest = any(obj.startswith(obj_name) for obj in expansion_obj_of_interest)
                    if not obj_in_obj_of_interest:
                        # print(f"removing {obj_name}")
                        del self.regions[region]
            self.xy_region_kwargs_list = get_xy_region_kwargs_list_from_regions_info(self.regions)

    return ExpandedMu


def edit_bddl_file_with_swap_objects(bddl_file, swap_objects_dict):
    with open(bddl_file, 'r') as file:
        content = file.read()
    modified_content = content
    for obj, swap_obj in swap_objects_dict.items():
        modified_content = modified_content.replace(obj, swap_obj)
    with open(bddl_file, 'w') as file:
        file.write(modified_content)




def generate_ood_init_wrapper(expanded_mu_cls, expansion_obj_of_interest, expansion_half_len_factor):
    """
    Given a class with expanded initial regions, generate an environment wrapper that ensures objects of interest
    are in the expanded region.
    """

    class OODInitWrapper(gym.Wrapper):
        global expanded_mu_cls

        def __init__(self, env):
            super().__init__(env)
            self.env = env
            mu = expanded_mu_cls()
            self.obj_of_interest_to_region_map = mu.obj_of_interest_to_region_map

        def reset(self, **kwargs):
            while True:
                print("Resampling objects and fixtures...")
                out = self.env.reset(**kwargs)
                for obj_name in expansion_obj_of_interest:
                    # Make sure at least one object of interest is in the expanded part of its initial region
                    obj_state = self.env.env.object_states_dict[obj_name]
                    obj_x, obj_y = obj_state.get_geom_state()["pos"][:2]
                    if obj_name not in self.obj_of_interest_to_region_map.keys():
                        # This is a rare case where the object of interest does not have
                        # an initial state region defined
                        print("Skipping", obj_name)
                        continue
                    init_range = self.obj_of_interest_to_region_map[obj_name]
                    x1_, y1_, x2_, y2_ = init_range[0]
                    width = (x2_ - x1_) / (1 + 2 * expansion_half_len_factor)
                    height = (y2_ - y1_) / (1 + 2 * expansion_half_len_factor)
                    x1, x2 = (x1_ + x2_) / 2 - width / 2, (x1_ + x2_) / 2 + width / 2
                    y1, y2 = (y1_ + y2_) / 2 - height / 2, (y1_ + y2_) / 2 + height / 2
                    if obj_x < x1 or obj_x > x2 or obj_y < y1 or obj_y > y2:
                        return out

    return OODInitWrapper

## TODO(YY): make sure train distractors are being removed correctly above!!
def get_expanded_libero_env(
    task, expansion_half_len_factor, ood_only, min_distractors, max_distractors, seed, distractor_seed, translation_seed, translation_scales_dict={}, swap_objects_dict={}, do_translation=False, permute_objs_of_interest=False, remove_train_distractors=False, resolution=256
):
    """
    Given a LIBERO task, generate an environment with expanded initial regions for the objects of interest
    by a factor of expansion_half_len_factor in each direction. If ood_only is True, *at least one* object
    of interest must be initialized in the expanded part of the region (as opposed to the entire expanded region).
    Distractor objects may be added by setting min_distractors > 0. The number and position of the distractor objects
    is determined for the environment's lifetime by distractor_seed (see comments to generate_mu_with_distractor_objects
    for more details).
    """
    bddl_files_default_path = get_libero_path("bddl_files")

    # Parse the bddl file for this task
    bddl_file = os.path.join(bddl_files_default_path, task.problem_folder, task.bddl_file)
    parsed = BDDLUtils.robosuite_parse_problem(bddl_file)

    # Get the scene name
    language = benchmark.grab_language_from_filename(task.bddl_file)
    scene_name = task.bddl_file.replace("_" + language.replace(" ", "_"), "").replace(".bddl", "")

    # Get the mu class
    mu_cls = get_scene_class(scene_name)
    mu_cls_name = mu_cls.__name__
    obj_of_interest = parsed["obj_of_interest"]
    goal_states = parsed["goal_state"]
    goal_states = [tuple(g) for g in goal_states]


    # Make a version of the class with expanded initial regions only for obj_of_interest
    # do translation before expansion (since translation is defined relative to object's size)
    new_mu_cls = mu_cls

    # if swap_objects_dict and False:
        # new_mu_cls = generate_swapped_object_mu(new_mu_cls, swap_objects_dict)

    # doing translation
    if do_translation:
        translation_scales_dict = translation_scales_dict ## TODO(YY): make this dict a param??
        new_mu_cls = generate_translated_mu(new_mu_cls, obj_of_interest, translation_scales_dict, translation_seed)

    # doing expansion
    if expansion_half_len_factor > 0:
        if permute_objs_of_interest:
            new_mu_cls = generate_expanded_mu_PERMUTE(new_mu_cls, obj_of_interest, expansion_half_len_factor, remove_train_distractors=remove_train_distractors)
        else:
            new_mu_cls = generate_expanded_mu(new_mu_cls, obj_of_interest, expansion_half_len_factor, remove_train_distractors=remove_train_distractors)

    # adding distractors
    if min_distractors > 0:
        new_mu_cls = generate_mu_with_distractor_objects(
            new_mu_cls, min_distractors, max_distractors, distractor_seed
        )


    new_mu_cls.__name__ = mu_cls_name + "Expanded"

    scene_dict = get_scene_dict()
    scene_type = [key for (key, value) in scene_dict.items() if mu_cls in value][0]  # noqa

    task_generation_utils.TASK_INFO = {}

    register_mu(scene_type=scene_type)(new_mu_cls)
    print("Registered", new_mu_cls.__name__)

    register_task_info(
        language=language,
        scene_name=scene_name + "_EXPANDED",
        objects_of_interest=obj_of_interest,
        goal_states=goal_states,
    )

    # Generate a temp bddl file for the expanded task
    generate_bddl_from_task_info(folder=f"/tmp/pddl_{seed}")
    ## to prevent race conditions, use the seed in the task name
    task_bddl_file = f"/tmp/pddl_{seed}/" + scene_name + "_EXPANDED" + "_" + task.language.replace(" ", "_") + ".bddl"

    if swap_objects_dict:
        edit_bddl_file_with_swap_objects(task_bddl_file, swap_objects_dict)

    task_description = task.language
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}


    print(f"Initializing OffScreenRenderEnv with args {env_args}")
    env = OffScreenRenderEnv(**env_args)
    print(f"Initialized OffScreenRenderEnv with args {env_args}")

    ## TODO: can't just edit raw ENV fields like this, breaks the underlying envrionment?
    # if remove_train_distractors:
    #     train_distractors = [x for x in parsed["objects"].keys() if x not in parsed["obj_of_interest"]]
    #     objects_dict = env.env.objects_dict
    #     for obj_name in list(objects_dict.keys()):
    #         if obj_name in train_distractors:
    #             del objects_dict[obj_name]

    env.seed(seed)  # Seed will affect object initial positions

    if ood_only and expansion_half_len_factor > 0: # only enforce "ood_only" if init regions were expanded, else we will get stuck in an infinite loop
        ood_init_wrapper_cls = generate_ood_init_wrapper(new_mu_cls, obj_of_interest, expansion_half_len_factor)
        env = ood_init_wrapper_cls(env)

    return env, task_description
