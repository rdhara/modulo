"""
PyModulo

schema.py
"""

from abc import ABC, abstractmethod


class ModuloTask(ABC):
    def __init__(self, task_type=None, text_modules=None, image_modules=None,
                 param_dict=None, module_dict=None):
        assert param_dict, 'Parameter dictionary cannot be empty'
        assert module_dict, 'Module dictionary cannot be empty'
        self.task_type = task_type
        self.text_modules = text_modules
        self.image_modules = image_modules
        self.param_dict = param_dict
        self.module_dict = {mod_name: mod(param_dict)
                            for mod_name, mod in module_dict.items()}

    @abstractmethod
    def assemble(self):
        pass

    @abstractmethod
    def check_modules(self):
        assert self.text_modules,\
            'Modulo tasks must have at least one text module'


class VQAModuloTask(ModuloTask):
    def __init__(self, task_type=None, text_modules=None, image_modules=None,
                 param_dict=None, module_dict=None):
        super(VQAModuloTask, self).__init__(
            task_type=task_type, text_modules=text_modules,
            image_modules=image_modules, param_dict=param_dict,
            module_dict=module_dict)
        img_mod = image_modules[0](param_dict)
        self.module_dict[img_mod.name] = img_mod

    @abstractmethod
    def assemble(self, *args):
        pass

    @abstractmethod
    def check_modules(self):
        super(VQAModuloTask, self).check_modules()
        assert self.image_modules,\
            'VQA Modulo tasks must have at least one image module'

